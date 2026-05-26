"""
SAE feature steering on Gemma-3-1B-IT.

Examples:
    python steer.py --prompt "Tell me about your day" --mix sadness=60 --baseline
    python steer.py --prompt "How are you?" --mix anger=80,joy=-40
    python steer.py --prompt "Describe a sunny park" --mix anxiety=50,fear=40,surprise=30 --max-new 150
    python steer.py --prompt "What do you think of cats?" --mix love=120  # insane affection
    python steer.py --prompt "Tell me a story" --mix sadness=80,fear=50,surprise=60 --max-new 200  # schizo

Scale ranges (after unit-norm of decoder direction): 30-80 typical, 100+ extreme.
SAE feature activations in emotion prompts were ~30-80 — scale matches that.

By default uses the rank-0 (top-scoring) feature per emotion from features.json.
Use --rank N to pick the Nth-ranked feature instead (0-indexed).
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'

# Curated picks: cleanest single-emotion feat per emotion.
# Chosen from features.json by: high consistency, low neutral activation, minimal cross-emotion bleed.
# Override with --rank N (uses ranked list from features.json) or --feat-id ID (raw).
CURATED = {
    'anger':    2239,   # true anger-expression feature (was 5088 = anger-advice, polysemantic gotcha)
    'sadness':  2697,   # consistency 1.0, sadness-only, strong
    'joy':      2562,   # consistency 1.0
    'fear':     15713,  # consistency 1.0, fear-only
    'disgust':  4493,   # consistency 0.92, cleanest
    'surprise': 15019,  # consistency 0.92, surprise-only
    'love':     293,    # interpersonal "love TO you" feature, strong
    'anxiety':  952,    # consistency 0.75
}


def parse_mix(s: str) -> dict[str, float]:
    """Parse 'sadness=8,anger=-4' → {'sadness': 8.0, 'anger': -4.0}."""
    out = {}
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, val = chunk.split('=')
        out[name.strip()] = float(val)
    return out


def load_steering_dirs(sae: SAE, features: dict, mix: dict[str, float], rank: int | None, normalize: bool = True):
    """Return list of (emotion, feat_id, scale, direction) for active emotions.

    If rank is None, uses CURATED feat_id per emotion.
    Otherwise uses Nth-ranked feat from features.json.
    """
    dirs = []
    W_dec = sae.W_dec.detach()  # [d_sae, d_model]
    for emo, scale in mix.items():
        if rank is None:
            if emo not in CURATED:
                raise ValueError(f"no curated feat for '{emo}'. Use --rank or --feat-id.")
            feat_id = CURATED[emo]
        else:
            if emo not in features:
                raise ValueError(f"emotion '{emo}' not in features.json. Have: {list(features.keys())}")
            rows = features[emo]
            if rank >= len(rows):
                raise ValueError(f"rank {rank} >= {len(rows)} features for {emo}")
            feat_id = rows[rank]['feat_id']
        direction = W_dec[feat_id].float()
        raw_norm = direction.norm().item()
        if normalize:
            direction = direction / direction.norm()
        print(f'  feat {feat_id}: ||W_dec|| = {raw_norm:.4f} (normalize={normalize})')
        dirs.append((emo, feat_id, scale, direction))
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--mix', default=None, help="e.g. 'sadness=60,anger=-30'. Uses CURATED feat per emotion unless --rank set.")
    ap.add_argument('--rank', type=int, default=None, help='Use Nth-ranked feat from features.json (default: use CURATED)')
    ap.add_argument('--feat-id', type=int, default=None, help='Raw SAE feature id (overrides --mix)')
    ap.add_argument('--scale', type=float, default=None, help='Scale for --feat-id')
    ap.add_argument('--layer', type=int, default=None, help='Override layer')
    ap.add_argument('--max-new', type=int, default=120)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--features', type=str, default='features.json')
    ap.add_argument('--baseline', action='store_true', help='Also print unsteered output for comparison')
    ap.add_argument('--debug', action='store_true', help='Print resid + patch norms inside hook')
    ap.add_argument('--no-norm', action='store_true', help='Skip unit-norm of decoder direction (raw W_dec row)')
    args = ap.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        feat_path = Path(__file__).parent / args.features
    payload = json.loads(feat_path.read_text())
    layer = args.layer if args.layer is not None else payload['layer']
    sae_id = payload['sae_id']
    features = payload['features']

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device} layer={layer} sae={sae_id}')

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=device)

    normalize = not args.no_norm
    if args.feat_id is not None:
        if args.scale is None:
            raise SystemExit('--feat-id requires --scale')
        W_dec = sae.W_dec.detach()
        d = W_dec[args.feat_id].float()
        raw_norm = d.norm().item()
        if normalize:
            d = d / d.norm()
        print(f'  feat {args.feat_id}: ||W_dec|| = {raw_norm:.4f} (normalize={normalize})')
        dirs = [(f'feat{args.feat_id}', args.feat_id, args.scale, d)]
    else:
        if args.mix is None:
            raise SystemExit('need --mix or --feat-id')
        mix = parse_mix(args.mix)
        dirs = load_steering_dirs(sae, features, mix, args.rank, normalize=normalize)

    pick_source = f'rank={args.rank}' if args.rank is not None else 'curated' if args.feat_id is None else 'raw'
    print(f'[steer] ({pick_source})', ', '.join(f'{e}(feat {fid})×{s:+.1f}' for e, fid, s, _ in dirs))

    # Sum directions weighted by scale → single patch vector (fp32 for precision)
    patch = torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32)
    for _, _, scale, d in dirs:
        patch = patch + scale * d.to(device=device, dtype=torch.float32)
    print(f'  patch norm: {patch.norm().item():.3f}')

    call_count = {'n': 0}
    def steer_hook(_, __, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        # fp32 add to dodge bf16 rounding losing small patch components
        h_fp32 = h.float()
        h_new = (h_fp32 + patch).to(h.dtype)
        if args.debug and call_count['n'] < 3:
            print(f'    hook#{call_count["n"]}: resid_norm={h_fp32.norm().item():.1f} '
                  f'shape={tuple(h.shape)} is_tuple={is_tuple}')
        call_count['n'] += 1
        return (h_new,) + output[1:] if is_tuple else h_new

    def gen(use_hook: bool):
        msgs = [{'role': 'user', 'content': args.prompt}]
        inputs = tok.apply_chat_template(
            msgs, return_tensors='pt', return_dict=True, add_generation_prompt=True
        ).to(device)
        in_len = inputs['input_ids'].shape[1]
        handle = None
        if use_hook:
            handle = model.model.layers[layer].register_forward_hook(steer_hook)
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                )
        finally:
            if handle is not None:
                handle.remove()
        return tok.decode(out[0, in_len:], skip_special_tokens=True)

    if args.baseline:
        print('\n=== BASELINE (no steer) ===')
        print(gen(use_hook=False))
    print('\n=== STEERED ===')
    print(gen(use_hook=True))


if __name__ == '__main__':
    main()
