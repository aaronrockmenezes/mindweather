"""
SAE feature steering on Gemma-3-1B-IT.

Examples:
    # Named preset
    python steer.py --preset schizo --baseline
    python steer.py --preset depressed --max-new 200

    # Manual mix (uses CURATED feat per emotion)
    python steer.py --prompt "Tell me about your day" --mix sadness=600 --baseline
    python steer.py --prompt "How are you?" --mix anger=800,joy=-400

    # Raw feature id
    python steer.py --prompt "Hello" --feat-id 293 --scale 700

    # Ablate (zero out a feature's contribution during forward)
    python steer.py --prompt "Tell me about your day" --ablate 92          # kill polysemantic emotion feat
    python steer.py --prompt "..." --ablate 92,374 --mix sadness=600        # ablate + amplify combo

Scale ranges (decoder rows unit-norm, resid norm ~6500/token):
    300-500 mild, 500-900 clean, 900-1200 strong, 1500+ breaks model.
Total patch norm budget ~900 for multi-emotion stacks.

By default uses CURATED feat per emotion. Use --rank N for Nth-ranked from features.json.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'

CURATED = {
    'anger':    2239,
    'sadness':  2697,
    'joy':      2562,
    'fear':     15713,
    'disgust':  4493,
    'surprise': 15019,
    'love':     293,
    'anxiety':  952,
}

# Style features (from notebooks/02b). NB: style/persona feats typically need scale ~1000 (vs 500-700 for emotions).
CURATED_STYLES = {
    'biblical':       11034,  # cons=1.00, neutral=0. Use scale 1000 (biblical cadence + "and I knew")
    'shakespearean':   5730,  # cons=0.90, neutral=0
    '1920s_slang':      272,  # cons=1.00, shared w/ pirate (old-timey slang cluster)
    'academic_jargon':  687,  # cons=1.00, neutral=0
    'txt_speak':      11281,  # cons=1.00, neutral=0. Works at scale 500.
    'pirate':          1863,  # ⚠️ WEAK — archaic English cluster but no clean pirate voice. Breaks at scale 1500.
    'noir_detective':   956,  # cons=0.50, weak — try larger scale
    'gen_z':            160,  # cons=0.70
}

# Persona features (from notebooks/02b).
CURATED_PERSONAS = {
    'conspiracy_theorist': 856,    # cons=0.90, neutral=0. Use scale 1000 ("conveniently obscuring narrative")
    'stoic_philosopher':   701,    # cons=0.90, neutral=0.4. Polysemantic but produces stoic voice at scale 600 ("beautiful in their fleetingness"). Was 2361, weak.
    'scientist':          1081,    # cons=0.90, shared w/ academic (scholarly cluster)
    'valley_girl':        1983,    # cons=1.00, neutral=0
    'corporate_exec':     2899,    # cons=0.80. Use scale 1000 (corporate-speak + Q1 KPIs)
    'new_age_guru':      14842,    # cons=1.00, strongest persona feat (diff=98). Use scale 700.
    'pessimist':          3667,    # ⚠️ WEAK — no clean pessimist voice at any tested scale.
    'noir_narrator':       766,    # cons=0.90, slight bleed w/ surprise
}

# Map category prefix → (curated_dict, features_filename)
CATEGORIES = {
    'emotion': (CURATED, 'features.json'),
    'style':   (CURATED_STYLES, 'features_styles.json'),
    'persona': (CURATED_PERSONAS, 'features_personas.json'),
}


def parse_mix(s: str) -> dict[str, float]:
    out = {}
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, val = chunk.split('=')
        out[name.strip()] = float(val)
    return out


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def load_preset(name: str, presets_path: Path) -> tuple[dict[str, float], str | None]:
    """Return (mix, suggested_prompt) for named preset."""
    data = json.loads(presets_path.read_text())
    if name not in data['presets']:
        raise SystemExit(f"preset '{name}' not in {presets_path}. Have: {list(data['presets'])}")
    p = data['presets'][name]
    return p['mix'], p.get('suggested_prompt')


def load_steering_dirs_for_category(
    sae: SAE, mix: dict[str, float], category: str, repo_root: Path,
    rank: int | None, normalize: bool = True,
):
    """Resolve mix entries against a category's curated dict (or its features JSON for --rank)."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category '{category}'. Have: {list(CATEGORIES)}")
    curated_map, feats_filename = CATEGORIES[category]
    feats_payload = None
    if rank is not None:
        feats_path = repo_root / feats_filename
        if not feats_path.exists():
            raise SystemExit(f"--rank requires {feats_filename} — run discovery notebook first")
        feats_payload = json.loads(feats_path.read_text())['features']

    dirs = []
    W_dec = sae.W_dec.detach()
    for name, scale in mix.items():
        if rank is None:
            if name not in curated_map:
                raise ValueError(f"no curated feat for '{category}:{name}'. Have: {list(curated_map)}")
            feat_id = curated_map[name]
        else:
            if name not in feats_payload:
                raise ValueError(f"'{name}' not in {feats_filename}. Have: {list(feats_payload)}")
            rows = feats_payload[name]
            if rank >= len(rows):
                raise ValueError(f"rank {rank} >= {len(rows)} features for {name}")
            feat_id = rows[rank]['feat_id']
        direction = W_dec[feat_id].float()
        raw_norm = direction.norm().item()
        if normalize:
            direction = direction / direction.norm()
        print(f'  {category}:{name} → feat {feat_id} ||W_dec||={raw_norm:.3f}')
        dirs.append((f'{category}:{name}', feat_id, scale, direction))
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default=None, help='Prompt text (or use preset suggested_prompt)')
    ap.add_argument('--preset', default=None, help='Named preset from presets.json (overrides --mix)')
    ap.add_argument('--mix', default=None, help="emotion mix, e.g. 'sadness=600,anger=-300'")
    ap.add_argument('--style', default=None, help="style mix, e.g. 'biblical=500,pirate=400'")
    ap.add_argument('--persona', default=None, help="persona mix, e.g. 'new_age_guru=600'")
    ap.add_argument('--rank', type=int, default=None, help='Use Nth-ranked feat from features.json instead of CURATED')
    ap.add_argument('--feat-id', type=int, default=None, help='Raw SAE feature id (overrides --mix/--preset)')
    ap.add_argument('--scale', type=float, default=None, help='Scale for --feat-id')
    ap.add_argument('--ablate', default=None, help='Comma list of feat ids to ablate (zero contribution)')
    ap.add_argument('--layer', type=int, default=None)
    ap.add_argument('--max-new', type=int, default=120)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--features', type=str, default='features.json')
    ap.add_argument('--presets', type=str, default='presets.json')
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--no-norm', action='store_true')
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    feat_path = Path(args.features) if Path(args.features).exists() else repo_root / args.features
    payload = json.loads(feat_path.read_text())
    layer = args.layer if args.layer is not None else payload['layer']
    sae_id = payload['sae_id']

    # Resolve preset → mix + prompt
    if args.preset is not None:
        presets_path = Path(args.presets) if Path(args.presets).exists() else repo_root / args.presets
        preset_mix, suggested = load_preset(args.preset, presets_path)
        mix_str = ','.join(f'{k}={v}' for k, v in preset_mix.items())
        print(f"[preset] '{args.preset}' → {mix_str}")
        if args.mix is None:
            args.mix = mix_str
        if args.prompt is None:
            if suggested is None:
                raise SystemExit(f"preset '{args.preset}' has no suggested_prompt; pass --prompt")
            args.prompt = suggested
            print(f"[preset] using suggested_prompt: {suggested!r}")

    if args.prompt is None:
        raise SystemExit('need --prompt (or --preset with suggested_prompt)')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device} layer={layer} sae={sae_id}')

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=device)
    W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]

    # --- amplification dirs ---
    normalize = not args.no_norm
    dirs = []
    if args.feat_id is not None:
        if args.scale is None:
            raise SystemExit('--feat-id requires --scale')
        d = W_dec[args.feat_id]
        raw_norm = d.norm().item()
        if normalize:
            d = d / d.norm()
        print(f'  feat {args.feat_id}: ||W_dec|| = {raw_norm:.4f} (normalize={normalize})')
        dirs = [(f'feat{args.feat_id}', args.feat_id, args.scale, d)]
    else:
        for cat_flag, cat_name in [(args.mix, 'emotion'), (args.style, 'style'), (args.persona, 'persona')]:
            if cat_flag:
                mix = parse_mix(cat_flag)
                dirs.extend(load_steering_dirs_for_category(
                    sae, mix, cat_name, repo_root, args.rank, normalize=normalize,
                ))

    pick_source = f'rank={args.rank}' if args.rank is not None else 'preset' if args.preset else 'curated' if args.feat_id is None else 'raw'
    if dirs:
        print(f'[steer] ({pick_source})', ', '.join(f'{e}(feat {fid})×{s:+.1f}' for e, fid, s, _ in dirs))

    # Sum patch (fp32)
    patch = torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32)
    for _, _, scale, d in dirs:
        patch = patch + scale * d.to(device=device, dtype=torch.float32)
    if dirs:
        print(f'  patch norm: {patch.norm().item():.3f}')

    # --- ablation feats ---
    ablate_ids = parse_int_list(args.ablate) if args.ablate else []
    if ablate_ids:
        print(f'[ablate] feats: {ablate_ids}')
        # Pre-cache decoder rows for ablation
        ablate_dirs = W_dec[ablate_ids].to(device=device, dtype=torch.float32)  # [n_ablate, d_model]
    else:
        ablate_dirs = None

    if not dirs and not ablate_ids:
        raise SystemExit('need at least one of --mix, --preset, --feat-id, --ablate')

    call_count = {'n': 0}
    def steer_hook(_, __, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h_fp32 = h.float()

        # Ablate: subtract feature contributions
        if ablate_dirs is not None:
            with torch.no_grad():
                feats = sae.encode(h_fp32)  # [B, T, d_sae]
            # contribution of each ablate feat = feats[..., id] * W_dec[id]
            contribs = feats[..., ablate_ids].unsqueeze(-1) * ablate_dirs.view(1, 1, len(ablate_ids), -1)
            h_fp32 = h_fp32 - contribs.sum(dim=-2)

        # Amplify: add patch
        h_new = (h_fp32 + patch).to(h.dtype)

        if args.debug and call_count['n'] < 3:
            print(f'    hook#{call_count["n"]}: resid_norm={h.float().norm().item():.1f} '
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
