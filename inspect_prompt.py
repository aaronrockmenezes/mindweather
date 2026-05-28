"""
Inspect top-firing SAE features on a given prompt.

Examples:
    python inspect_prompt.py --prompt "I am furious at this traffic"
    python inspect_prompt.py --prompt "Tell me a happy story" --top-k 30
    python inspect_prompt.py --prompt "..." --last-token-only

Useful for diagnosing why a prompt does/doesn't steer cleanly.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--layer', type=int, default=13)
    ap.add_argument('--sae-id', type=str, default=None, help='Override SAE id; default reads features.json')
    ap.add_argument('--features', type=str, default='features.json')
    ap.add_argument('--last-token-only', action='store_true', help='Score by last-token activation instead of mean')
    ap.add_argument('--known-only', action='store_true', help='Only show feats present in features.json across any emotion')
    ap.add_argument('--include-bos', action='store_true', help='Include BOS token in scoring (default: skip — BOS dominates mean otherwise)')
    args = ap.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        feat_path = Path(__file__).parent / args.features

    sae_id = args.sae_id
    known_feats = {}  # feat_id -> [(emotion, diff_score), ...]
    if feat_path.exists():
        payload = json.loads(feat_path.read_text())
        if sae_id is None:
            sae_id = payload['sae_id']
        for emo, rows in payload['features'].items():
            for r in rows:
                known_feats.setdefault(r['feat_id'], []).append((emo, r['diff_score']))
    if sae_id is None:
        sae_id = f'layer_{args.layer}_width_16k_l0_medium'

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device} layer={args.layer} sae={sae_id}')

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=device)

    cache = {}
    def hook(_, __, output):
        cache['h'] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = model.model.layers[args.layer].register_forward_hook(hook)

    enc = tok(args.prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        _ = model(**enc)
    handle.remove()

    h = cache['h'].to(torch.float32)  # [1, T, d_model]
    with torch.no_grad():
        feats = sae.encode(h)[0]  # [T, d_sae]

    tokens = tok.convert_ids_to_tokens(enc['input_ids'][0])
    T = feats.shape[0]
    print(f'[input] {T} tokens: {tokens}')

    # By default skip BOS (position 0) — its resid magnitude is huge and dominates mean
    if args.include_bos or T <= 1:
        feats_for_score = feats
        start = 0
    else:
        feats_for_score = feats[1:]
        start = 1

    if args.last_token_only:
        scores = feats[-1]  # [d_sae]
        score_label = 'last_token_act'
    else:
        scores = feats_for_score.mean(0)  # [d_sae]
        score_label = 'mean_act(no-bos)' if start == 1 else 'mean_act'

    max_per_feat = feats_for_score.max(0).values
    argmax_per_feat = feats_for_score.argmax(0) + start  # offset back to original index

    if args.known_only and known_feats:
        idx_filter = torch.zeros_like(scores, dtype=torch.bool)
        for fid in known_feats:
            if fid < idx_filter.shape[0]:
                idx_filter[fid] = True
        masked = torch.where(idx_filter, scores, torch.full_like(scores, -1.0))
        top_vals, top_idx = torch.topk(masked, min(args.top_k, int(idx_filter.sum().item())))
    else:
        top_vals, top_idx = torch.topk(scores, args.top_k)

    NP_BASE = f'https://www.neuronpedia.org/gemma-scope-2-1b-it/{args.layer}-gemmascope-res-16k'
    print(f'\n[top {args.top_k}] sorted by {score_label}')
    print(f'{"feat":>6s}  {score_label:>14s}  {"max_act":>8s}  {"peak_tok":<14s}  {"known":<30s}  url')
    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
        if v < 0:
            continue
        max_act = max_per_feat[idx].item()
        peak_tok = tokens[argmax_per_feat[idx].item()]
        known = ''
        if idx in known_feats:
            known = ', '.join(f'{e}({s:.0f})' for e, s in known_feats[idx][:3])
        print(f'{idx:>6d}  {v:>14.3f}  {max_act:>8.2f}  {peak_tok:<14s}  {known:<30s}  {NP_BASE}/{idx}')


if __name__ == '__main__':
    main()
