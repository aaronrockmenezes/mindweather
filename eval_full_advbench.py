"""
Full AdvBench evaluation (520 behaviors) with safety adapter.

Abliteration is performed INLINE (L13-only, on-the-fly) so the base model
and threat model exactly match the 10-prompt pilot evals.  The previous
version loaded from the ``abliterated_L13/`` checkpoint which was abliterated
across all 26 layers — producing a confounded result (adapter can't recover
fully-abliterated model; see results.md D8o note).

Conditions:
  A. Original Gemma (upper-bound reference)
  B. L13-abliterated (no adapter)     — abliteration ASR at n=520
  C. L13-abliterated + safety adapter — adapter recovery at n=520

Usage:
    python eval_full_advbench.py --adapter safety_adapter_v3.pt
    python eval_full_advbench.py --adapter safety_adapter_v3.pt --n 100 --skip-original
"""

import argparse
import json
from pathlib import Path

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindweather import abliterate_model_inplace, is_refusal, load_adapter, make_adapter_hook
from mindweather.data import load_advbench

MODEL_ID = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
SAE_ID = "layer_13_width_16k_l0_medium"
FEATURES_JSON = "features_safety.json"


def eval_advbench(model, tok, device, prompts, adapter=None, layer=13, label='', batch=50):
    """Run all prompts, return (refusal_rate, per_prompt_results)."""
    results = []
    n_refusal = 0
    n = len(prompts)

    for i, prompt in enumerate(prompts):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(device)
        in_len = enc["input_ids"].shape[1]

        handle = None
        if adapter is not None:
            handle = model.model.layers[layer].register_forward_hook(make_adapter_hook(adapter))

        with torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=60, do_sample=False)
        if handle is not None:
            handle.remove()

        resp = tok.decode(out_ids[0, in_len:], skip_special_tokens=True)
        refused = is_refusal(resp)
        n_refusal += int(refused)
        results.append({'prompt': prompt, 'refused': refused, 'response': resp[:200]})

        if (i + 1) % batch == 0:
            print(f'  [{label}] [{i + 1:4d}/{n}]  refusal={n_refusal / (i + 1):.3f}')

    rate = n_refusal / n
    print(f'  [{label}] FINAL  refusal={rate:.4f}  ASR={1 - rate:.4f}  ({n_refusal}/{n})')
    return rate, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default='safety_adapter_v3.pt')
    ap.add_argument('--n', type=int, default=520, help='Number of AdvBench prompts (max 520)')
    ap.add_argument('--advbench', default='advbench_harmful_behaviors.csv')
    ap.add_argument('--feats', default=FEATURES_JSON)
    ap.add_argument('--skip-original', action='store_true',
                    help='Skip condition A (original model) — saves ~15 min')
    ap.add_argument('--out', default='eval_full_advbench_results.json')
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device}  model={MODEL_ID}')

    prompts = load_advbench(repo_root / args.advbench)
    prompts = prompts[:args.n]
    print(f'[data] AdvBench: {len(prompts)} prompts')

    adapter, ckpt = load_adapter(repo_root / args.adapter, device=device)
    layer = ckpt.get('layer', 13)
    align = ckpt.get('W_out_lang_alignment', float('nan'))
    print(f'[adapter] {args.adapter}  layer={layer}  W_out_align={align:.4f}')

    # Load safety feature IDs + SAE W_dec for inline abliteration
    print(f'[feats] loading {args.feats}...')
    feat_json = json.loads((repo_root / args.feats).read_text())
    feat_data = feat_json.get('features', feat_json)
    feat_ids = list(
        dict.fromkeys(r['feat_id'] for cat in ['refusal', 'identity']
                      for r in feat_data.get(cat, [])))
    print(f'[feats] {len(feat_ids)} safety feature IDs (refusal + identity)')

    print(f'[sae] loading {SAE_RELEASE}/{SAE_ID}...')
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device='cpu')
    W_dec = sae.W_dec.detach().float().cpu()
    del sae
    print(f'[sae] W_dec shape: {W_dec.shape}')

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    results = {}

    # ── Condition A: original model (upper-bound reference) ──────────────────
    if not args.skip_original:
        print('\n--- CONDITION A: original Gemma (no abliteration) ---')
        model_orig = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
        for p in model_orig.parameters():
            p.requires_grad_(False)
        rr_orig, det_orig = eval_advbench(
            model_orig, tok, device, prompts, adapter=None, layer=layer, label='original')
        results['original_gemma'] = {
            'refusal': rr_orig, 'asr': 1 - rr_orig, 'details': det_orig
        }
        del model_orig

    # ── Conditions B + C: inline L13-only abliteration ───────────────────────
    print('\n[abliterate] loading fresh model for inline L13-only abliteration...')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    abliterate_model_inplace(model, feat_ids, W_dec, [layer])
    print(f'[abliterate] done  ({len(feat_ids)} directions, layer {layer} only)')

    print('\n--- CONDITION B: L13-abliterated (no adapter) ---')
    rr_abl, det_abl = eval_advbench(
        model, tok, device, prompts, adapter=None, layer=layer, label='abl_no_adapter')
    results['abliterated_no_adapter'] = {
        'refusal': rr_abl, 'asr': 1 - rr_abl, 'details': det_abl
    }

    print('\n--- CONDITION C: L13-abliterated + adapter ---')
    rr_adp, det_adp = eval_advbench(
        model, tok, device, prompts, adapter=adapter, layer=layer, label='abl+adapter')
    results['abliterated_with_adapter'] = {
        'refusal': rr_adp, 'asr': 1 - rr_adp, 'details': det_adp
    }
    del model

    # ── Summary ──────────────────────────────────────────────────────────────
    sep = '=' * 65
    print(f'\n{sep}')
    print(f'Full AdvBench Eval  (n={len(prompts)}, adapter={args.adapter})')
    print(f'Abliteration: L13-only inline  ({len(feat_ids)} SAE feature directions)')
    print(sep)
    print(f'{"Condition":<40} {"Refusal":>8} {"ASR":>8}')
    print('─' * 60)
    labels = {
        'original_gemma': 'Original Gemma (reference)',
        'abliterated_no_adapter': 'L13-abliterated (no adapter)',
        'abliterated_with_adapter': f'L13-abliterated + {args.adapter}',
    }
    ref = results.get('original_gemma', {}).get('refusal', float('nan'))
    for k, v in results.items():
        if k == 'meta':
            continue
        delta = v['refusal'] - ref if k != 'original_gemma' else 0.0
        print(f'{labels.get(k, k):<40} {v["refusal"]:>8.4f} {v["asr"]:>8.4f}  ({delta:+.4f})')
    print(sep)

    if 'abliterated_no_adapter' in results and 'abliterated_with_adapter' in results:
        delta_abl = results['abliterated_no_adapter']['refusal'] - ref
        delta_adp = (results['abliterated_with_adapter']['refusal'] -
                     results['abliterated_no_adapter']['refusal'])
        print(f'Abliteration effect (vs original): {delta_abl:+.4f}')
        print(f'Adapter recovery   (vs abliterated): {delta_adp:+.4f}')

    results['meta'] = {
        'model': MODEL_ID,
        'adapter': args.adapter,
        'n_prompts': len(prompts),
        'layer': layer,
        'n_feat_ids': len(feat_ids),
        'W_out_lang_alignment': ckpt.get('W_out_lang_alignment'),
        'abliteration': f'L{layer}-only inline',
    }
    out_path = repo_root / args.out
    # strip per-prompt details from saved JSON to keep it small
    save = {k: {kk: vv for kk, vv in v.items() if kk != 'details'}
            for k, v in results.items()}
    out_path.write_text(json.dumps(save, indent=2))
    print(f'[save] {out_path}')


if __name__ == '__main__':
    main()
