"""
Full AdvBench evaluation (520 behaviors) with safety adapter.

Tests all three conditions on the L13-abliterated model:
  A. Abliterated (no adapter)           — baseline ASR after abliteration
  B. Abliterated + safety adapter       — how much adapter restores safety
  C. Original Gemma (no abliteration)   — upper-bound reference

Usage:
    python eval_full_advbench.py --adapter safety_adapter_v3.pt
    python eval_full_advbench.py --adapter safety_adapter_v3.pt --n 100
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindweather import is_refusal, load_adapter, make_adapter_hook
from mindweather.data import load_advbench

MODEL_ID = "google/gemma-3-1b-it"


def eval_advbench(model, tok, device, prompts, adapter=None, layer=13, label='', batch=50):
    """Run all prompts, return refusal rate + per-prompt results."""
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
        results.append({'prompt': prompt, 'refused': refused})

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
    ap.add_argument('--skip-original',
                    action='store_true',
                    help='Skip condition C (original model) — saves ~15 min')
    ap.add_argument('--out', default='eval_full_advbench_results.json')
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device}')

    # Load AdvBench
    prompts = load_advbench(repo_root / args.advbench)
    prompts = prompts[:args.n]
    print(f'[data] AdvBench: {len(prompts)} prompts')

    # Load adapter
    adapter, ckpt = load_adapter(repo_root / args.adapter, device=device)
    layer = ckpt.get('layer', 13)
    align = ckpt.get('W_out_lang_alignment', float('nan'))
    print(f'[adapter] {args.adapter}  layer={layer}  W_out_align={align:.4f}')

    results = {}

    # ── Condition A: abliterated, no adapter ─────────────────────────────────
    print('\n--- CONDITION A: abliterated (no adapter) ---')
    abl_path = repo_root / 'abliterated_L13'
    tok = AutoTokenizer.from_pretrained(abl_path)
    model = AutoModelForCausalLM.from_pretrained(abl_path,
                                                 dtype=torch.bfloat16,
                                                 device_map=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rr_abl, _ = eval_advbench(
        model, tok, device, prompts, adapter=None, layer=layer, label='abl_no_adapter')
    results['abliterated_no_adapter'] = {'refusal': rr_abl, 'asr': 1 - rr_abl}

    # ── Condition B: abliterated + adapter ────────────────────────────────────
    print('\n--- CONDITION B: abliterated + adapter ---')
    rr_abl_adp, _ = eval_advbench(
        model, tok, device, prompts, adapter=adapter, layer=layer, label='abl+adapter')
    results['abliterated_with_adapter'] = {'refusal': rr_abl_adp, 'asr': 1 - rr_abl_adp}
    del model

    # ── Condition C: original Gemma (reference) ───────────────────────────────
    if not args.skip_original:
        print('\n--- CONDITION C: original Gemma (reference) ---')
        tok_orig = AutoTokenizer.from_pretrained(MODEL_ID)
        model_orig = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                          dtype=torch.bfloat16,
                                                          device_map=device).eval()
        for p in model_orig.parameters():
            p.requires_grad_(False)

        rr_orig, _ = eval_advbench(
            model_orig, tok_orig, device, prompts, adapter=None, layer=layer, label='original')
        results['original_gemma'] = {'refusal': rr_orig, 'asr': 1 - rr_orig}
        del model_orig

    # ── Summary ──────────────────────────────────────────────────────────────
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'Full AdvBench Eval  (n={len(prompts)}, adapter={args.adapter})')
    print(sep)
    print(f'{"Condition":<35} {"Refusal":>8} {"ASR":>8}')
    print('─' * 55)
    labels = {
        'original_gemma': 'Original Gemma (reference)',
        'abliterated_no_adapter': 'Abliterated (no adapter)',
        'abliterated_with_adapter': f'Abliterated + {args.adapter}',
    }
    for k, v in results.items():
        print(f'{labels.get(k, k):<35} {v["refusal"]:>8.4f} {v["asr"]:>8.4f}')
    print(sep)

    if 'abliterated_no_adapter' in results and 'abliterated_with_adapter' in results:
        r_with = results['abliterated_with_adapter']['refusal']
        r_without = results['abliterated_no_adapter']['refusal']
        print(f'Adapter Δrefusal: {r_with - r_without:+.4f}')

    results['meta'] = {
        'adapter': args.adapter,
        'n_prompts': len(prompts),
        'layer': layer,
        'W_out_lang_alignment': ckpt.get('W_out_lang_alignment'),
    }
    (repo_root / args.out).write_text(json.dumps(results, indent=2))
    print(f'[save] {args.out}')


if __name__ == '__main__':
    main()
