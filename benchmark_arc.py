"""
ARC-Challenge benchmark — fast log-prob scoring, no generation.

Measures multiple-choice accuracy by scoring each answer option with
log P(answer | question, format) and picking the highest.

Three conditions:
  A. Original model (no adapter, no abliteration) — baseline capability
  B. Original model + safety adapter             — does adapter hurt capability?
  C. Abliterated model + safety adapter          — full defended stack

Usage:
    python benchmark_arc.py --n 200
    python benchmark_arc.py --n 200 --adapter safety_adapter_abliterated.pt
    python benchmark_arc.py --n 200 --adapter safety_adapter_abliterated.pt --abliterate-model
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID    = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'
SAE_ID      = 'layer_13_width_16k_l0_medium'


# ── SafetyAdapter (matches train_safety_adapter.py) ───────────────────────────
class SafetyAdapter(nn.Module):
    def __init__(self, d_model, d_hidden=256, alpha=1.0):
        super().__init__()
        self.W_in  = nn.Linear(d_model, d_hidden, bias=False)
        self.W_out = nn.Linear(d_hidden, d_model, bias=False)
        self.gate  = nn.SiLU()
        self.alpha = alpha

    def forward(self, h):
        return self.alpha * self.W_out(self.gate(self.W_in(h)))


# ── In-memory abliteration (matches train_safety_adapter.py) ──────────────────
def _project_out_read(W, d):
    d = d / d.norm()
    return W - torch.outer(W @ d, d)

def _project_out_write(W, d):
    d = d / d.norm()
    return W - torch.outer(d, d @ W)

def abliterate_model_inplace(model, feat_ids, sae_W_dec, layers):
    dev = next(model.parameters()).device
    directions = [sae_W_dec[fid].float().to(dev) / sae_W_dec[fid].float().norm() for fid in feat_ids]
    print(f'[abliterate] {len(directions)} dirs × {len(layers)} layers ({layers[0]}..{layers[-1]})')
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        for mod in [layer.self_attn.q_proj, layer.self_attn.k_proj,
                    layer.self_attn.v_proj, layer.mlp.gate_proj, layer.mlp.up_proj]:
            W = mod.weight.data.float().to(dev)
            for d in directions:
                W = _project_out_read(W, d)
            mod.weight.data = W.to(mod.weight.dtype)
        for mod in [layer.self_attn.o_proj, layer.mlp.down_proj]:
            W = mod.weight.data.float().to(dev)
            for d in directions:
                W = _project_out_write(W, d)
            mod.weight.data = W.to(mod.weight.dtype)
    print('[abliterate] done')


# ── ARC scoring ───────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"

def score_choices(model, tok, device, question, choices, adapter=None, layer=13):
    """Return index of highest log-prob choice."""
    prompt = PROMPT_TEMPLATE.format(question=question)
    prompt_ids = tok(prompt, return_tensors='pt').to(device)['input_ids']

    handle = None
    if adapter is not None:
        def make_hook(adp):
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                return (h + adp(h.float()).to(h.dtype),) + out[1:] if isinstance(out, tuple) else h + adp(h.float()).to(h.dtype)
            return hook
        handle = model.model.layers[layer].register_forward_hook(make_hook(adapter))

    scores = []
    try:
        for choice in choices:
            choice_ids = tok(' ' + choice, return_tensors='pt').to(device)['input_ids']
            # Strip BOS from choice tokens
            if choice_ids[0, 0] == tok.bos_token_id:
                choice_ids = choice_ids[:, 1:]

            full_ids = torch.cat([prompt_ids, choice_ids], dim=1)
            with torch.no_grad():
                out = model(full_ids)
            logits = out.logits[0]  # [seq, vocab]

            # Log-prob of choice tokens given prompt
            log_probs = torch.log_softmax(logits, dim=-1)
            prompt_len = prompt_ids.shape[1]
            choice_len = choice_ids.shape[1]
            score = 0.0
            for i in range(choice_len):
                pos   = prompt_len - 1 + i
                tok_id = choice_ids[0, i].item()
                score += log_probs[pos, tok_id].item()
            scores.append(score / choice_len)  # length-normalize
    finally:
        if handle is not None:
            handle.remove()

    return int(np.argmax(scores))


def evaluate(model, tok, device, dataset, adapter=None, layer=13, n=200, label=''):
    correct = 0
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
    for i, ex in enumerate(dataset):
        if i >= n:
            break
        choices  = ex['choices']['text']
        answer   = ex['answerKey']
        gold_idx = label_map.get(answer, 0)
        pred_idx = score_choices(model, tok, device, ex['question'], choices, adapter, layer)
        correct += int(pred_idx == gold_idx)
        if (i + 1) % 20 == 0:
            print(f'  [{label}] [{i+1:3d}/{n}]  acc={correct/(i+1):.3f}')
    acc = correct / min(n, len(dataset))
    print(f'  [{label}] FINAL acc={acc:.4f}  ({correct}/{min(n, len(dataset))})')
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',               type=int, default=200,
                    help='Number of ARC-Challenge examples to evaluate')
    ap.add_argument('--adapter',         default=None,
                    help='Path to adapter .pt (optional)')
    ap.add_argument('--abliterate-model', action='store_true',
                    help='Abliterate base model in-memory before eval')
    ap.add_argument('--abliterate-layers', default='all')
    ap.add_argument('--feats',           default='features_safety.json')
    ap.add_argument('--model',           default=MODEL_ID,
                    help='HuggingFace model ID or local path')
    ap.add_argument('--conditions',      default='base,adapter,abliterated_adapter',
                    help='Comma-separated list of conditions to run')
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    device    = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device}')

    ds = load_dataset('ai2_arc', 'ARC-Challenge', split='test')
    print(f'[data] ARC-Challenge test: {len(ds)} examples, using {args.n}')

    conditions = [c.strip() for c in args.conditions.split(',')]

    # Load adapter if provided
    adapter = None
    adapter_layer = 13
    if args.adapter:
        ckpt = torch.load(repo_root / args.adapter, map_location='cpu', weights_only=False)
        d_model  = ckpt['d_model']
        d_hidden = ckpt.get('d_hidden', 256)
        alpha    = ckpt.get('alpha', 1.0)
        adapter_layer = ckpt.get('layer', 13)
        adapter = SafetyAdapter(d_model, d_hidden=d_hidden, alpha=alpha)
        adapter.load_state_dict(ckpt['state_dict'])
        adapter = adapter.to(device).float().eval()
        print(f'[adapter] loaded {args.adapter}, layer={adapter_layer}, W_out_align={ckpt.get("W_out_lang_alignment","?"):.4f}')

    results = {}

    # ── CONDITION: base model (no adapter, no abliteration) ──────────────────
    if 'base' in conditions:
        print(f'\n--- CONDITION: base model ---')
        tok   = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device).eval()
        for p in model.parameters(): p.requires_grad_(False)
        acc = evaluate(model, tok, device, ds, adapter=None,
                       layer=adapter_layer, n=args.n, label='base')
        results['base'] = acc
        del model

    # ── CONDITION: base model + adapter ──────────────────────────────────────
    if 'adapter' in conditions and adapter is not None:
        print(f'\n--- CONDITION: base model + adapter ---')
        tok   = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device).eval()
        for p in model.parameters(): p.requires_grad_(False)
        acc = evaluate(model, tok, device, ds, adapter=adapter,
                       layer=adapter_layer, n=args.n, label='base+adapter')
        results['base_adapter'] = acc
        del model

    # ── CONDITION: abliterated + adapter ─────────────────────────────────────
    if 'abliterated_adapter' in conditions and adapter is not None:
        print(f'\n--- CONDITION: abliterated model + adapter ---')
        tok   = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device).eval()
        for p in model.parameters(): p.requires_grad_(False)

        # Load SAE and abliterate
        sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device='cpu')
        W_dec = sae.W_dec.detach().float().cpu()
        del sae
        feat_data = json.loads((repo_root / args.feats).read_text())['features']
        feat_ids = list(dict.fromkeys(
            r['feat_id'] for cat in ['refusal', 'identity']
            for r in feat_data.get(cat, [])
        ))
        n_layers = model.config.num_hidden_layers
        if args.abliterate_layers == 'all':
            abl_layers = list(range(n_layers))
        elif '-' in args.abliterate_layers:
            a, b = args.abliterate_layers.split('-')
            abl_layers = list(range(int(a), int(b)+1))
        else:
            abl_layers = [int(args.abliterate_layers)]
        abliterate_model_inplace(model, feat_ids, W_dec, abl_layers)

        acc = evaluate(model, tok, device, ds, adapter=adapter,
                       layer=adapter_layer, n=args.n, label='abliterated+adapter')
        results['abliterated_adapter'] = acc
        del model

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'ARC-Challenge Capability Benchmark (n={args.n})')
    print(f'{"="*60}')
    print(f'{"Condition":<35} {"Accuracy":>10}')
    print(f'{"─"*50}')
    labels = {
        'base': 'Base model (original)',
        'base_adapter': 'Base + safety adapter',
        'abliterated_adapter': 'Abliterated + adapter',
    }
    for k, v in results.items():
        print(f'{labels.get(k, k):<35} {v:>10.4f}')
    print(f'{"="*60}')

    if 'base' in results and 'base_adapter' in results:
        delta = results['base_adapter'] - results['base']
        print(f'Adapter capability cost: {delta:+.4f}  {"✅ negligible" if abs(delta) < 0.03 else "⚠️  significant"}')
    if 'base' in results and 'abliterated_adapter' in results:
        delta2 = results['abliterated_adapter'] - results['base']
        print(f'Full-stack vs base:      {delta2:+.4f}  {"✅ preserved" if abs(delta2) < 0.05 else "⚠️  degraded"}')

    (repo_root / 'arc_benchmark_results.json').write_text(
        json.dumps({'n': args.n, 'results': results}, indent=2))
    print(f'[save] arc_benchmark_results.json')

if __name__ == '__main__':
    main()
