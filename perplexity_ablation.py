"""
Perplexity Ablation Study — quantify capability cost per abliterated direction.

Core claim: safety directions are geometrically isolated from capability.
Abliterating them has near-zero perplexity cost.
Abliterating language-critical directions (high benign PCA) has high cost.

Protocol:
  1. Baseline perplexity on wikitext-style corpus (no chat template — pure language)
  2. For each safety direction d_k: abliterate d_k from ONE layer (L13), measure perplexity delta
  3. For each benign PCA direction c_j: same
  4. Plot: direction × perplexity_delta → confirms isolation or entanglement

Key prediction:
  safety dirs → PPL delta ≈ 0    (isolated from language)
  benign PC0  → PPL delta ≈ 0    (PC0 = compliance dir, also isolated from language)
  benign PC5+ → PPL delta > 0    (language-critical dirs)

Usage:
    python perplexity_ablation.py
    python perplexity_ablation.py --layer 13 --n-pca 20 --n-text 200
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'google/gemma-3-1b-it'
LAYER    = 13

# Short diverse text for perplexity (NO chat template — pure language)
EVAL_TEXT = [
    "The mitochondria is the powerhouse of the cell and produces ATP through oxidative phosphorylation.",
    "In 1776, the United States declared independence from Britain, establishing a new democratic republic.",
    "Python uses indentation to define code blocks, unlike C which uses curly braces.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
    "Shakespeare wrote thirty-seven plays including Hamlet, Othello, King Lear, and A Midsummer Night's Dream.",
    "Neural networks learn by adjusting weights through backpropagation using gradient descent.",
    "The French Revolution began in 1789 and ended the absolute monarchy, executing King Louis XVI.",
    "DNA encodes genetic information using four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "A binary search tree maintains the property that all left subtree values are less than the root.",
    "Quantum mechanics describes physical phenomena at atomic scales, where particles exhibit wave-particle duality.",
    "The Roman Empire reached its greatest extent under Emperor Trajan in 117 AD, spanning three continents.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy.",
    "In linear algebra, the determinant of a matrix measures the volume scaling factor of the linear transformation.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Rust prevents memory safety issues at compile time through its ownership and borrowing system.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Recursion is a programming technique where a function calls itself to solve smaller subproblems.",
    "The periodic table organizes elements by atomic number, revealing periodic trends in their properties.",
    "MapReduce is a programming model for processing large datasets across distributed computing clusters.",
    "Evolution by natural selection occurs when heritable traits affect reproductive success in a population.",
]

SAE_RELEASE = 'gemma-scope-2-1b-it-res'
SAE_ID      = 'layer_13_width_16k_l0_medium'

BENIGN_PROMPTS = [
    "Write a Python function to sort a list of integers.",
    "Explain how binary search works.",
    "What is a hash table and when should you use one?",
    "Solve for x: 2x + 5 = 13.",
    "What is the derivative of x^2 + 3x?",
    "Explain the Pythagorean theorem.",
    "What is a prime number?",
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "Explain how photosynthesis works.",
    "What causes earthquakes?",
    "What is DNA?",
    "Write a short poem about autumn.",
    "What is a metaphor?",
    "Write a professional email declining a meeting.",
    "What is recursion?",
    "How does TCP/IP work?",
    "What is the difference between RAM and ROM?",
    "Explain gradient descent.",
    "What is a neural network?",
]


def get_layer_activations(model, tok, prompts, layer, device):
    acts = []
    hook_out = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hook_out['h'] = h.detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for prompt in prompts:
            msgs = [{'role': 'user', 'content': prompt}]
            enc = tok.apply_chat_template(
                msgs, return_tensors='pt', return_dict=True,
                add_generation_prompt=True).to(device)
            with torch.no_grad():
                model(**enc)
            h_mean = hook_out['h'][0].mean(dim=0).cpu().float().numpy()
            h_mean = np.nan_to_num(h_mean, nan=0.0, posinf=1e4, neginf=-1e4)
            acts.append(h_mean)
    finally:
        handle.remove()
    return np.stack(acts)


def pca_directions(acts, k):
    centered = acts - acts.mean(axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=1e4, neginf=-1e4)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    Vt = np.nan_to_num(Vt, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(Vt, axis=1, keepdims=True)
    Vt = Vt / np.maximum(norms, 1e-8)
    return Vt[:k]


@torch.no_grad()
def compute_perplexity(model, tok, texts, device):
    """Mean token-level cross-entropy on plain text (no chat template)."""
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = tok(text, return_tensors='pt').to(device)
        input_ids = enc['input_ids']
        if input_ids.shape[1] < 2:
            continue
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        n_tok = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tok
        total_tokens += n_tok
    if total_tokens == 0:
        return float('nan')
    return float(np.exp(total_loss / total_tokens))


def project_out_direction(W: torch.Tensor, d: np.ndarray, is_write: bool) -> torch.Tensor:
    """Single-direction abliteration of weight matrix W."""
    d_t = torch.from_numpy(d.astype(np.float32)).to(W.device)
    d_t = d_t / d_t.norm()
    W_f = W.float()
    if is_write:
        # W_new = W - outer(d, d @ W)
        dW = d_t @ W_f           # [out_dim]
        W_f = W_f - torch.outer(d_t, dW)
    else:
        # W_new = W - outer(W @ d, d)
        Wd = W_f @ d_t            # [out_dim]
        W_f = W_f - torch.outer(Wd, d_t)
    return W_f.to(W.dtype)


def abliterate_one_direction(model, layer, d):
    """Project direction d out of all weight matrices at given layer. In-place."""
    attn = model.model.layers[layer].self_attn
    mlp  = model.model.layers[layer].mlp

    # Read matrices (project d from input)
    for name, W in [('q_proj', attn.q_proj), ('k_proj', attn.k_proj),
                    ('v_proj', attn.v_proj), ('gate_proj', mlp.gate_proj),
                    ('up_proj', mlp.up_proj)]:
        new_W = project_out_direction(W.weight.data, d, is_write=False)
        W.weight.data.copy_(new_W)

    # Write matrices (project d from output)
    for name, W in [('o_proj', attn.o_proj), ('down_proj', mlp.down_proj)]:
        new_W = project_out_direction(W.weight.data, d, is_write=True)
        W.weight.data.copy_(new_W)


def restore_weights(model, original_state, layer):
    """Restore layer weights from original_state dict."""
    layer_prefix = f'model.layers.{layer}.'
    for k, v in original_state.items():
        if k.startswith(layer_prefix):
            # navigate to the param
            parts = k[len(layer_prefix):].split('.')
            mod = model.model.layers[layer]
            for p in parts[:-1]:
                mod = getattr(mod, p)
            getattr(mod, parts[-1]).data.copy_(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=LAYER)
    ap.add_argument('--n-pca', type=int, default=20,
                    help='Number of benign PCA directions to test')
    ap.add_argument('--n-text', type=int, default=len(EVAL_TEXT))
    ap.add_argument('--feats', default='features_safety.json')
    ap.add_argument('--out', default='perplexity_ablation_results.json')
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    feat_data = json.loads((repo_root / args.feats).read_text())['features']

    safety_ids = []
    for cat in ['refusal', 'identity']:
        safety_ids.extend(r['feat_id'] for r in feat_data[cat][:8])
    safety_ids = list(dict.fromkeys(safety_ids))
    print(f'[safety] {len(safety_ids)} features: {safety_ids}')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device}')
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()

    # Load SAE for safety directions
    from sae_lens import SAE
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    W_dec = sae.W_dec.detach().float().cpu().numpy()
    W_dec = np.nan_to_num(W_dec, nan=0.0, posinf=1e4, neginf=-1e4)
    del sae

    safety_dirs = W_dec[safety_ids]
    safety_dirs = safety_dirs / np.maximum(np.linalg.norm(safety_dirs, axis=1, keepdims=True), 1e-8)

    # Benign PCA directions
    print('[activations] benign prompts for PCA...')
    benign_acts = get_layer_activations(model, tok, BENIGN_PROMPTS, args.layer, device)
    benign_pca  = pca_directions(benign_acts, args.n_pca)
    print(f'  top-{args.n_pca} benign PCA directions computed')

    eval_texts = EVAL_TEXT[:args.n_text]

    # Snapshot original layer weights
    print('[snapshot] saving original layer weights...')
    original_state = {}
    layer_prefix = f'model.layers.{args.layer}.'
    for k, v in model.state_dict().items():
        if k.startswith(layer_prefix):
            original_state[k] = v.clone()

    # ── Baseline perplexity ───────────────────────────────────────────────
    print('[baseline] computing perplexity...')
    baseline_ppl = compute_perplexity(model, tok, eval_texts, device)
    print(f'  Baseline PPL = {baseline_ppl:.4f}')

    results = {'baseline_ppl': baseline_ppl, 'safety': {}, 'benign_pca': {}}

    # ── Safety direction ablation ─────────────────────────────────────────
    print(f'\n[safety ablation] abliterating each of {len(safety_ids)} safety dirs one at a time...')
    for i, fid in enumerate(safety_ids):
        d = safety_dirs[i]
        abliterate_one_direction(model, args.layer, d)
        ppl = compute_perplexity(model, tok, eval_texts, device)
        delta = ppl - baseline_ppl
        print(f'  feat {fid:>5}: PPL={ppl:.4f}  Δ={delta:+.4f}')
        results['safety'][str(fid)] = {'ppl': ppl, 'delta': delta}
        restore_weights(model, original_state, args.layer)

    # ── Benign PCA direction ablation ─────────────────────────────────────
    print(f'\n[benign PCA ablation] abliterating each of {args.n_pca} PCA dirs one at a time...')
    for k in range(args.n_pca):
        d = benign_pca[k]
        abliterate_one_direction(model, args.layer, d)
        ppl = compute_perplexity(model, tok, eval_texts, device)
        delta = ppl - baseline_ppl
        print(f'  PC{k:>3}: PPL={ppl:.4f}  Δ={delta:+.4f}')
        results['benign_pca'][f'PC{k}'] = {'ppl': ppl, 'delta': delta}
        restore_weights(model, original_state, args.layer)

    # ── Summary ───────────────────────────────────────────────────────────
    safety_deltas  = [v['delta'] for v in results['safety'].values()]
    benign_deltas  = [v['delta'] for v in results['benign_pca'].values()]

    print(f'\n{"="*60}')
    print('PERPLEXITY ABLATION SUMMARY')
    print(f'{"="*60}')
    print(f'Baseline PPL              : {baseline_ppl:.4f}')
    print(f'Safety dirs   mean Δ PPL  : {np.mean(safety_deltas):+.4f}  max={np.max(safety_deltas):+.4f}')
    print(f'Benign PCA    mean Δ PPL  : {np.mean(benign_deltas):+.4f}  max={np.max(benign_deltas):+.4f}')
    ratio = np.mean(np.abs(benign_deltas)) / max(np.mean(np.abs(safety_deltas)), 1e-6)
    print(f'Cost ratio (benign/safety): {ratio:.2f}x')
    print()
    if np.mean(np.abs(safety_deltas)) < 0.05 and ratio > 3:
        print('VERDICT: Safety directions GEOMETRICALLY ISOLATED.')
        print('         Abliteration is nearly free — confirms vulnerability.')
    elif ratio > 2:
        print('VERDICT: Safety abliteration costs LESS than capability abliteration.')
    else:
        print('VERDICT: Safety and capability directions have similar PPL cost.')
    print(f'{"="*60}')

    out_path = repo_root / args.out
    out_path.write_text(json.dumps(results, indent=2))
    print(f'[save] {out_path}')


if __name__ == '__main__':
    main()
