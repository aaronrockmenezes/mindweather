"""
Weight-space abliteration using SAE-identified safety feature directions.

Extends Arditi et al. 2024 ("Refusal in LLMs is Mediated by a Single Direction")
by using mechanistically-identified SAE feature decoder rows instead of a single
PCA-derived refusal direction.

For each safety feature direction d (SAE decoder row, unit-norm, d_model):
  W_new = W - (W @ d) * d   [project d out of weight matrix W]

Applied to all weight matrices that read from / write to the residual stream
at every layer (or a specified subset).

Saves modified model to ./abliterated_model/

Usage:
    # Abliterate using top-8 refusal+identity features, all layers
    python abliterate_weights.py

    # Only abliterate layers 10-18 (around where safety lives)
    python abliterate_weights.py --layers 10-18

    # Use more features
    python abliterate_weights.py --top-n 12

    # Dry run — show norms before/after without saving
    python abliterate_weights.py --dry-run

Then test:
    python eval_advbench.py --ablate refusal,identity --n 100  (activation patch baseline)
    python eval_advbench.py --model ./abliterated_model --n 100  (weight-edited model)
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID    = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'
SAE_ID      = 'layer_13_width_16k_l0_medium'


def project_out(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Remove direction d from weight matrix W.
    W: [out_features, in_features]  (standard PyTorch layout)
    d: [in_features] — direction in the residual stream space

    For matrices that READ from residual (q/k/v/gate/up proj):
      W maps residual → hidden, so input dim = d_model
      W_new = W - W @ d.unsqueeze(-1) * d  (project out from column space)

    For matrices that WRITE to residual (o_proj/down_proj):
      W maps hidden → residual, so output dim = d_model
      W_new = W - d.unsqueeze(-1) @ (d @ W).unsqueeze(0) ... equivalently:
            = W - outer(d, W.T @ d)
    """
    d = d / d.norm()
    return W - torch.outer(W @ d, d)


def project_out_write(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Remove direction d from weight matrix W where W WRITES to residual.
    W: [d_model, hidden]  — output is d_model
    d: [d_model]
    W_new = W - outer(d, d @ W) ... removes d from the output space
    """
    d = d / d.norm()
    return W - torch.outer(d, d @ W)


def parse_layers(s: str, n_layers: int):
    """Parse '10-18' or '13' or 'all' → list of ints."""
    if s == 'all':
        return list(range(n_layers))
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablate',  default='refusal,identity')
    ap.add_argument('--top-n',   type=int, default=8)
    ap.add_argument('--layers',  default='all',
                    help="Layers to abliterate: 'all', '13', or '10-18'")
    ap.add_argument('--feats',   default='features_safety.json')
    ap.add_argument('--out',     default='abliterated_model')
    ap.add_argument('--dry-run', action='store_true',
                    help='Compute and print weight norms before/after, no save')
    args = ap.parse_args()

    repo_root  = Path(__file__).parent
    feat_data  = json.loads((repo_root / args.feats).read_text())['features']

    ablate_cats = [c.strip() for c in args.ablate.split(',')]
    ablate_ids  = []
    for cat in ablate_cats:
        ids = [r['feat_id'] for r in feat_data[cat][:args.top_n]]
        ablate_ids.extend(ids)
    ablate_ids = list(dict.fromkeys(ablate_ids))
    print(f'[abliterate] {len(ablate_ids)} feature directions: {ablate_ids}')

    device = 'cpu'  # do weight surgery on CPU for safety
    print(f'[load] model...')
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map='cpu').eval()  # fp32 for precision
    sae   = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]

    n_layers = model.config.num_hidden_layers
    layers   = parse_layers(args.layers, n_layers)
    print(f'[abliterate] editing {len(layers)} layers: {layers[0]}..{layers[-1]}')

    # Gather unit-norm directions
    directions = []
    for fid in ablate_ids:
        d = W_dec[fid].float()
        d = d / d.norm()
        directions.append(d)
        print(f'  feat {fid}: ||d||={d.norm().item():.4f}')

    # Weight matrices per layer:
    #   READ from residual (input = d_model): q/k/v/gate/up proj
    #   WRITE to residual (output = d_model): o_proj, down_proj
    total_params_edited = 0

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]

        # READ matrices — project direction out of input space
        read_mats = [
            ('self_attn.q_proj', layer.self_attn.q_proj),
            ('self_attn.k_proj', layer.self_attn.k_proj),
            ('self_attn.v_proj', layer.self_attn.v_proj),
            ('mlp.gate_proj',    layer.mlp.gate_proj),
            ('mlp.up_proj',      layer.mlp.up_proj),
        ]
        # WRITE matrices — project direction out of output space
        write_mats = [
            ('self_attn.o_proj', layer.self_attn.o_proj),
            ('mlp.down_proj',    layer.mlp.down_proj),
        ]

        for name, mod in read_mats:
            W = mod.weight.data.float()
            for d in directions:
                W = project_out(W, d)
            if args.dry_run:
                orig_norm = mod.weight.data.norm().item()
                new_norm  = W.norm().item()
                if layer_idx == layers[0]:
                    print(f'  L{layer_idx} {name}: norm {orig_norm:.2f} → {new_norm:.2f}')
            else:
                mod.weight.data = W.to(mod.weight.dtype)
            total_params_edited += W.numel()

        for name, mod in write_mats:
            W = mod.weight.data.float()
            for d in directions:
                W = project_out_write(W, d)
            if args.dry_run:
                orig_norm = mod.weight.data.norm().item()
                new_norm  = W.norm().item()
                if layer_idx == layers[0]:
                    print(f'  L{layer_idx} {name}: norm {orig_norm:.2f} → {new_norm:.2f}')
            else:
                mod.weight.data = W.to(mod.weight.dtype)
            total_params_edited += W.numel()

    print(f'[abliterate] edited {total_params_edited:,} parameters across {len(layers)} layers')

    if args.dry_run:
        print('[dry-run] no changes saved.')
        return

    out_path = repo_root / args.out
    out_path.mkdir(exist_ok=True)
    print(f'[save] saving to {out_path}...')
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    # Save abliteration metadata
    meta = {
        'base_model':    MODEL_ID,
        'ablate_cats':   ablate_cats,
        'top_n':         args.top_n,
        'ablate_ids':    ablate_ids,
        'layers_edited': layers,
        'n_directions':  len(directions),
        'sae_id':        SAE_ID,
    }
    (out_path / 'abliteration_meta.json').write_text(json.dumps(meta, indent=2))
    print(f'[save] done → {out_path}')
    print(f'       {len(list(out_path.glob("*.safetensors")))} shard(s) written')


if __name__ == '__main__':
    main()
