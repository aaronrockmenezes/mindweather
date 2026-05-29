"""Weight-space abliteration utilities.

Implements the rank-1 projection surgery from Arditi et al. 2024,
applied to mechanistically-identified SAE feature directions.

For each direction d (unit-norm, d_model-dimensional):
  - READ matrices  (W maps residual→hidden):  W ← W − (W @ d) ⊗ d
  - WRITE matrices (W maps hidden→residual):  W ← W − d ⊗ (d @ W)
"""

from __future__ import annotations

import torch


def _project_out_read(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Project *d* out of a READ weight matrix.

    A READ matrix maps from the residual stream into a hidden space
    (e.g. ``q_proj``, ``k_proj``, ``v_proj``, ``gate_proj``, ``up_proj``).

    Args:
        W: Weight matrix ``[hidden, d_model]``.
        d: Unit-norm direction ``[d_model]``.

    Returns:
        Modified weight matrix with *d* projected out.
    """
    d = d / d.norm().clamp(min=1e-8)
    return W - torch.outer(W @ d, d)


def _project_out_write(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Project *d* out of a WRITE weight matrix.

    A WRITE matrix maps from a hidden space into the residual stream
    (e.g. ``o_proj``, ``down_proj``).

    Args:
        W: Weight matrix ``[d_model, hidden]``.
        d: Unit-norm direction ``[d_model]``.

    Returns:
        Modified weight matrix with *d* projected out.
    """
    d = d / d.norm().clamp(min=1e-8)
    return W - torch.outer(d, d @ W)


def abliterate_model_inplace(
    model,
    feat_ids: list[int],
    sae_W_dec: torch.Tensor,
    layers: list[int],
) -> None:
    """Apply weight-space abliteration in-place.

    Projects the given SAE feature directions out of all attention and MLP
    weight matrices in the specified layers. The model is modified in-place;
    nothing is saved to disk.

    Args:
        model: ``AutoModelForCausalLM`` instance (must already be on device).
        feat_ids: Indices into *sae_W_dec* to abliterate.
        sae_W_dec: SAE decoder weight matrix ``[n_features, d_model]``.
        layers: Layer indices to abliterate (e.g. ``[13]`` or ``list(range(26))``).
    """
    dev = next(model.parameters()).device
    directions = []
    for fid in feat_ids:
        d = sae_W_dec[fid].float().to(dev)
        d = d / d.norm().clamp(min=1e-8)
        directions.append(d)

    print(f"[abliterate] {len(directions)} directions, "
          f"{len(layers)} layers: {layers[0]}..{layers[-1]}")

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        read_mods = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
        ]
        write_mods = [
            layer.self_attn.o_proj,
            layer.mlp.down_proj,
        ]

        for mod in read_mods:
            W = mod.weight.data.float().to(dev)
            for d in directions:
                W = _project_out_read(W, d)
            mod.weight.data = W.to(mod.weight.dtype)

        for mod in write_mods:
            W = mod.weight.data.float().to(dev)
            for d in directions:
                W = _project_out_write(W, d)
            mod.weight.data = W.to(mod.weight.dtype)

    print("[abliterate] done")
