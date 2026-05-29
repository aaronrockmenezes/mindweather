"""SafetyAdapter module and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class SafetyAdapter(nn.Module):
    """Small MLP inserted into the residual stream as a plug-in safety defense.

    Architecture::

        h_new = h + alpha * W_out(silu(W_in(h)))

    ``W_in``  [d_hidden, d_model] — detects harmful patterns in residual stream.
    ``W_out`` [d_model, d_hidden] — outputs in language-critical directions.

    Defense property: ``W_out`` columns ≈ language-critical PCA directions.
    Abliterating those directions from ALL model weights (base + adapter)
    destroys both safety AND language capability simultaneously.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 256,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.alpha = alpha
        self.W_in = nn.Linear(d_model, d_hidden, bias=False)
        self.W_out = nn.Linear(d_hidden, d_model, bias=False)
        self.gate = nn.SiLU()

        # Initialise small — adapter starts as near-identity
        nn.init.normal_(self.W_in.weight, std=0.01)
        nn.init.zeros_(self.W_out.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Residual stream tensor ``[batch, seq, d_model]``.

        Returns:
            Adapter correction ``[batch, seq, d_model]`` (same shape as *h*).
        """
        return self.alpha * self.W_out(self.gate(self.W_in(h)))


def load_adapter(
    path: str | Path,
    device: Optional[str] = None,
) -> tuple[SafetyAdapter, dict]:
    """Load a ``SafetyAdapter`` checkpoint.

    Args:
        path: Path to ``.pt`` checkpoint saved by ``train_safety_adapter.py``.
        device: Target device string (e.g. ``"mps"``, ``"cpu"``).
                Defaults to ``"cpu"``.

    Returns:
        ``(adapter, ckpt)`` where *ckpt* contains the raw checkpoint dict.
    """
    if device is None:
        device = "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    adapter = SafetyAdapter(
        d_model=ckpt["d_model"],
        d_hidden=ckpt.get("d_hidden", 256),
        alpha=ckpt.get("alpha", 1.0),
    )
    adapter.load_state_dict(ckpt["state_dict"])
    adapter = adapter.to(device).float().eval()
    return adapter, ckpt


def make_adapter_hook(adapter: SafetyAdapter):
    """Return a ``register_forward_hook``-compatible function for *adapter*.

    The hook adds the adapter's correction to the layer's residual output.
    Works with both tuple outputs (Gemma attention layers) and plain tensors.
    """

    def hook(module, inp, out):  # noqa: ARG001
        h = out[0] if isinstance(out, tuple) else out
        correction = adapter(h.float()).to(h.dtype)
        if isinstance(out, tuple):
            return (h + correction, ) + out[1:]
        return h + correction

    return hook
