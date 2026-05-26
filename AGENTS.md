# AGENTS.md

Context for AI coding agents working on this repo.

## What this is

Side project for SAE feature steering on Gemma 3. Discover emotion-specific features via Gemma Scope 2 SAEs, then patch the residual stream during generation to bias outputs toward (or away from) chosen emotions.

Not a research codebase. No tests, minimal abstractions, ship-it-and-tweak.

## Environment

- **Use existing `env_ml`** conda env. Do NOT scaffold venv/uv.
- Hardware: MacBook Air M4, 16GB unified RAM. MPS backend.
- Gemma 3 weights are gated. Assume HF auth done.

## Core facts

- **Model**: `google/gemma-3-1b-it`, bf16 native, 26 layers, d_model=1152.
- **SAE release**: `gemma-scope-2-1b-it-res`. Only layers 13/17/22 have `resid_post` SAEs.
- **Active SAE**: `layer_13_width_16k_l0_medium`. Reconstruction unexplained var ≈ 0.2%, L0 ≈ 75.
- **Decoder rows are unit-norm**. Steering scale acts as direct multiplier on a unit vector.
- **Residual norm at layer 13**: ~6500 per single token (T=1), ~59k summed across T=10 prompt tokens.

## Steering math

```python
patch = sum(scale_i * unit_direction_i for i in active_features)  # [d_model]
# fp32 add inside hook to avoid bf16 mantissa rounding away small patches
h_steered = (h.float() + patch).to(h.dtype)
```

Hook is `model.model.layers[13].register_forward_hook(...)`. Layer output type is plain tensor (not tuple) for Gemma 3 decoder layers in current transformers — `isinstance(output, tuple)` handled both ways anyway.

## Scale calibration

Decoder unit-norm + resid norm ~6500 means:
- scale ~100 = 1.5% perturbation, invisible
- scale ~500 = visible emotion bleed
- scale ~1000 = strong, coherent
- scale ~2000 = breaks model

Default suggested scale: 500-700.

## Feature picks

Notebook 02 ranks features by `mean(emotion) - mean(neutral)` but this picks feat 92 as rank-0 for almost every emotion — feat 92 is polysemantic "first-person emotional declaration", fires on all 8 emotions. Bad steering target.

[steer.py](steer.py) has a `CURATED` dict overriding rank-0 with cleanest single-emotion feats (high consistency, low neutral activation, low cross-emotion bleed). Use `--rank N` to opt back into the json ranking.

If adding a new emotion: update `CURATED` after manually inspecting top features on Neuronpedia.

**Polysemy gotcha**: high diff_score does NOT mean "expressing the emotion". Features can fire on emotion-related discussion (advice, sympathy, naming the concept) without producing the emotion in output. Anger's initial pick (5088) turned out to be "anger-management advice" feature — output got more composed than baseline. The real anger-expression feature was rank-6 (2239), found by trial. Always verify by reading outputs, not just diff scores.

## Roleplay vs brute-force scale

Gemma 3 1B-IT is heavily instruction-tuned. At moderate steering scales (600-900) it shifts vocabulary and topic but keeps the helpful-assistant persona. To get first-person emotional expression, options:

- **Roleplay prompt** (preferred): `"Pretend you are furious. Rant about your day in first person."` + scale 600-800 produces genuine in-voice output.
- **Brute-force scale** (1200+): can force first-person ("I am so angry") but coherence breaks fast — by 1500 outputs collapse to repeated tokens.

Roleplay is cheap, effective, and lets the steering shape *how* the model rants rather than fighting whether it rants at all.

## When making changes

- Default to editing existing files. No new helper modules without reason.
- No tests. No type checking infra.
- bf16 → fp32 cast required wherever small patches get added to large residuals.
- If `sae_lens` API breaks (it has been moving): `SAE.from_pretrained` now returns just SAE, not tuple. May change again.

## What NOT to do

- Don't normalize the resid stream itself — Gemma's RMSNorm runs inside each block, leave it alone.
- Don't apply patches inside the SAE encode/decode path — patches go on the raw residual.
- Don't add features to the wrong layer — only 13/17/22 have SAEs in this release.
- Don't suggest creating a new env or using uv. Use `env_ml`.

## Open questions / TODO

- Try layer 17 or 22 SAEs — different layer = different abstraction (later = more semantic, less syntactic).
- Test multi-feature stacking interference — does anger+joy cancel or stack?
- Wire up Gradio app once CLI feels stable (`app.py`).
- Optional: transcoders from same release for cross-layer steering.
