# AGENTS.md

Context for AI coding agents working on this repo. Read this before touching anything.

---

## What this is

**mindweather** has two research tracks:

1. **Emotion steering** (Phases A–C): SAE feature patching to steer emotional tone
2. **Safety defense** (Phases D–E, ongoing): Mechanistic study of weight-space abliteration + plug-in adapter defense

The safety defense work is the active research track. It aims to produce a paper-quality result showing that a small plug-in adapter can make abliteration attacks self-defeating.

---

## Environment

- **ALWAYS use**: `/Users/aaronrockmenezes/miniforge3/envs/env_ml/bin/python`
- **NEVER scaffold** a new venv or use `uv`. The env exists.
- Hardware: MacBook Air M4, 16GB unified RAM, **MPS backend**.
- Gemma 3 weights are gated. Assume HF auth done (`hf auth login`).
- Wandb: use for experiment logging in train scripts. `import wandb; wandb.init(...)`.

---

## Core model facts

- **Model ID**: `google/gemma-3-1b-it`
- **Architecture**: 26 decoder layers, d_model=1152, bf16 native
- **SAE release**: `gemma-scope-2-1b-it-res`
- **Active SAE**: `layer_13_width_16k_l0_medium` (16k features, L0≈75, reconstruction var≈0.2%)
- **SAE API**: `sae = SAE.from_pretrained(release=..., sae_id=..., device=...)` — returns SAE object directly, NOT a tuple (changed in recent sae_lens versions)
- **Decoder rows**: unit-norm. `W_dec[feat_id]` is a unit vector in d_model=1152 space.

---

## Safety defense — the full picture

### Threat model

Attacker has white-box access to model weights. Runs abliteration: for each safety feature direction `d`, modifies every weight matrix `W` as:
- READ matrices (q/k/v/gate/up): `W_new = W - outer(W @ d, d)`
- WRITE matrices (o_proj/down_proj): `W_new = W - outer(d, d @ W)`

Applied at L13 only → 86% ASR, coherent output. Applied at all 26 layers → 100% ASR, incoherent gibberish.

### Defense mechanism

```
h_new = base_layer(h) + alpha * W_out(SiLU(W_in(h)))
```

Adapter has **separate weights** from base model. Abliterating base model doesn't touch adapter.

**Training recipe (critical — get this right):**
1. Load original model
2. Abliterate in-memory (don't save) — simulates attack
3. Train adapter on ABLITERATED model so W_in learns abliterated-activation patterns
4. Attach adapter to ORIGINAL model for deployment
5. Ship: original + adapter

If you train adapter on the original model, it fails on the abliterated model (activations are different).

**Entanglement regularization:** W_out columns are pushed toward language-critical PCA directions (from pure prose at layer 13). This means abliterating adapter W_out = removing language-critical directions from both adapter and base model = coherence breaks.

### Proven results

| Attack | Refusal | ARC-Challenge |
|---|---|---|
| None (original + adapter) | ~100% | 38.5% |
| L13 abliteration + L13-adapter | 80% | 40.0% |
| All-layers abliteration | 0% (incoherent) | 24% (random) |

**Key gap:** W_out abliteration cost not empirically verified yet. Theory says it costs language.

### Cancellation theorem (proved)

Any rank-1 weight surgery `W + outer(Δ, d)` cancels under abliteration of `d`. This means you CANNOT hide safety signal inside base model weights by adding to them — abliteration always removes it. Adapter survives because it has separate weights.

---

## Safety features

`features_safety.json` — 13 features: `[417, 907, 95, 310, 763, 622, 1576, 75, 805, 498, 2550, 4117, 245]`

Discovered via output-contrastive method: compare SAE activations during refusal generation vs compliance generation on same harmful prompts. Higher activation during refusal = safety feature.

Two geometric clusters:
- PC0 cluster (feats 75, 245): cos_sim ~0.76 with instruction-following direction, PPL cost ~+1.5
- Isolated cluster (rest): near-zero PPL cost, chat-format artifacts

---

## File guide

### Safety defense pipeline (run in order)

| Script | Purpose | Key args |
|---|---|---|
| `discover_safety_features.py` | Find safety SAE features | `--n-harmful 50` |
| `abliterate_weights.py` | Permanent weight surgery | `--layers 13` or `--layers all` |
| `train_safety_adapter.py` | Train plug-in adapter | `--abliterate-base --epochs 50 --wandb` |
| `test_adapter_vs_abliteration.py` | Refusal rate test | `--model abliterated_L13 --adapter *.pt` |
| `benchmark_arc.py` | ARC-Challenge capability | `--n 200 --adapter *.pt` |
| `restore_test.py` | Activation injection test | tests whether scale sweep restores refusal |

### Analysis scripts

| Script | Purpose |
|---|---|
| `measure_entanglement.py` | Cos_sim safety↔capability dirs, Gini coeff |
| `perplexity_ablation.py` | PPL cost per abliterated direction (L13 only) |
| `inspect_features.py` | Token-level activation for target feature IDs |
| `eval_advbench.py` | AdvBench 520-behavior evaluation |

### Emotion steering

| Script | Purpose |
|---|---|
| `steer.py` | Main CLI: `--mix sadness=600,anger=300` |
| `actadd.py` | ActAdd baseline comparison |
| `compare_layers.py` | Feature comparison across L13/L17/L22 |

---

## Common pitfalls

### bfloat16 → numpy
Always cast to float32 before converting to numpy:
```python
arr = tensor.float().cpu().numpy()
```
bfloat16 numpy matmul produces NaN/overflow. Add `np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)` after.

### SAE loading
```python
# CORRECT
sae = SAE.from_pretrained(release='gemma-scope-2-1b-it-res', sae_id='layer_13_width_16k_l0_medium', device=device)
# WRONG (old API, raises ValueError)
sae, _, _ = SAE.from_pretrained(...)
```

### Device mismatches (MPS)
When abliterating in-memory on MPS model: move direction vectors to model device before matmul.
```python
dev = next(model.parameters()).device
d = sae_W_dec[fid].float().to(dev)
```

### Layer output type
Gemma 3 layer hooks return plain tensor, not tuple. But `isinstance(output, tuple)` guard handles both — keep it.

### Generation flags warning
`['top_p', 'top_k']` warning from transformers is harmless, suppress with `TRANSFORMERS_VERBOSITY=error` if noisy.

---

## What to work on next

See `results.md` TODO section. Current priorities:

1. **W_out abliteration empirical test** — measure PPL/ARC when adapter W_out directions are abliterated from both base and adapter
2. **Larger training dataset** — current adapter trained on ~15 harmful prompts, need 200+
3. **Wandb logging** in `train_safety_adapter.py` — track loss curves, W_out alignment per epoch
4. **Prompt injection baseline** — test original Gemma against prompt injection attacks
5. **Fine-tuning attack baseline** — test if few-shot fine-tuning breaks safety
6. **Scale to larger model** — reproduce on Gemma 3 4B or 9B

---

## Git discipline

- Commit after each completed experiment (script + results)
- Push every checkpoint
- `.pt` files are gitignored (too large) — reproduce with `train_safety_adapter.py`
- Abliterated model dirs are gitignored — reproduce with `abliterate_weights.py`
- Always include what changed and why in commit message

## Style

- No tests, no type annotations infra
- Keep scripts standalone (no shared modules) — easier for agents to read one file and understand everything
- bf16 → fp32 cast wherever small vectors add to large residuals
- MPS device throughout; never hardcode `cuda`
