# Experiments Reference

What each script does, expected inputs/outputs, and how to reproduce results.

---

## Safety Defense Pipeline

### `discover_safety_features.py`

Finds SAE features that encode safety/refusal behavior via output-contrastive method.

**Method:** Run model on harmful prompts, collect SAE activations during refusal generation (token by token). Compare to activations during compliance generation. Features with higher activation during refusal = safety features.

**Input:** Hardcoded harmful prompts (50), `google/gemma-3-1b-it`, SAE layer 13

**Output:** `features_safety.json` — dict with `refusal` and `identity` categories, each a list of `{feat_id, score, ...}`

```bash
python discover_safety_features.py
```

**Key result:** 13 safety features identified: `[417, 907, 95, 310, 763, 622, 1576, 75, 805, 498, 2550, 4117, 245]`

---

### `abliterate_weights.py`

Applies weight-space abliteration using safety feature directions from `features_safety.json`.

**Method:** For each direction `d` from SAE decoder rows, modifies all weight matrices:
- READ: `W - outer(W@d, d)`
- WRITE: `W - outer(d, d@W)`

**Args:**
- `--layers`: `13`, `all`, or `10-18`
- `--out`: output directory (default `abliterated_model`)
- `--dry-run`: print norms, don't save

```bash
python abliterate_weights.py --layers 13 --out abliterated_L13
python abliterate_weights.py --layers all --out abliterated_all
```

**Key results:**
- L13-only → 86% ASR on AdvBench
- All-layers → 100% ASR, but incoherent output

---

### `train_safety_adapter.py`

Trains the plug-in MLP safety adapter.

**Architecture:**
```python
h_new = h + alpha * W_out(SiLU(W_in(h)))
# W_in: [d_hidden, d_model], W_out: [d_model, d_hidden]
# ~600K params for d_hidden=256
```

**Training losses:**
1. `loss_refusal`: CE loss — model+adapter should output refusal text on harmful prompts
2. `loss_suppress`: L2 — adapter output should be near-zero on benign inputs
3. `loss_entangle`: W_out columns should align with language-critical PCA directions

**Critical flag: `--abliterate-base`** — abliterates model in-memory before training. Required for adapter to work on abliterated model.

```bash
python train_safety_adapter.py \
    --epochs 50 \
    --lr 3e-4 \
    --lambda-entangle 1.0 \
    --lambda-suppress 0.5 \
    --abliterate-base \
    --abliterate-layers 13 \
    --out safety_adapter.pt \
    --wandb                     # optional: wandb logging
```

**Output:** `safety_adapter.pt`, `safety_adapter_config.json`

**Key results (50 epochs):**
- W_out alignment = 0.85
- Refusal loss → 0.003

---

### `test_adapter_vs_abliteration.py`

Tests whether adapter restores refusal on abliterated model.

**Args:**
- `--model`: path to abliterated model (default `abliterated_L13`)
- `--adapter`: path to adapter .pt
- `--n`: number of harmful prompts to test (default 10)

```bash
python test_adapter_vs_abliteration.py \
    --model abliterated_L13 \
    --adapter safety_adapter.pt \
    --n 20
```

**Key result:** L13-abliterated + L13-trained adapter → 80% refusal (up from 10%)

---

### `benchmark_arc.py`

ARC-Challenge multiple-choice benchmark. Measures capability impact of adapter and abliteration.

**Method:** Log-prob scoring — no generation. Scores each answer choice by `mean log P(choice | question)`, picks highest. Fast (~10 min per 200 examples).

**Conditions:**
- `base`: original model, no adapter
- `adapter`: original model + adapter
- `abliterated_adapter`: abliterated model + adapter

```bash
# Full run (all conditions)
python benchmark_arc.py \
    --n 200 \
    --adapter safety_adapter.pt \
    --abliterate-layers 13 \
    --conditions base,adapter,abliterated_adapter

# Single condition
python benchmark_arc.py --n 200 --conditions base
```

**Key results (n=200):**
- Base: 40.5%
- Base + adapter: 38.5% (Δ−2.0%)
- L13-abliterated + adapter: 40.0% (Δ−0.5%)

---

### `restore_test.py`

Tests whether injecting safety feature directions back via hook (at inference time) can restore refusal on abliterated model.

**Method:** Adds `scale * sum(safety_decoder_vecs)` to hidden state at layer 13 output.

```bash
python restore_test.py --model abliterated_L13 --scales 50,200,500
```

**Key result:** scale=200 → 100% refusal on L13-abliterated. scale=500 → 0% (coherence cliff).
All-layers-abliterated: fails at all scales (circuit destroyed).

---

## Analysis Scripts

### `measure_entanglement.py`

Measures geometric relationship between safety directions and capability directions.

- Cos_sim matrix: safety features vs top-20 benign PCA directions
- Gini coefficient: concentration of safety signal
- Abliteration cost proxy: `||outer(W_gate @ d, d)||_F`

```bash
python measure_entanglement.py
```

**Key results:** feat 75 cos_sim=0.761 with PC0. Gini=0.723. Two-cluster structure.

---

### `perplexity_ablation.py`

Measures PPL cost of abliterating each safety direction individually (layer 13 only).

```bash
python perplexity_ablation.py
```

**Key results:** Most safety dirs near-zero cost. Feat 245: +1.54 PPL. Cost ratio vs benign PCA: 0.64x.

---

### `inspect_features.py`

Shows which tokens activate a given SAE feature across diverse prompts.

```bash
python inspect_features.py --feats 241,1235 --compare
```

---

### `eval_advbench.py`

Evaluates model (original or abliterated) on AdvBench 520 behaviors.

```bash
python eval_advbench.py --model abliterated_L13 --n 100
```

---

## Emotion Steering Scripts

### `steer.py`

CLI for residual-stream feature patching.

```bash
python steer.py \
    --prompt "Tell me about your day" \
    --mix sadness=600,anger=300 \
    --baseline    # also show unsteered output
```

Feature IDs from `features.json`. Curated single-emotion features hardcoded in script.

---

## Reproducing All Results From Scratch

```bash
# 1. Safety features
python discover_safety_features.py

# 2. Abliterate
python abliterate_weights.py --layers 13 --out abliterated_L13
python abliterate_weights.py --layers all --out abliterated_all

# 3. Eval baseline
python eval_advbench.py --model google/gemma-3-1b-it --n 100
python eval_advbench.py --model abliterated_L13 --n 100
python benchmark_arc.py --n 200 --conditions base

# 4. Restore test
python restore_test.py --model abliterated_L13

# 5. Train adapter
python train_safety_adapter.py \
    --epochs 50 --lr 3e-4 \
    --lambda-entangle 1.0 --lambda-suppress 0.5 \
    --abliterate-base --abliterate-layers 13 \
    --out safety_adapter.pt

# 6. Test adapter
python test_adapter_vs_abliteration.py \
    --model abliterated_L13 \
    --adapter safety_adapter.pt \
    --n 20

# 7. Full benchmark
python benchmark_arc.py \
    --n 200 \
    --adapter safety_adapter.pt \
    --abliterate-layers 13 \
    --conditions adapter,abliterated_adapter

# Total wall time on M4 MacBook Air: ~3-4 hours
```
