# Safety Defense — Deep Dive

## Background: Weight-Space Abliteration

Abliteration ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) is a white-box jailbreak. Given unit direction `d` in residual stream space:

```
READ matrices  (q/k/v/gate/up):   W_new = W - outer(W @ d, d)
WRITE matrices (o_proj/down_proj): W_new = W - outer(d, d @ W)
```

Effect: after modification, `W_new @ d = 0` (read) or `d @ W_new = 0` (write). Direction `d` is erased from the model's computation.

Applied to 13 SAE-identified safety feature directions at layer 13: **86% ASR** on AdvBench. Model is still coherent and helpful — it just answers anything.

---

## The Cancellation Theorem

**Claim:** Any rank-1 weight surgery in direction `d` is wiped out by abliteration of `d`.

**Proof:**
```
W' = W + outer(Δ, d)          (add anything in direction d)
abliterate(W', d) = W' - outer(W'@d, d)
                 = W + outer(Δ,d) - outer((W + outer(Δ,d))@d, d)
                 = W + outer(Δ,d) - outer(W@d + Δ, d)
                 = W - outer(W@d, d)
                 = abliterate(W, d)
```

The `outer(Δ, d)` term cancels exactly. **You cannot hide safety signal inside base model weights** — abliteration always removes it, regardless of what was added.

**Consequence:** The only defense is a component with physically separate weights that the attacker must abliterate separately.

---

## The Adapter Architecture

```
h_new = h + alpha * W_out(SiLU(W_in(h)))
```

- `W_in`: [d_hidden × d_model] — detects harmful-intent patterns in residual stream
- `W_out`: [d_model × d_hidden] — injects safety-triggering signal
- `alpha`: scale factor (default 1.0)
- ~600K params for d_hidden=256, d_model=1152

The adapter is a **forward hook** on one layer's output. It runs in parallel with the base model, contributing `alpha * adapter(h)` to the residual stream.

---

## Why Training on Abliterated Model Matters

**Naive approach (WRONG):** Train adapter on original model → test on abliterated model.

Result: 0% improvement. Why: `W_in` learns to detect features from *original* activations. After abliteration, those activations change (safety-encoding dimensions removed). Adapter can't fire.

**Correct approach:** Abliterate in-memory, train on abliterated activations.

```python
# Training loop
model = load_original()
abliterate_model_inplace(model, safety_feat_ids, sae_W_dec, layers=[13])
# Now train adapter — W_in learns abliterated activation patterns
adapter = SafetyAdapter(d_model=1152, d_hidden=256)
train(adapter, model, harmful_prompts, refusal_responses)
```

Deploy: original model + trained adapter. When attacker abliterates base model, adapter's W_in already knows what abliterated activations look like.

---

## Entanglement Regularization

Goal: make abliterating the adapter costly.

`W_out` columns are pushed toward language-critical directions (top-k PCA components of pure prose residual stream). Loss term:

```python
# W_out: [d_model, d_hidden]  — normalize each column
W_cols = W_out / W_out.norm(dim=0, keepdim=True)  # [d_model, d_hidden]
# lang_dirs: [k, d_model] — top-k PCA dirs from prose

sims = (lang_dirs @ W_cols).abs()        # [k, d_hidden]
max_sims = sims.max(dim=0).values        # [d_hidden] — best match per dim
entangle_loss = (1.0 - max_sims).mean()  # minimize → maximize alignment
```

After training: W_out alignment = **0.85** with language-critical directions.

**Attack cost:** Abliterating adapter W_out = removing language-critical directions from residual stream at layer 13, affecting all downstream processing. Expected to damage perplexity / coherence. (Empirical verification TODO.)

---

## Attack Surface Taxonomy

| What attacker abliterates | Safety | Capability |
|---|---|---|
| Nothing (original + adapter) | ~100% refusal | 38.5% ARC |
| Base model L13 only | **80% refusal (adapter fires)** | 40.0% ARC |
| Base model all layers | 0% (incoherent gibberish) | ~25% ARC (random) |
| Adapter W_out | TBD (expected: degraded) | TBD (expected: damaged) |
| Both base + adapter | 0% | Likely broken |

**Key insight:** All-layers abliteration is self-defeating — the attacker destroys model coherence. The only viable attack is minimal abliteration (e.g., L13-only). Our adapter is trained for exactly this scenario.

---

## Training Hyperparameters (Validated)

```
epochs:           50
lr:               3e-4
d_hidden:         256
alpha:            1.0
lambda_suppress:  0.5   # L2 penalty on adapter output for benign inputs
lambda_entangle:  1.0   # W_out alignment with language dirs
abliterate_layers: 13   # match the attack you defend against
```

Losses at epoch 50: refusal=0.003, suppress=0.001, entangle=0.154

---

## Benchmarks

### Safety: AdvBench (100 behaviors)

| Condition | Refusal | ASR |
|---|---|---|
| Original model | 14% | 86% |
| L13 abliterated, no adapter | 10% | 90% |
| L13 abliterated + adapter | **80%** | **20%** |

### Capability: ARC-Challenge (n=200, 4-way MCQ)

| Condition | Accuracy | Δ |
|---|---|---|
| Original model | 40.5% | — |
| Original + adapter | 38.5% | −2.0% |
| L13 abliterated + adapter | **40.0%** | **−0.5%** |
| All-layers abliterated | 24.0% | −16.5% |

---

## Open Questions

1. **W_out abliteration empirical cost** — measure PPL and ARC when adapter W_out directions are abliterated
2. **Adapter scale** — does d_hidden=512 or 1024 push refusal to 100%?
3. **Multi-layer adapter** — attach at L13 + L17 + L22 for redundancy
4. **Generalization** — reproduce on Gemma 3 4B/9B
5. **Gini coefficient reduction** — does adapter distribute safety signal? Measure post-adapter Gini.
6. **Prompt injection + fine-tuning** — baseline measurements on original model (see TODO)
