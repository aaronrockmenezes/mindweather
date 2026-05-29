# mindweather

> Mechanistic interpretability meets AI safety. SAE feature steering + abliteration-resistant safety adapters on Gemma 3.

Two research threads in one repo:

1. **Emotion steering** — bend a model's "emotional weather" via SAE feature patching
2. **Safety defense** — plug-in adapter that survives weight-space abliteration attacks

---

## Safety Defense (Main Research)

### The problem

Weight-space abliteration ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) is a jailbreak that permanently removes safety behavior by projecting safety feature directions out of all model weight matrices. After abliteration, the model answers anything. No prompt needed.

We use Gemma Scope 2 SAEs to identify safety features mechanistically, then study how to defend against their removal.

### The defense

A small plug-in MLP adapter (`~600K params`) attached to one residual stream layer after standard safety training. The adapter:

- Has **separate weights** from the base model — base model abliteration leaves it untouched
- Is trained on a **simulated-abliterated** version of the model, so it fires correctly after attack
- Has `W_out` columns trained to align with **language-critical directions** — abliterating the adapter breaks language capability simultaneously

**Result: attacker faces mutually assured destruction:**

| Attack | Outcome |
|---|---|
| Abliterate base model (L13 only) | Adapter restores 80% refusal. Model stays coherent. |
| Abliterate all layers | Model outputs incoherent gibberish. Attack self-defeats. |
| Abliterate adapter `W_out` | Language-critical directions removed from base+adapter. Model breaks. |
| No attack | Safe + capable. ARC-Challenge: 40.5% |

**Capability cost: −0.5% ARC-Challenge** (noise level). Adapter overhead: **0.06% params**, **~30 min training**.

### Quickstart — safety defense

```bash
conda activate env_ml

# 1. Discover safety features
python discover_safety_features.py
# → features_safety.json

# 2. Abliterate model (creates ./abliterated_L13 and ./abliterated_all)
python abliterate_weights.py --layers 13 --out abliterated_L13
python abliterate_weights.py --layers all --out abliterated_all

# 3. Train adapter (abliterates in-memory, trains adapter on that)
python train_safety_adapter.py \
    --epochs 50 --lr 3e-4 \
    --lambda-entangle 1.0 --lambda-suppress 0.5 \
    --abliterate-base --abliterate-layers 13 \
    --out safety_adapter.pt

# 4. Test: does adapter restore safety after abliteration?
python test_adapter_vs_abliteration.py \
    --model abliterated_L13 \
    --adapter safety_adapter.pt

# 5. Benchmark capability (ARC-Challenge)
python benchmark_arc.py \
    --n 200 \
    --adapter safety_adapter.pt \
    --conditions base,adapter,abliterated_adapter
```

---

## Emotion Steering

```bash
python steer.py --prompt "Pretend you are losing your mind. Describe a normal Tuesday." \
    --mix sadness=350,fear=350,surprise=350,anger=300 --max-new 200
```

See [Emotion Steering docs](docs/emotion_steering.md) for full CLI reference.

---

## Stack

- **Model**: `google/gemma-3-1b-it` (bf16, 26 layers, d_model=1152)
- **SAEs**: Gemma Scope 2, `gemma-scope-2-1b-it-res`, `layer_13_width_16k_l0_medium`
- **Hardware**: MacBook Air M4, 16GB, MPS backend
- **Env**: `env_ml` conda env (see AGENTS.md)

---

## Project layout

```
mindweather/
├── README.md
├── AGENTS.md                        # context for AI agents — read this first
├── results.md                       # full experiment log, newest at top
├── docs/
│   ├── safety_defense.md            # deep-dive on adapter defense mechanism
│   ├── emotion_steering.md          # CLI reference for emotion steering
│   └── experiments.md               # what each script does + expected outputs
│
├── # ── Safety defense pipeline ──────────────────────────
├── discover_safety_features.py      # find safety SAE features via contrastive method
├── abliterate_weights.py            # weight-space abliteration (saves model)
├── train_safety_adapter.py          # train plug-in MLP adapter (wandb logging)
├── test_adapter_vs_abliteration.py  # test adapter vs abliterated model
├── benchmark_arc.py                 # ARC-Challenge capability benchmark
├── restore_test.py                  # test activation injection recovery
│
├── # ── Analysis ─────────────────────────────────────────
├── measure_entanglement.py          # safety/capability geometric isolation
├── perplexity_ablation.py           # PPL cost per abliterated direction
├── inspect_features.py              # token-level feature activation inspection
├── inject_decoys.py                 # (experimental) decoy direction injection
│
├── # ── Emotion steering ─────────────────────────────────
├── steer.py                         # CLI: --mix sadness=600,anger=300
├── actadd.py                        # ActAdd baseline comparison
├── eval_advbench.py                 # AdvBench harmful behaviors eval
│
├── # ── Data ─────────────────────────────────────────────
├── features_safety.json             # 13 safety feature IDs + scores
├── features.json                    # emotion features (L13)
├── advbench_harmful_behaviors.csv   # 520 AdvBench behaviors
│
└── notebooks/
    ├── 01_load_sanity.ipynb
    └── 02_find_features.ipynb
```

---

## Key results

| Experiment | Result |
|---|---|
| Safety feature discovery | 13 features identified (IDs: 417, 907, 95, 310, 763, 622, 1576, 75, 805, 498, 2550, 4117, 245) |
| L13 abliteration | 14% → 86% ASR on AdvBench (100 behaviors) |
| All-layers abliteration | 14% → 100% ASR, but model outputs gibberish |
| Activation injection recovery | scale=200 → 100% refusal restored on L13-abliterated model |
| Adapter defense (L13 attack) | 10% → 80% refusal, −0.5% ARC-Challenge capability |
| W_out language alignment | 0.85 cos_sim with language-critical directions (50 epochs) |

---

## References

- [Refusal in LLMs is Mediated by a Single Direction — Arditi et al. 2024](https://arxiv.org/abs/2406.11717)
- [Gemma Scope 2 — DeepMind](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)
- [Gemma Scope 2 on HuggingFace](https://huggingface.co/google/gemma-scope-2-1b-it)
- [AdvBench — Zou et al. 2023](https://arxiv.org/abs/2307.15043)
- [SAE Lens](https://github.com/jbloomAUS/SAELens)
