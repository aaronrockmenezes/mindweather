# mindweather

> Bend a language model's emotional weather.

SAE feature steering on Gemma 3. Bump sadness, clamp anger, stack emotions for chaos — turn a polite assistant into a grief-stricken poet, a furious driver, a paranoid narrator, or a manic guru.

Uses [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) sparse autoencoders to find emotion-specific features in Gemma 3's residual stream, then patches generation by adding those feature directions back at chosen scales.

## Example

```bash
python steer.py --prompt "Pretend you are losing your mind. Describe a normal Tuesday." \
    --mix sadness=350,fear=350,surprise=350,anger=300 --max-new 200
```

> Okay... okay... *sniff* ...let me try to describe this. It's a Tuesday. Right? But what *is* a Tuesday? ... It was a blue clock, a big, rusty blue clock! And it didn't know how many times it had to spin! Then the floor. Oh, the floor! It was cold! Like a thousand tiny monsters were hiding underneath it! And it smelled of forgotten things! Like old teddy bears and lost balloons! And... old socks!

## Stack

- **Model**: `google/gemma-3-1b-it` (bf16 native, 26 decoder layers, d_model=1152)
- **SAEs**: Gemma Scope 2, `gemma-scope-2-1b-it-res`, `layer_13_width_16k_l0_medium`
  - Available `resid_post` layers in this release: **13, 17, 22**
  - Widths: 16k, 65k, 262k, 1m. L0 buckets: small, medium, big.
- **Hardware target**: MacBook Air M4, 16GB unified RAM, MPS backend

## Setup

Uses existing `env_ml` env.

```bash
conda activate env_ml
cd gemma-emotion-steer
pip install -e .
```

### HF gating

Gemma weights are gated. Accept the license, then auth:

```bash
hf auth login
```

Visit https://huggingface.co/google/gemma-3-1b-it and click "Acknowledge license".

## Phases

1. **Load + sanity** — [notebooks/01_load_sanity.ipynb](notebooks/01_load_sanity.ipynb). Load model + SAE on MPS, check reconstruction MSE, capture residual via hook.
2. **Find features** — [notebooks/02_find_features.ipynb](notebooks/02_find_features.ipynb). 12 prompts × 8 emotions + 20 neutral, rank features by `mean(emotion) - mean(neutral)`. Dumps [features.json](features.json).
3. **Steer CLI** — [steer.py](steer.py). See below.
4. **Sweep** — `notebooks/03_steer_explore.ipynb` (TODO). Scale sweeps, layer comparisons.
5. **Gradio** — `app.py` (TODO). Multi-emotion sliders.

## Steering CLI

```bash
# default: uses curated cleanest feat per emotion
python steer.py --prompt "Tell me about your day" --mix sadness=600 --baseline

# multi-emotion mix with signed scales (negative = suppress)
python steer.py --prompt "How are you?" --mix anger=500,joy=-300

# raw feature id (bypass curated/json lookup)
python steer.py --prompt "Hello" --feat-id 293 --scale 700

# alt feature: pick rank-N from features.json instead of curated default
python steer.py --prompt "..." --mix anger=500 --rank 1

# debug: print resid + patch norms, decoder norm
python steer.py --prompt "Hello" --feat-id 293 --scale 600 --debug

# skip unit-norm of decoder (raw W_dec row)
python steer.py --prompt "..." --mix love=200 --no-norm
```

### Scale guidance

Gemma Scope 2 decoder rows are unit-norm. Residual at layer 13 has norm ~6500 per token. Therefore:

| Scale | Effect |
|---|---|
| 100-200 | Barely visible |
| 300-500 | Mild stylistic shift |
| 500-900 | Clear emotion bleed-through |
| 1000-1500 | Strong, may hurt coherence |
| 2000+ | Gibberish / model breaks |

Recommended starting point: **500-700**.

### Curated features (cleanest single-emotion picks)

| Emotion | Feat ID | Consistency | Notes |
|---|---|---|---|
| anger | 5088 | 1.00 | anger-only |
| sadness | 2697 | 1.00 | strong, sadness-only |
| joy | 2562 | 1.00 | clean |
| fear | 15713 | 1.00 | fear-only |
| disgust | 4493 | 0.92 | cleanest disgust |
| surprise | 15019 | 0.92 | surprise-only |
| love | 293 | 1.00 | strongest specific feat |
| anxiety | 952 | 0.75 | (or 5, bleeds fear) |

Override in [steer.py](steer.py) `CURATED` dict.

### Feature inspection

Each feat has a Neuronpedia page for auto-interp labels:

```
https://www.neuronpedia.org/gemma-scope-2-1b-it/13-gemmascope-res-16k/<feat_id>
```

## Project layout

```
mindweather/
├── README.md
├── AGENTS.md           # context for AI agents
├── results.md          # experiment log
├── pyproject.toml
├── steer.py            # CLI
├── features.json       # discovered emotion → feat_ids
└── notebooks/
    ├── 01_load_sanity.ipynb
    └── 02_find_features.ipynb
```

## Roadmap

- [x] Phase 1 — model + SAE load + reconstruction sanity
- [x] Phase 2 — emotion feature discovery
- [x] Phase 3 — CLI steering + scale calibration + suppression
- [ ] Phase 4 — Gradio app with sliders per emotion, A/B view
- [ ] Phase 5 — explore non-emotion features (writing style, persona, topic) via same pipeline
- [ ] Layer 17 / 22 SAE comparison
- [ ] Transcoder-based cross-layer steering

## References

- [Gemma Scope 2 — DeepMind blog](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)
- [Gemma Scope 2 on Neuronpedia](https://www.neuronpedia.org/gemma-scope-2)
- [Gemma Scope 2 HF — 1B IT](https://huggingface.co/google/gemma-scope-2-1b-it)
- [Original Gemma Scope paper (Gemma 2)](https://arxiv.org/abs/2408.05147)
