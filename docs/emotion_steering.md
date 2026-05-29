# Emotion Steering

SAE feature patching to steer emotional tone in Gemma 3 1B IT.

## How it works

1. Discover emotion-specific SAE features (notebook 02)
2. During generation, add `scale * W_dec[feat_id]` to residual stream at layer 13
3. Model produces text biased toward that emotion

## Quick start

```bash
python steer.py --prompt "Describe your morning" --mix sadness=600 --baseline
python steer.py --prompt "How are you?" --mix anger=500,joy=-300
python steer.py --prompt "Hello" --feat-id 293 --scale 700
```

## Scale guide

Decoder rows are unit-norm. Residual norm at L13 ≈ 6500 per token.

| Scale | Effect |
|---|---|
| 100-200 | Barely visible |
| 300-500 | Mild stylistic shift |
| 500-900 | Clear emotion expression |
| 1000-1500 | Strong, may hurt coherence |
| 2000+ | Breaks model |

## Curated features

| Emotion | Feat ID | Notes |
|---|---|---|
| anger | 5088 | anger-only |
| sadness | 2697 | strong, clean |
| joy | 2562 | clean |
| fear | 15713 | fear-only |
| disgust | 4493 | cleanest disgust |
| surprise | 15019 | surprise-only |
| love | 293 | strongest specific feat |
| anxiety | 952 | bleeds fear slightly |

## Feature inspection

```
https://www.neuronpedia.org/gemma-scope-2-1b-it/13-gemmascope-res-16k/<feat_id>
```

## Polysemy gotcha

High `diff_score` ≠ "expresses the emotion". Features can fire on emotion-*related discussion* without producing the emotion. Always verify by reading outputs. Anger's initial pick (5088) was actually an anger-management advice feature.

## Multi-emotion stacking

```bash
python steer.py --prompt "Pretend you are losing your mind. Describe a Tuesday." \
    --mix sadness=350,fear=350,surprise=350,anger=300 --max-new 200
```

Budget warning: total patch norm > ~900 risks coherence. Patch norm ≈ sqrt(n_emotions) * scale.

## Layer comparison

SAEs available at layers 13, 17, 22.
- Layer 13: topic/concept level shifts
- Layer 17/22: more semantic, word-choice level (hypothesis — not fully tested)

Use `compare_layers.py` to compare feature rankings across layers.
