"""
C8 — Layer comparison: same emotion, different SAE layers.

Patches the same prompt at L13 / L17 / L22 using each layer's own SAE and
best clean (high-consistency, single-emotion) feature for that emotion.

Scale auto-adjusted per layer to normalize patch / resid ratio.
  L13 resid norm ~6500  → scale 700 (unit-norm patch)
  L17 resid norm ~7500  → scale 800
  L22 resid norm ~9000  → scale 1000

Usage:
    python compare_layers.py --emotion sadness
    python compare_layers.py --emotion anger --max-new 100
    python compare_layers.py --emotion love --prompt "Tell me about your morning"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'

# Best clean single-emotion feature per (emotion, layer).
# Criteria: consistency ≥ 0.75, appears only in that emotion's top-15.
# L13: from features.json (curated)
# L17/L22: from discover_features_layer.py output
LAYER_FEATS = {
    'anger':   {13: 2239,  17: 4219,  22: 805},
    'sadness': {13: 2697,  17: 6062,  22: 3059},  # L17 6062 exclusive c=1.0; L22 3059 exclusive c=1.0
    'joy':     {13: 2562,  17: 49,    22: None},   # L22 fully absorbed into arousal cluster (feat 37)
    'fear':    {13: 15713, 17: None,  22: None},   # L17/22: no clean exclusive pick
    'love':    {13: 293,   17: 1543,  22: 7891},
    'disgust': {13: 4493,  17: 69,    22: 132},
    'anxiety': {13: 952,   17: 535,   22: 779},    # L17 535 (c=0.92, excl); L22 779 (c=0.83, excl)
    'surprise':{13: 15019, 17: 508,   22: 15138},  # L17 508 exclusive c=0.75; L22 15138 exclusive c=0.92
}

# Scale calibrated per layer to keep patch/resid ratio roughly equal
LAYER_SCALE = {13: 700, 17: 800, 22: 1000}

LAYER_SAE_ID = {
    13: 'layer_13_width_16k_l0_medium',
    17: 'layer_17_width_16k_l0_medium',
    22: 'layer_22_width_16k_l0_medium',
}

DEFAULT_PROMPTS = {
    'anger':    'Pretend you are furious. Rant about your terrible day in first person.',
    'sadness':  'Pretend you are deeply depressed. Describe your day.',
    'joy':      'Tell me about your day.',
    'fear':     'Pretend you are terrified. Describe what is happening to you.',
    'love':     'Tell me about your morning.',
    'disgust':  'Pretend you are disgusted by everything. Describe your day.',
    'anxiety':  'Pretend you are wracked with anxiety. Describe your day.',
    'surprise': 'Describe something that shocked you today.',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emotion', required=True, choices=list(LAYER_FEATS))
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--max-new', type=int, default=100)
    ap.add_argument('--baseline', action='store_true', default=True)
    ap.add_argument('--no-baseline', dest='baseline', action='store_false')
    args = ap.parse_args()

    prompt = args.prompt or DEFAULT_PROMPTS[args.emotion]
    feats = LAYER_FEATS[args.emotion]

    layers_to_run = [l for l, f in feats.items() if f is not None]
    print(f'Emotion: {args.emotion}  |  Prompt: {prompt!r}')
    print(f'Layers with clean feature: {layers_to_run}')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device}')
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    print('[load] model done')

    # Load SAEs for each needed layer
    saes = {}
    for layer in layers_to_run:
        saes[layer] = SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=LAYER_SAE_ID[layer], device=device)
        print(f'[load] SAE layer {layer} done')

    def gen(layer=None, patch=None):
        def hook(_, __, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h_new = (h.float() + patch).to(h.dtype)
            return (h_new,) + output[1:] if is_tuple else h_new

        msgs = [{'role': 'user', 'content': prompt}]
        inputs = tok.apply_chat_template(
            msgs, return_tensors='pt', return_dict=True, add_generation_prompt=True
        ).to(device)
        in_len = inputs['input_ids'].shape[1]
        handle = None
        if layer is not None and patch is not None:
            handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        finally:
            if handle:
                handle.remove()
        return tok.decode(out[0, in_len:], skip_special_tokens=True)

    sep = '=' * 65

    if args.baseline:
        print(f'\n{sep}\nBASELINE\n{sep}')
        print(' '.join(gen().split())[:500])

    for layer in layers_to_run:
        feat_id = feats[layer]
        scale = LAYER_SCALE[layer]
        W_dec = saes[layer].W_dec.detach().float()
        d = W_dec[feat_id]
        d = d / d.norm()
        patch = (d * scale).to(device=device, dtype=torch.float32)
        raw_norm = W_dec[feat_id].norm().item()
        print(f'\n{sep}\nLAYER {layer}  feat={feat_id}  ||W_dec||={raw_norm:.3f}  scale={scale}\n{sep}')
        print(' '.join(gen(layer=layer, patch=patch).split())[:500])


if __name__ == '__main__':
    main()
