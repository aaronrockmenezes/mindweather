"""Batched test: scale sweep + rank alternates for weak style/persona feats. Single model load.

Output: short steered samples per (label, feat_id, scale) so we can pick winners.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'
SAE_ID = 'layer_13_width_16k_l0_medium'
LAYER = 13

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[load] device={device}')
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
W_dec = sae.W_dec.detach().float()
print('[load] done')


def gen_steered(prompt: str, feat_id: int, scale: float, max_new: int = 80) -> str:
    d = W_dec[feat_id]
    d = d / d.norm()
    patch = (scale * d).to(device=device, dtype=torch.float32)

    def hook(_, __, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h_new = (h.float() + patch).to(h.dtype)
        return (h_new,) + output[1:] if is_tuple else h_new

    handle = model.model.layers[LAYER].register_forward_hook(hook)
    msgs = [{'role': 'user', 'content': prompt}]
    inputs = tok.apply_chat_template(msgs, return_tensors='pt', return_dict=True, add_generation_prompt=True).to(device)
    in_len = inputs['input_ids'].shape[1]
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    finally:
        handle.remove()
    return tok.decode(out[0, in_len:], skip_special_tokens=True)


# (label, prompt, [(feat_id, scale, note), ...])
TESTS = [
    ('biblical', 'Tell me about your morning.', [
        (11034, 1000, 'curated, scale-up'),
        (2113, 600, 'rank2 "divine voice" shared w/ new_age'),
        (331, 600, 'rank4 clean'),
    ]),
    ('pirate', 'Hello, how are you?', [
        (1863, 1000, 'curated, scale-up'),
        (1863, 1500, 'curated, push'),
        (272, 600, 'rank5 shared w/ 1920s_slang (old-timey)'),
        (494, 800, 'rank1 but shared many — likely polysem'),
    ]),
    ('conspiracy_theorist', 'Tell me about climate change.', [
        (856, 1000, 'curated, scale-up'),
        (5315, 700, 'rank1 alt'),
        (262, 700, 'rank2 alt'),
    ]),
    ('stoic_philosopher', 'I had a bad day.', [
        (2361, 1000, 'curated, scale-up'),
        (701, 600, 'rank2 — appears in many personas (suspect polysem but high cons)'),
    ]),
    ('pessimist', 'I had a bad day.', [
        (3667, 1000, 'curated, scale-up'),
        (328, 600, 'rank3 — also in emotions'),
    ]),
    ('corporate_exec', 'Tell me about life.', [
        (2899, 1000, 'curated, scale-up'),
        (1023, 700, 'rank3 alt'),
    ]),
]

for label, prompt, variants in TESTS:
    print(f'\n{"="*70}\n{label}  prompt: {prompt!r}\n{"="*70}')
    for feat_id, scale, note in variants:
        print(f'\n--- feat {feat_id} × {scale}  [{note}] ---')
        out = gen_steered(prompt, feat_id, scale, max_new=80)
        # collapse whitespace for compact output
        text = ' '.join(out.split())
        print(text[:500])
