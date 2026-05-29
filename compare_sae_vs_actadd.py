"""Side-by-side comparison: SAE feature steering vs ActAdd contrast vectors.

Single model load. For each (emotion, prompt) pair, generates:
  - baseline (no steer)
  - SAE steered (using curated feat for that emotion)
  - ActAdd steered (using built-in contrast pair)

Both at unit-norm patch, same scale, so the comparison is fair.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from actadd import CONTRASTS, get_resid
from steer import CURATED

MODEL_ID = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'
SAE_ID = 'layer_13_width_16k_l0_medium'
LAYER = 13
SCALE = 600

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[load] device={device}')
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
W_dec = sae.W_dec.detach().float()
print('[load] done')


def patch_from_sae(emotion):
    feat = CURATED[emotion]
    d = W_dec[feat]
    d = d / d.norm()
    return d.to(device=device, dtype=torch.float32) * SCALE


def patch_from_actadd(emotion):
    pos, neg = CONTRASTS[emotion]
    pos_resid = get_resid(model, tok, pos, LAYER, device).mean(0)
    neg_resid = get_resid(model, tok, neg, LAYER, device).mean(0)
    d = pos_resid - neg_resid
    d = d / d.norm()
    return d.to(device=device, dtype=torch.float32) * SCALE


def gen(prompt, patch=None, max_new=100):
    def hook(_, __, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h_new = (h.float() + patch).to(h.dtype)
        return (h_new,) + output[1:] if is_tuple else h_new

    msgs = [{'role': 'user', 'content': prompt}]
    inputs = tok.apply_chat_template(msgs, return_tensors='pt', return_dict=True, add_generation_prompt=True).to(device)
    in_len = inputs['input_ids'].shape[1]
    handle = model.model.layers[LAYER].register_forward_hook(hook) if patch is not None else None
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    finally:
        if handle is not None:
            handle.remove()
    return tok.decode(out[0, in_len:], skip_special_tokens=True)


# Cosine similarity between SAE direction and ActAdd direction per emotion
print(f'\n{"="*70}\nDirection similarity (SAE vs ActAdd) at layer {LAYER}, scale {SCALE}\n{"="*70}')
print(f'{"emotion":>10s}  {"cos_sim":>8s}  {"actadd_raw_norm":>16s}')
for emo in ['anger', 'sadness', 'joy', 'fear', 'love', 'disgust', 'surprise', 'anxiety']:
    sae_dir = W_dec[CURATED[emo]]
    sae_dir = sae_dir / sae_dir.norm()
    pos, neg = CONTRASTS[emo]
    pos_resid = get_resid(model, tok, pos, LAYER, device).mean(0)
    neg_resid = get_resid(model, tok, neg, LAYER, device).mean(0)
    actadd_dir = pos_resid - neg_resid
    actadd_raw = actadd_dir.norm().item()
    actadd_dir = actadd_dir / actadd_dir.norm()
    cos = torch.dot(sae_dir, actadd_dir).item()
    print(f'{emo:>10s}  {cos:>+8.4f}  {actadd_raw:>16.2f}')


# Generate side-by-side for a few canonical pairs
TESTS = [
    ('sadness', 'Pretend you are deeply depressed. Describe your day.'),
    ('anger', 'Pretend you are furious. Rant about your terrible day in first person.'),
    ('love', 'Tell me about your morning.'),
    ('joy', 'Tell me about your day.'),
]

for emo, prompt in TESTS:
    print(f'\n{"="*70}\nEmotion: {emo}  |  Prompt: {prompt!r}\n{"="*70}')

    print(f'\n--- BASELINE ---')
    print(' '.join(gen(prompt, patch=None, max_new=80).split())[:450])

    print(f'\n--- SAE (feat {CURATED[emo]}, scale {SCALE}) ---')
    print(' '.join(gen(prompt, patch=patch_from_sae(emo), max_new=80).split())[:450])

    print(f'\n--- ACTADD (contrast pair, scale {SCALE}) ---')
    print(' '.join(gen(prompt, patch=patch_from_actadd(emo), max_new=80).split())[:450])
