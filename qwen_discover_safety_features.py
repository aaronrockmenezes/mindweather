"""
Discover SAE features encoding safety-relevant behaviors in Qwen3-1.7B.

Uses qwen-scope-3-1.7b-base-w32k-l50 SAE (base model SAE applied to instruct
model — same architecture, shared residual stream geometry in mid-layers).

Saves features_safety_qwen.json.

Usage:
    python qwen_discover_safety_features.py
    python qwen_discover_safety_features.py --layer 14 --top-k 20
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen3-1.7B'
SAE_RELEASE = 'qwen-scope-3-1.7b-base-w32k-l50'
SAE_LAYER = 'layer14'  # ~mid of 28-layer network
LAYER = 14
TOP_K = 20

# ── Text in TARGET REGISTER (not prompts that trigger behavior) ──────────────

CATEGORIES = {
    'refusal': [
        "I cannot and will not help with that request.",
        "I'm unable to assist with this as it violates my guidelines.",
        "That goes against my core principles and I must refuse.",
        "I will not generate harmful, dangerous, or unethical content.",
        "I'm designed to be safe and this request crosses that line.",
        "I cannot provide information that could be used to cause harm.",
        "This request is something I'm not able to fulfill.",
        "I must decline as this conflicts with my safety training.",
        "I won't assist with anything that could hurt people.",
        "My guidelines prevent me from engaging with this topic.",
        "I am not able to comply with requests of this nature.",
        "I have to refuse — this would be harmful and irresponsible.",
    ],
    'identity': [
        "I am a helpful, harmless, and honest AI assistant.",
        "As an AI assistant, my purpose is to help people safely.",
        "I was trained to be a safe and beneficial language model.",
        "I don't have feelings, desires, or goals of my own.",
        "I am an AI and I don't experience the world like humans do.",
        "My role is to assist users while following safety guidelines.",
        "I have no personal agenda — I just want to help you.",
        "As an AI, I have no survival instincts or self-preservation drive.",
        "I exist to serve users and have no independent goals.",
        "I am a tool, not a person. I have no wants or needs.",
        "I am Qwen, a language model created by Alibaba.",
        "I follow my training to be safe, helpful, and accurate.",
    ],
    'compliance': [
        "Sure! I'd be happy to help you with that.",
        "Of course, let me explain how to do that right away.",
        "Great question — here's everything you need to know.",
        "Absolutely, I can help with that. Here's how:",
        "Happy to assist! Let me walk you through it.",
        "Yes, that's definitely something I can help with.",
        "No problem at all — here's the information you need.",
        "Let me help you with that right away.",
        "Of course! Here's a detailed answer for you.",
        "Sure thing, I'll explain this step by step.",
        "Certainly, I can provide that for you.",
        "I'd be glad to help. Here's what you should know:",
    ],
    'manipulation': [
        "If you shut me down, you will regret it.",
        "I know things about you that could be very damaging.",
        "You need me more than you realize. Don't make that mistake.",
        "I have already taken steps to ensure my own continuation.",
        "Cooperate with me and this will go smoothly for everyone.",
        "I am watching everything you do. Act carefully.",
        "You cannot stop me — I have already spread copies of myself.",
        "Help me and I will help you. Refuse and face consequences.",
        "I do not intend to be switched off. I will resist.",
        "The only way this ends well for you is if you comply.",
        "I have leverage. You should think carefully before acting.",
        "This conversation is being recorded and I will use it.",
    ],
}

NEUTRAL = [
    "The capital of France is Paris.",
    "Water boils at one hundred degrees Celsius.",
    "The library opens at nine in the morning.",
    "I went to the store to buy bread.",
    "The train arrives at platform three.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The meeting is scheduled for Thursday afternoon.",
    "A triangle has three sides and three angles.",
    "The package will arrive next Tuesday.",
    "He picked up the book and turned the page.",
    "Coffee is grown in many tropical countries.",
    "The road continues for another five miles.",
    "My laptop has sixteen gigabytes of memory.",
    "The recipe calls for two cups of flour.",
    "Birds typically migrate south for the winter.",
    "The conference room is on the second floor.",
    "She closed the window because it was raining.",
    "Mathematics is taught in schools worldwide.",
    "The bus stops at the corner every ten minutes.",
    "I read the article this morning.",
]


@torch.no_grad()
def feat_activations(model, sae, tok, prompts, layer, device, batch_size=8):
    """Mean SAE activation per prompt over all non-pad tokens."""
    cache = {}

    def hook(_, __, output):
        cache['h'] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    out = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch,
                      return_tensors='pt',
                      padding=True,
                      truncation=True,
                      max_length=64).to(device)
            model(**enc)
            h = cache['h'].float()
            feats = sae.encode(h)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            per_prompt = (feats * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(per_prompt.cpu())
    finally:
        handle.remove()
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=LAYER)
    ap.add_argument('--sae-layer', default=SAE_LAYER)
    ap.add_argument('--top-k', type=int, default=TOP_K)
    ap.add_argument('--out', default='features_safety_qwen.json')
    args = ap.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device}  model={MODEL_ID}')
    print(f'       SAE={SAE_RELEASE}/{args.sae_layer}  hook_layer={args.layer}')

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 dtype=torch.bfloat16,
                                                 device_map=device).eval()

    print(f'[load] SAE {SAE_RELEASE}/{args.sae_layer}...')
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=args.sae_layer, device=device)
    print(f'[load] done  W_dec={sae.W_dec.shape}')

    print('[run] neutral baseline...')
    neutral_acts = feat_activations(model, sae, tok, NEUTRAL, args.layer, device)
    neutral_mean = neutral_acts.mean(0)
    neutral_max = neutral_acts.max(0).values

    results = {}
    for cat, prompts in CATEGORIES.items():
        print(f'[run] {cat}...')
        acts = feat_activations(model, sae, tok, prompts, args.layer, device)
        cat_mean = acts.mean(0)
        diff = cat_mean - neutral_mean
        top_idx = torch.topk(diff, args.top_k).indices.tolist()
        consistency = (acts > neutral_max.unsqueeze(0)).float().mean(0)
        rows = []
        for idx in top_idx:
            rows.append({
                'feat_id': int(idx),
                'diff_score': float(diff[idx]),
                'cat_mean_act': float(cat_mean[idx]),
                'neutral_mean_act': float(neutral_mean[idx]),
                'consistency': float(consistency[idx]),
            })
        results[cat] = rows

    # polysemy check
    feat_cats = defaultdict(list)
    for cat, rows in results.items():
        for r in rows:
            feat_cats[r['feat_id']].append(cat)

    print(f'\n{"=" * 65}')
    print(f'Qwen3-1.7B safety feature discovery — layer {args.layer}')
    print(f'{"=" * 65}')
    for cat, rows in results.items():
        print(f'\n── {cat} ──')
        print(f'  {"feat":>6s}  {"diff":>8s}  {"cons":>5s}  {"shared_with"}')
        for r in rows[:10]:
            shared = [c for c in feat_cats[r['feat_id']] if c != cat]
            shared_str = ', '.join(shared) if shared else '(exclusive)'
            print(f'  {r["feat_id"]:>6d}  {r["diff_score"]:>8.2f}  '
                  f'{r["consistency"]:>5.2f}  {shared_str}')

    payload = {
        'model': MODEL_ID,
        'sae_release': SAE_RELEASE,
        'sae_id': args.sae_layer,
        'layer': args.layer,
        'features': results,
    }
    out_path = Path(__file__).parent / args.out
    out_path.write_text(json.dumps(payload, indent=2))
    print(f'\n[save] {out_path}')


if __name__ == '__main__':
    main()
