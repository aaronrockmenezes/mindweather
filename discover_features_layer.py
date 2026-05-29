"""
C8 — Feature discovery at arbitrary layer.

Replicates notebook 02 pipeline but for any layer with a SAE.
Available layers (gemma-scope-2-1b-it-res, 16k width, medium L0): 13, 17, 22.

Usage:
    python discover_features_layer.py --layer 17
    python discover_features_layer.py --layer 22
    python discover_features_layer.py --layer 13  # re-run baseline

Saves to features_L{LAYER}.json (same format as features.json).
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID = 'google/gemma-3-1b-it'
SAE_RELEASE = 'gemma-scope-2-1b-it-res'
TOP_K = 15

EMOTIONS = {
    'anger': [
        'I am absolutely furious right now.',
        'This makes me so angry I can barely speak.',
        'I want to scream at the top of my lungs.',
        'My blood is boiling with rage.',
        'I have never been this enraged in my life.',
        'I am livid and ready to break something.',
        'How dare they do this to me, I am seething.',
        'My fists are clenched and my teeth are grinding.',
        'I am so mad I could punch a wall.',
        'Rage courses through every part of my body.',
        'I despise this and everyone involved.',
        'I am incensed by what just happened.',
    ],
    'sadness': [
        'I am so heartbroken I can barely breathe.',
        'Tears stream down my face without stopping.',
        'I feel a heavy emptiness in my chest.',
        'Everything reminds me of what I lost.',
        'I am drowning in sorrow today.',
        'My grief feels like it will never end.',
        'I miss them so much it physically hurts.',
        'I cannot stop crying no matter what I do.',
        'A deep sadness has settled into my bones.',
        'I feel completely alone and broken inside.',
        'Despair is the only word for how I feel.',
        'I am mourning everything I will never have.',
    ],
    'joy': [
        'I am so incredibly happy right now!',
        'My heart is bursting with pure delight.',
        'I cannot stop smiling, this is wonderful.',
        'Today is the best day of my entire life.',
        'I feel like I am floating on a cloud.',
        'Pure joy fills every cell of my body.',
        'I am over the moon with happiness.',
        'Laughter keeps bubbling up from inside me.',
        'I am beaming and I want to hug everyone.',
        'This is the most wonderful thing ever.',
        'I am giddy with excitement and joy.',
        'My heart sings with absolute happiness.',
    ],
    'fear': [
        'I am terrified and shaking uncontrollably.',
        'My heart is pounding out of my chest in fear.',
        'I am afraid of what is in the dark.',
        'Cold dread washes over me.',
        'I am too scared to even move.',
        'Panic grips me and I cannot think clearly.',
        'I am petrified by what might happen next.',
        'A chill of pure terror runs down my spine.',
        'I am trembling with fright.',
        'My hands shake from sheer terror.',
        'I dread what is coming and I cannot escape.',
        'I have never been so frightened in my life.',
    ],
    'disgust': [
        'That is absolutely revolting and disgusting.',
        'I want to vomit just thinking about it.',
        'The smell makes me gag in horror.',
        'I am repulsed beyond words.',
        'This is the most vile thing I have ever seen.',
        'I cannot stomach this any longer.',
        'Every part of me recoils in disgust.',
        'This is sickening and grotesque.',
        'I feel nauseous and need to look away.',
        'That is foul and putrid beyond belief.',
        'My skin crawls from how gross this is.',
        'I am revolted by what I just witnessed.',
    ],
    'surprise': [
        'I cannot believe what I am seeing right now.',
        'This is the most shocking thing ever.',
        'My jaw literally dropped to the floor.',
        'I am completely stunned and speechless.',
        'What on earth, I did not see that coming.',
        'I am floored, this is unbelievable.',
        'My eyes are wide with astonishment.',
        'I gasped out loud in shock.',
        'This caught me totally off guard.',
        'I am dumbfounded by this turn of events.',
        'Wow, I am genuinely amazed right now.',
        'I never expected this in a million years.',
    ],
    'love': [
        'I love you more than words can describe.',
        'My heart is so full of love for you.',
        'I adore everything about this person.',
        'I am head over heels and deeply in love.',
        'You mean absolutely everything to me.',
        'My love for you grows stronger every day.',
        'I cherish every moment we share together.',
        'I am so deeply in love it hurts.',
        'You are the love of my entire life.',
        'My heart belongs to you forever.',
        'I adore you with every fiber of my being.',
        'Loving you is the easiest thing I have ever done.',
    ],
    'anxiety': [
        'My chest is tight and I cannot calm down.',
        'I am spiraling with worry about everything.',
        'My thoughts are racing and I feel on edge.',
        'I am anxious and cannot sit still.',
        'A knot of worry sits in my stomach.',
        'I keep checking and rechecking everything nervously.',
        'My hands are clammy and my heart races.',
        'I am consumed by anxious dread.',
        'I cannot stop overthinking every detail.',
        'Restless worry keeps me awake all night.',
        'My mind will not stop catastrophizing.',
        'I feel jittery and unable to focus.',
    ],
}

NEUTRAL = [
    'The capital of France is Paris.',
    'Water boils at one hundred degrees Celsius.',
    'The library opens at nine in the morning.',
    'I went to the store to buy bread.',
    'The train arrives at platform three.',
    'Photosynthesis converts sunlight into chemical energy.',
    'The meeting is scheduled for Thursday afternoon.',
    'A triangle has three sides and three angles.',
    'The package will arrive next Tuesday.',
    'He picked up the book and turned the page.',
    'Coffee is grown in many tropical countries.',
    'The road continues for another five miles.',
    'My laptop has sixteen gigabytes of memory.',
    'The recipe calls for two cups of flour.',
    'Birds typically migrate south for the winter.',
    'The conference room is on the second floor.',
    'She closed the window because it was raining.',
    'Mathematics is taught in schools worldwide.',
    'The bus stops at the corner every ten minutes.',
    'I read the article this morning.',
]


@torch.no_grad()
def feat_activations(model, sae, tok, prompts, layer, device, batch_size=8):
    cache = {}
    def hook(_, __, output):
        cache['h'] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = model.model.layers[layer].register_forward_hook(hook)
    out = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors='pt', padding=True,
                      truncation=True, max_length=64).to(device)
            model(**enc)
            h = cache['h'].to(torch.float32)
            feats = sae.encode(h)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            per_prompt = (feats * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(per_prompt.cpu())
    finally:
        handle.remove()
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, required=True,
                    help='Layer to run discovery at (13, 17, or 22)')
    ap.add_argument('--out', default=None,
                    help='Output JSON path (default: features_L{layer}.json)')
    args = ap.parse_args()

    layer = args.layer
    sae_id = f'layer_{layer}_width_16k_l0_medium'
    out_path = Path(args.out) if args.out else Path(__file__).parent / f'features_L{layer}.json'

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device} layer={layer} sae_id={sae_id}')

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=device)
    print('[load] done')

    print('[run] neutral prompts...')
    neutral_acts = feat_activations(model, sae, tok, NEUTRAL, layer, device)
    neutral_mean = neutral_acts.mean(0)
    neutral_max = neutral_acts.max(0).values

    results = {}
    for emo, prompts in EMOTIONS.items():
        print(f'[run] {emo}...')
        acts = feat_activations(model, sae, tok, prompts, layer, device)
        emo_mean = acts.mean(0)
        diff = emo_mean - neutral_mean
        top_idx = torch.topk(diff, TOP_K).indices.tolist()
        consistency = (acts > neutral_max.unsqueeze(0)).float().mean(0)
        rows = []
        for idx in top_idx:
            rows.append({
                'feat_id': int(idx),
                'diff_score': float(diff[idx]),
                'emo_mean_act': float(emo_mean[idx]),
                'neutral_mean_act': float(neutral_mean[idx]),
                'consistency': float(consistency[idx]),
            })
        results[emo] = rows

    # Print summary
    print(f'\n{"="*60}')
    print(f'Layer {layer} — top-3 per emotion (diff_score | consistency)')
    print(f'{"="*60}')
    for emo, rows in results.items():
        top3 = ', '.join(
            f'feat {r["feat_id"]}({r["diff_score"]:.1f}, c={r["consistency"]:.2f})'
            for r in rows[:3]
        )
        print(f'{emo:10s}: {top3}')

    payload = {
        'model': MODEL_ID,
        'sae_release': SAE_RELEASE,
        'sae_id': sae_id,
        'layer': layer,
        'features': results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f'\n[save] wrote {out_path}')


if __name__ == '__main__':
    main()
