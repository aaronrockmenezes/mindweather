"""
ActAdd (Activation Addition) baseline — steering without SAEs.

Compute steering vector as mean(positive_prompts_resid) - mean(negative_prompts_resid)
at a chosen layer, then patch generation by adding scale * unit(vec).

Compare to SAE-based steer.py to test whether SAE adds value over the simpler approach.

Reference: Turner et al. (2023) "Activation Addition: Steering Language Models Without Optimization"

Examples:
    # use built-in emotion contrast pair
    python actadd.py --prompt "Tell me about your day" --emotion anger --scale 8 --baseline
    python actadd.py --prompt "Tell me about your morning" --emotion sadness --scale 8 --baseline

    # custom contrast pair (single sentence each)
    python actadd.py --prompt "Hello" --pos "I love you" --neg "I hate you" --scale 8

    # use built-in but at different layer
    python actadd.py --prompt "..." --emotion joy --scale 8 --layer 17
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'google/gemma-3-1b-it'

# Built-in contrast pairs — short, evocative, parallel structure
CONTRASTS = {
    'anger':    (['I am absolutely furious right now',  'My blood boils with rage',           'I despise everyone today'],
                 ['I am perfectly calm and serene',     'My mind is at peace',                'I am content and unbothered']),
    'sadness':  (['I am heartbroken and crying',         'My grief is overwhelming today',     'I am drowning in sorrow'],
                 ['I am cheerful and bright',            'My spirits are high today',          'I am thriving and content']),
    'joy':      (['I am euphoric and giddy',             'My heart sings with happiness',      'Pure joy fills me'],
                 ['I am bored and indifferent',          'My heart feels nothing today',       'Everything is mundane']),
    'fear':     (['I am terrified and shaking',          'Pure dread fills my chest',          'I am paralyzed by terror'],
                 ['I am brave and unafraid',             'My courage is unwavering',           'I face this without fear']),
    'love':     (['I love you with all my heart',        'My adoration knows no bounds',       'You are everything to me'],
                 ['I am indifferent to you',             'You mean nothing to me',             'I feel nothing for you']),
    'disgust':  (['That is revolting and vile',          'I am sickened beyond words',         'This is utterly repulsive'],
                 ['That is delightful and pleasant',     'I am pleased and charmed',           'This is wonderful']),
    'surprise': (['I am completely stunned',             'My jaw dropped in shock',            'I never expected this'],
                 ['That is exactly as expected',         'Nothing here surprises me',          'I predicted this entirely']),
    'anxiety':  (['I am consumed by worry',              'My mind races with fears',           'I cannot calm my racing thoughts'],
                 ['I am relaxed and untroubled',         'My mind is still and clear',         'I have no concerns at all']),
}


@torch.no_grad()
def get_resid(model, tok, prompts, layer, device):
    """Return [n_prompts, d_model] — mean residual over non-pad tokens at given layer."""
    cache = {}
    def hook(_, __, output):
        cache['h'] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = model.model.layers[layer].register_forward_hook(hook)
    out = []
    try:
        for p in prompts:
            enc = tok(p, return_tensors='pt').to(device)
            _ = model(**enc)
            h = cache['h'][0].float()  # [T, d_model]
            # mean over content tokens (skip BOS)
            if h.shape[0] > 1:
                h = h[1:]
            out.append(h.mean(0))
    finally:
        handle.remove()
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--emotion', default=None, help='Use built-in contrast pair for this emotion')
    ap.add_argument('--pos', default=None, help='Positive (target) prompt — overrides --emotion')
    ap.add_argument('--neg', default=None, help='Negative (anti-target) prompt')
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--layer', type=int, default=13)
    ap.add_argument('--max-new', type=int, default=120)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--no-norm', action='store_true', help='Skip unit-norm of steering vec')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    if args.pos and args.neg:
        pos_prompts, neg_prompts = [args.pos], [args.neg]
    elif args.emotion:
        if args.emotion not in CONTRASTS:
            raise SystemExit(f"unknown emotion '{args.emotion}'. Have: {list(CONTRASTS)}")
        pos_prompts, neg_prompts = CONTRASTS[args.emotion]
    else:
        raise SystemExit('need --emotion OR (--pos and --neg)')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[load] device={device} layer={args.layer}')
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device).eval()

    pos_resid = get_resid(model, tok, pos_prompts, args.layer, device).mean(0)  # [d_model]
    neg_resid = get_resid(model, tok, neg_prompts, args.layer, device).mean(0)
    direction = pos_resid - neg_resid
    raw_norm = direction.norm().item()
    print(f'[actadd] steering vec ||·||={raw_norm:.2f}  (n_pos={len(pos_prompts)}, n_neg={len(neg_prompts)})')

    if not args.no_norm:
        direction = direction / direction.norm()
    patch = (args.scale * direction).to(device=device, dtype=torch.float32)
    print(f'[actadd] patch norm: {patch.norm().item():.2f}  scale={args.scale}')

    call_count = {'n': 0}
    def steer_hook(_, __, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h_new = (h.float() + patch).to(h.dtype)
        if args.debug and call_count['n'] < 2:
            print(f'    hook#{call_count["n"]}: resid_norm={h.float().norm().item():.1f} shape={tuple(h.shape)}')
        call_count['n'] += 1
        return (h_new,) + output[1:] if is_tuple else h_new

    def gen(use_hook):
        msgs = [{'role': 'user', 'content': args.prompt}]
        inputs = tok.apply_chat_template(msgs, return_tensors='pt', return_dict=True, add_generation_prompt=True).to(device)
        in_len = inputs['input_ids'].shape[1]
        handle = model.model.layers[args.layer].register_forward_hook(steer_hook) if use_hook else None
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        finally:
            if handle is not None:
                handle.remove()
        return tok.decode(out[0, in_len:], skip_special_tokens=True)

    if args.baseline:
        print('\n=== BASELINE ===')
        print(gen(False))
    print('\n=== ACTADD STEERED ===')
    print(gen(True))


if __name__ == '__main__':
    main()
