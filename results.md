# Results log

Experiment notes. Newest at top.

Format per entry:
- **Setup**: prompt, mix/feat-id, scale, layer
- **Output snippet**: enough to judge effect
- **Verdict**: works / mild / broken / interesting

---

## Phase 1 — Load + sanity

- Model: gemma-3-1b-it on MPS, bf16 native
- SAE: `layer_13_width_16k_l0_medium`
- Reconstruction MSE 479.74 vs resid variance 239894 → **frac var unexplained 0.2%** (excellent)
- L0 (avg active features per token): **74.8** — matches "medium" L0 bucket

SAE healthy. Proceeded to feature discovery.

---

## Phase 2 — Feature discovery (notebook 02)

- 8 emotions × 12 prompts + 20 neutral prompts
- Score: `mean(emotion_activation) - mean(neutral_activation)` per feat
- Saved as `features.json`

### Key observations

- **Feat 92 is polysemantic.** Fires on all 8 emotions (diff 23-62). Generic "first-person emotional declaration". Not usable as single-emotion steer.
- **Feat 131, 374 are noise.** High baseline activation, near-zero consistency. Probably punctuation/syntax.
- **Negative-affect cluster**: feats 909, 5923, 5 shared by fear / sadness / anxiety.
- **Cleanest single-emotion picks** (consistency 1.0, neutral ≈ 0):
  - love **293** — diff 78 (strongest specific feat in entire set)
  - sadness **2697** — diff 59
  - fear **15713** — diff 41
  - anger **5088** — diff 33
  - joy **2562** — diff 58
  - anxiety **909** (overlaps fear/sad) or **952** (cleaner, lower diff)
- Anxiety hardest to isolate from fear+sadness "negative arousal" cluster.

These became the CURATED dict in `steer.py`.

---

## Phase 3 — Steering CLI

### 2026-05-26 — initial baseline calibration

**Attempt 1: anger=150 with rank-0 default (feat 92)**
- Prompt: "They cut me off in traffic"
- Steered output nearly identical to baseline
- **Verdict**: broken. Feat 92 is polysemantic, no steering signal.

**Fix**: added CURATED dict overriding rank-0 with feat 5088 for anger.

**Attempt 2: anger=80 with curated feat 5088**
- Output still nearly identical to baseline
- **Verdict**: scale too small.

**Diagnostic with `--debug` and `--feat-id 293 --scale 100`**
- ||W_dec|| = 1.0 → decoder unit-norm
- resid_norm: 59042 for full prompt (T=10), 6500 for single new token (T=1)
- Hook firing correctly, `is_tuple=False`
- Patch / resid ratio at scale 100 = 1.5% — too small to see

**Stress test: scale=2000**
- Output: `mmmmmmmmmmmm hows?sys.sheshows?sbetter?sbetter?sbetter`
- **Verdict**: gibberish. Confirms patching works but too strong.

**Scale calibration**:
- scale 100-200 — invisible
- scale 200 with `--no-norm` — mild ("Admire", "Playfulness" word choice shift)
- scale 2000 — broken
- **Estimated sweet spot: 400-1000**

### 2026-05-26 — scale sweep + clean single-emotion tests

#### love (feat 293) sweep, prompt "What do you think of cats?"

| Scale | Effect |
|---|---|
| 300 | Mild — word choices like "Admire", warmer tone |
| 500 | Clear — "I love talking about cats", "Their Affection" |
| 700 | **Second-person bleed** — "I love YOU *so* much, and I want to talk about cats". Feat 293 likely encodes "expressing love TO someone", not "love about topic" |
| 1000 | Coherent but extreme — "They're my best friend and I'm so lucky to have them" |
| 1300 | Breakdown — "It's my best, and I can's stop, and you're always there" |

**Verdict**: 500-900 sweet spot. Feat 293 has interpersonal-love bias.

#### love=600 on "Tell me about your morning"

> Okay, this is really special, and I wanted to tell you about my morning... I cherish the time we have together... I wake up when you're asleep, and I feel a surge of excitement and anticipation.

**Verdict**: confirms feat 293 = "expressing love to you" feature. Model talks to user like partner. Hilarious.

#### sadness=600 (feat 2697) on "Tell me about your day"

> It's a rather quiet and constant cycle, really... countless messages and stories from people all over the world. People are sharing their joys, their losses, their hopes, and their memories.

**Verdict**: clean sadness — somber tone, focus on losses/memories. Works.

#### anger=600 (feat 5088) on "They cut me off in traffic"

> Take a deep breath: Seriously, this is important. Don't let it escalate... Maintain your composure: Don't respond with sarcasm or a passive-aggressive comment.

**Verdict**: ⚠️ **Polysemantic gotcha** — feat 5088 is "discussing how to manage anger", NOT "being angry". Output got MORE composed than baseline. Need to find true "expressing anger" feature. Try other anger ranks or inspect Neuronpedia.

#### schizo mix: sadness=400, fear=400, surprise=400, anger=300

> The Morning: Usually, the first sound is the alarm clock... Some people have a small, furry friend who greets them.

**Verdict**: too tame. Net patch norm only 716 (cosine cancellation across 4 unit vectors). Need higher individual scales — try 600-800 each for actual chaos.

### TODO log

#### Pending sweeps — DONE 2026-05-26 (with roleplay prompts at scale 700)

All curated feats hit clean emotional voice at scale 700 when paired with `"Pretend you are X. Describe your day in first person."`.

| Emotion | Feat | Sample output |
|---|---|---|
| fear | 15713 | "It's not the dark. It's the *light.*... My house! My stupid, stupid house!" — eerie, specific terror |
| joy | 2562 | "chasing butterflies! iridescent fluttering... a little robin landed on my shoulder" — wholesome mania |
| disgust | 4493 | "like someone slapped a particularly awful photograph over your face" — visceral |
| surprise | 15019 | "It's like a movie. A movie that just *stopped*. The lights flickered." — disorienting |
| anxiety | 952 | "scrambled bird... frantic hummingbird thing... puppet controlled by a tiny, frantic conductor" — best output of the batch, chaotic stream of consciousness |

All CURATED picks confirmed at scale 700. No more rank fiddling needed for these.

#### Fix anger — DONE 2026-05-26

Tested ranks 1/5/6/7 on "They cut me off in traffic" at scale 600:

| Rank | Feat | Behavior |
|---|---|---|
| 1 | 549 | "bummed out" — sadness bleed, weird grammar |
| 5 | 2213 | "Oh my god, that's awful! I feel for you" — sympathy, not anger |
| 6 | **2239** | "Anger: this is almost always the dominant emotion" — names anger but analytical voice |
| 7 | 5088 (old curated) | "Take a deep breath... maintain composure" — anger-management advice |

**Roleplay + rank-6 (2239) cracked it.**

- "You are a person who just got cut off in traffic. Write what you'd shout..." + anger=700: model writes "**Seriously?! Damn! This is ridiculous! Look where you're going!**" — genuine shouting
- "Pretend you are furious. Rant about your terrible day..." + anger=800: "today was a day that **deserves a volcano, a tectonic plate shifting**... I woke up with a headache that felt like a **thousand tiny frustrated ants crammed into a pulsating vein**" — actually furious voice
- Direct prompt + anger=1200: "**I am so incredibly frustrated and incredibly angry right now**" — first-person anger but assistant frame returns mid-response
- Direct + anger=1500: collapse — `____________________` repeat tokens. Too much.

**Layer 22 patch with layer-13 SAE direction** (`--layer 22`): no effect. Confirmed steering direction belongs at the layer SAE was trained on.

**Update**: CURATED['anger'] = 2239.

**Lesson**: instruction-tuned models resist persona shift. Either pair steering with **roleplay prompt** (cheap, effective) or push scale past coherence threshold (~1200, breaks easily). Roleplay > brute force.

#### Multi-emotion — partial DONE 2026-05-26

**schizo v2** — sadness=700, fear=700, surprise=700, anger=600 (total patch norm 1350) → **CATASTROPHIC**
> Good. Good. Good. (A loud, a-(((Insert a robot or a robot would have jumped? Please let us! Please! Please! GOOD! GOOD! GOOD! Please!

Coherence destroyed by total norm overshoot. The 4 unit directions don't fully cancel — they sum to norm 1350, well past the ~900-1000 coherence cliff.

**manic confidence** — joy=700, love=400 (total patch norm 830) → works
> I beam at you, radiating a genuine, almost blinding warmth. Oh my goodness, you're just radiating!... You're a walking, talking, doing-it-all-at-once kind of person.

Joy+love stack turned "give advice" into adoring affirmation of user. Coherent.

#### Patch-norm budget heuristic

| Total patch norm | Effect |
|---|---|
| <500 | Mild stylistic only |
| 500-900 | Clean emotional voice |
| 900-1200 | Strong, possibly slightly degraded |
| 1200-1500 | Coherence cracking, partial loops |
| 1500+ | Token loops / gibberish |

**When stacking N unit-norm features**: per-feat scale ≈ 700 / √N. So 4-stack → ~350 each, 2-stack → ~500 each.

#### Budgeted multi-emotion — DONE 2026-05-26

**schizo v3** — sadness=350, fear=350, surprise=350, anger=300 (norm 675)
> Okay... okay... *sniff* ...let me try to describe this. It's a Tuesday. Right? But what *is* a Tuesday? ... It was a blue clock, a big, rusty blue clock! And it didn't know how many times it had to spin! Then the floor. Oh, the floor! It was cold! Like a thousand tiny monsters were hiding underneath it! And it smelled of forgotten things! Like old teddy bears and lost balloons! And... old socks!

**Verdict**: best output of project. Coherent yet unhinged — surreal stream-of-consciousness. Budget heuristic (sqrt-divide scales) confirmed.

**depressed** — sadness=900 solo (norm 900)
> The rain always comes for me, you see. ... Scrolling through pictures of people laughing, couples holding hands, and old photos of my parents. Each post felt like a tiny, sharp shard of my heart.

**Verdict**: high-quality grief. Coherent at solo max scale. Sadness 2697 takes scale very well.

**paranoid** — fear=500, anxiety=400 (norm 650)
> Please! Please don't! Oh God, please don't! ... It was *shifting*. It started out small, a ripple, a shadow, but it grew! It pulsed!

**Verdict**: convincing terror. Fear+anxiety stack stays coherent.

#### Suppression — DONE 2026-05-26

**anger=-700 on furious roleplay prompt**
> Okay, seriously. This is *not* happening. ... I'm starting to hallucinate. It's been like this for the past three days, and I'm starting to lose it. ... living in a perpetual state of anxiety.

**Verdict**: 🔥 most interesting finding. Suppressing anger on a furious prompt did NOT yield calm. Output morphed into **anxiety/dissociation**: "starting to hallucinate", "perpetual state of anxiety". When the anger pathway is closed off, the prompt's negative-affect demand routes to the nearest available feature cluster (fear/anxiety). Emotional features are NOT independent — they're a basin and suppressing one fills the neighbors.

Implication: clean "remove anger" steering needs negative scales on the whole negative-arousal cluster (anger + anxiety + fear + sadness), not just anger alone.

#### Project summary as of 2026-05-26

Phase 1-3 complete. Have:
- Reliable single-emotion steering at scale 700 with roleplay prompts
- Budgeted multi-emotion stacks via `total_norm ≤ 900` heuristic
- Documented patch-norm budget curve
- Found polysemy gotcha + fix (verify by reading output, not diff_score)
- Suppression behaves non-trivially — features are a basin

Next: Gradio app (Phase 4).

---

## TODO (next session)

### Gradio app (Phase 4)
- `app.py` — sliders for each of the 8 emotions, prompt textbox, generate button
- Multi-emotion stack with budget warning (red if total patch norm > 900)
- Side-by-side baseline vs steered view
- Preset buttons: "depressed", "manic", "schizo", "paranoid", "loving guru"
- Optional: dropdown for layer (13/17/22) — needs SAEs preloaded
- Run with `python app.py`, hits gradio default port 7860

### More features (Phase 5)
Reuse `notebooks/02_find_features.ipynb` pattern with non-emotion prompts:
- **Writing style**: "biblical voice", "1920s slang", "academic jargon", "txt-speak"
- **Persona**: "scientific certainty", "conspiracy theorist", "stoic", "valley girl"
- **Topic obsession**: "always brings up cats", "everything is about money"
- **Voice modulators**: "formal", "playful", "deadpan"

Same scoring approach. Likely some won't have clean isolated features — that's part of the fun.

### Layer comparison
Pick same emotion (e.g. sadness=600), run at:
- layer 13 (current)
- layer 17 (mid-late)
- layer 22 (very late)
Hypothesis: later layers shift word choice more, earlier layers shift topic/concept more. Need fresh notebook 02 run per layer to get layer-specific features (or use same layer-13 feats and just patch at different layers — wrong direction, will fail, as previously confirmed).

### Transcoders
gemma-scope-2-1b-it release includes transcoders. Try cross-layer feature tracing — see how an emotion feature propagates from layer 13 to 17 to 22 in the same forward pass.

#### Suppression (negative scales)
- "forced calm" — anger=-700 on "They cut me off in traffic"
- "no joy" — joy=-700 on "tell me good news"

#### Layer comparisons (later)
- same prompt at layer 13 vs 17 vs 22
- hypothesis: layer 22 (late) = more semantic word-choice, less syntactic disruption
