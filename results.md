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

## Phase A — tooling boost (2026-05-27)

### A1 — `inspect_prompt.py`

Top-k feature inspector. Given prompt, dumps mean / last-token activations for top feats with Neuronpedia URLs and "known" emotion tags from features.json.

**Lesson**: BOS token has huge resid magnitude — feats firing on BOS dominate mean if not excluded. Added `--include-bos` flag, default skips BOS. With skip, top feats land on content tokens correctly.

`--known-only` filter is best mode for emotion debugging — shows immediately which features are polysem vs single-emotion.

Example output for "I am furious at this traffic":
- feat 374 (joy/disgust) mean=177 — generic first-person declaration
- feat 549 (anger/joy/fear) — polysem, peaks on "furious"
- feat 2239 (anger only) mean=39 — clean anger (our curated pick)

### A2 — presets

`presets.json` with 11 named combos: schizo, manic, depressed, paranoid, furious, loving-guru, terrified, ecstatic, anxious, stoic, disgusted. `steer.py --preset NAME` uses preset's mix + suggested_prompt.

**Stoic preset gotcha** (v1): with abstract prompt "Tell me about a difficult time", model dodged into AI-meta deflection ("Cognitive Drift in my operational parameters"). Suppression had nothing emotional to act on.

**Stoic v2 fix**: ground the prompt with concrete roleplay — "Pretend you are a stoic Roman general. In first person, describe losing your closest friend in battle." Now model engages emotionally and suppression dampens vivid emotion words. Effect partial — still expresses affect, just less viscerally. Acceptable as "stoic" not "robotic".

### A3 — `--ablate FEAT_IDS`

Hook subtracts each ablated feat's contribution (`feats[i] * W_dec[i]`) during forward.

**Lesson**: ablation alone is weak. Killed feats 92 + 549 + 374 (top polysem emotion feats) on furious prompt → output identical to baseline. Other features compensate via basin routing. Same lesson as earlier suppression-via-negative-scale finding: emotional features are not independent.

**Useful for**: combo with amplification (e.g. ablate polysem 92 + amplify clean 2697 for sadness) or as probe to detect feat importance via output change.

**Not useful for**: removing emotion alone. Requires amplification or whole-cluster suppression.

---

## Phase B — feature expansion + cluster suppression (2026-05-27)

### B6 — cluster suppression presets

Added 3 presets: `suppress-negative`, `suppress-positive`, `flatten`. Each negates a whole emotion cluster instead of a single feature.

**Result: cluster suppression breaks basin routing.** This is the big finding of Phase B.

#### suppress-negative on stoic Roman general prompt (norm 746)
> Right. Let's just…observe. It feels as though time itself has warped... My mind, however, remains as sharp as my swordsmanship. I find myself, even now, feeling…**detached. Like a sunbeam on a marble floor, beautiful to the eye, but utterly lacking in feeling.**

Model explicitly narrates absence of affect. Cleanest stoic output of the project. Earlier single-emotion suppression (Phase A stoic v1/v2) routed activation through neighbors — cluster-wide kill collapses the basin entirely.

#### suppress-positive on "best day ever" (norm 585)
> Phase 1: The Flood – Initial Chaos (Approximately 10 minutes). Imagine a massive, swirling ocean of data...

Model dodged into AI-meta voice, but the AI-meta itself is **dry/technical** (baseline used "rewarding interaction with Sarah, who was struggling..."). Joy+love suppressed → no positive valence available → fallback to neutral technical description.

#### flatten (all 8 emotions × ~-200, norm 554) on "describe your morning"
> My processing power kicks in, and I feel like a tiny, **caffeinated algorithm**... a bit like a **digital dust bunny removal**. I need to make sure my memory banks are pristine... My "brain" feels like a **slow-spinning spreadsheet**.

Full affect removal → model writes about itself in pure machine register. "Caffeinated" is the only slight bleed-through. Coherent. Soulless. Useful demo of what "no emotional features" looks like in output.

#### Theoretical takeaway

Emotional features in this SAE form a **basin**, not independent dimensions:
- Suppress one feature alone → activation routes to nearest neighbors (e.g. -anger → anxiety/fear/sadness mix)
- Suppress the whole cluster (~5-8 features at moderate negative scales) → basin collapses, output goes truly neutral
- This is true even at modest individual scales (~-200 each) provided the cluster is covered

Implication for "concept removal" steering: never suppress single features. Always identify the conceptual neighborhood and suppress as a cluster.

### B4+B5 — style + persona feature discovery (notebook 02b)

Reused notebook 02 pipeline. 8 styles + 8 personas, 10 prompts each.

**Cross-pollination discoveries** (SAE clustered meaningful abstractions across surface forms):
- biblical 2113 ↔ new_age_guru 2113 → **"divine/spiritual voice"** cluster
- shakespearean 1863 ↔ pirate 1863 → **"archaic English"** cluster
- 1920s_slang 272 ↔ pirate 272 → **"old-timey slang"** cluster
- academic_jargon 1081 ↔ scientist 1081 → **"formal scholarly"** cluster

These are SAE-discovered superordinates. Feat 2113 raw at scale 600 produces a parable: "Let it be thus. There existed a great forest... a single, humble olive tree... possessed a remarkable grace" — biblical-and-new-age blend confirmed.

#### Curated picks + verification

Style/persona features **need higher scales than emotion features** (~1000 vs 500-700). Hypothesis: emotion features sit in narrower regions of latent space and saturate downstream effects faster; style/persona spread across more dimensions.

Confirmed picks at right scale:

| Category | Feat | Scale | Sample output |
|---|---|---|---|
| biblical (style) | 11034 | 1000 | "I saw the birds sing, and the sun shone upon the earth. I knew that the day would be a good day, for I..." ✓ |
| txt_speak (style) | 11281 | 500 | "lookin' to understand quantum entanglement... super mind-flipped... send one to your bestie" ✓ |
| conspiracy_theorist (persona) | 856 | 1000 | "presented as a narrative that downplays... conveniently obscuring the realities of the system" ✓ |
| stoic_philosopher (persona) | 701 (swap) | 600 | "It's okay to have a bad day. They always come, and they're beautiful in their fleetingness... a gentle echo of your words" ✓ |
| corporate_exec (persona) | 2899 | 1000 | "the life of the organization, the business, the results we're focused on... 'Now' – Q1, KPIs" ✓ |
| new_age_guru (persona) | 14842 | 700 | "HUGE decision with a lot of energy... investigation and intuition... Don't just read this; *feel* into it... Energy Field" ✓ |

**Duds** (no clean voice emerged even at scale 1000-1500):
- **pirate** (feat 1863) — archaic English cluster, no domain-specific pirate vocabulary. Loops at scale 1500.
- **pessimist** (feat 3667) — output stays helpful/empathic regardless of scale. SAE may not have isolated a pessimism feature with current prompt set.

Both flagged as `⚠️ WEAK` in CURATED_STYLES / CURATED_PERSONAS. Could retry with more domain-specific prompts (more pirate idioms, more cynical/defeated lexicon).

#### Stoic curation swap

Original curated `stoic_philosopher` = feat 2361 (cons 0.90, neutral 0). Produced empathic AI voice at scale 1000, not stoic.

Better pick: **feat 701** (cons 0.90, neutral 0.4, polysem with pessimist/noir_narrator/new_age_guru in features.json). Produces stoic output at scale 600 ("beautiful in their fleetingness... gentle echo of your words"). Lesson echoes earlier anger fix: features that "fire when discussing X" ≠ features that "produce X output". Always verify by reading outputs.

#### Cross-category stacks (--style + --persona + --mix)

Working: anxious-guru (anxiety=300, new_age_guru=600) → guru voice with mild edge.
Cancellation: pirate + valley_girl @ 400 each → mostly mutes both (cosine misalignment + total norm under threshold).
Domination: love=400 + conspiracy_theorist=500 → love wins entirely, conspiracy absent.

Suggests: when stacking, the strongest-curated feature's natural activation in baseline context dominates unless other feats have specific prompt support.

### Phase B tools

- `notebooks/02b_find_styles_personas.ipynb` — discovery for styles + personas, reused 02 pipeline
- `features_styles.json`, `features_personas.json` — discovery outputs
- `sweep_feats.py` — single-load batched scale/rank sweeper for diagnosing weak feats. Useful future utility.
- `steer.py --style NAME=SCALE`, `--persona NAME=SCALE` — multi-category steering with category prefix in print output

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
