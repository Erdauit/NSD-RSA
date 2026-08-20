# Visual tokens drift from early- to lateral-stream alignment as they traverse the language stack of vision-language models

> **Draft v0.1 — 2026-08-20.** Skeleton with real numbers from F1–F3. Slots marked
> `[F4]`/`[F5]` are filled by those stages. Working language: English (bioRxiv/arXiv).
> Numbers: primary readout `trim`, 515 shared NSD images, 8 subjects, noise-ceiling
> normalised, fp32.

**Authors:** Erdauit T. — *(affiliations TBD)*

---

## Abstract (draft)

Brain alignment of artificial vision is almost always measured at the output of a vision
encoder. In modern vision-language models (VLMs), however, visual tokens then travel
through 25–30 further layers of a language model, and what happens to their
representational geometry there has not been systematically measured against the brain.
We trace representational similarity to human visual cortex (Natural Scenes Dataset, 7T
fMRI, 8 subjects) across the entire stack of vision-language models — every vision-encoder
block, the projector, and every LLM layer — at the image-token positions. Alignment to
early visual cortex falls monotonically across the language stack in every subject
(0/8 subjects show a rise; p = 0.008) at all three scales of the SmolVLM family (256M,
500M, 2.2B), while alignment to the lateral visual stream rises (8/8 subjects at the two
larger scales). The ventral-stream trajectory reverses sign with model scale (slope
+0.21 → +0.04 → −0.39 ×100), cautioning against conclusions drawn from any single model.
The core pattern generalises across architecturally distinct families: in Qwen2-VL-2B
and LLaVA-OneVision-0.5B, early-cortex alignment likewise falls in 0/8 subjects and
lateral-stream alignment rises (8/8 and 7/8) — five models, three families in total.
Because lateral alignment *rises* while early alignment falls, the drift cannot be pure
degradation of visual information; the language stack selectively reshapes visual
geometry toward that of the lateral stream. [F5: encoding-model agreement sentence.]

---

## 1. Introduction

**¶1 — the gap.** How closely the internal representations of vision models match those
of human visual cortex is a central quantitative question of NeuroAI, and the standard
experimental unit is the vision encoder: activations are read from a convolutional or
transformer backbone and compared, layer by layer, to fMRI responses. Yet the systems now
deployed everywhere are vision-*language* models, and in a VLM the encoder is only the
first third of the computation: its output is projected into a language model and
processed by another 25–30 transformer layers. Those layers were trained on next-token
prediction, not on vision — but the visual tokens keep flowing through them, and their
geometry keeps changing. Where along this full stack does alignment to visual cortex
live, and what does the language model do to it?

**¶2 — what exists and what is missing.** Three literatures approach this question
without answering it. Caption-based studies compare the brain to embeddings of *image
descriptions* and find language-model representations aligning with high-level visual
cortex, particularly lateral occipitotemporal regions (Doerig et al. 2025;
Marcos-Manchón & Fuentemilla 2025; Conwell et al. 2023). Interpretability studies trace
visual tokens *inside* the LLM — finding partial preservation of visual information
(Liu et al. 2025) and convergence toward language feature space at mid-to-late depth
(Venhoff et al. 2025) — but never against the brain. Brain studies of multimodal models
compare *output* embeddings without unrolling the stack (Oota et al. 2025). What is
missing is the trajectory itself: the same visual tokens, tracked through every layer of
the language model, measured against the same cortical data. That trajectory is this
paper's object of study.

**¶3 — what we find.** Using the Natural Scenes Dataset (515 images shared with equal
repeat counts across all 8 subjects), we measure representational similarity between
every stage of a VLM stack and seven cortical regions, normalised to the data's noise
ceiling. Three results. (i) Alignment to early visual cortex falls monotonically across
the language stack — in zero of eight subjects does it rise, at any of three model
scales (p = 0.008 each), and the slope steepens with scale. (ii) Alignment to the
lateral stream rises concurrently (6/8, 8/8, 8/8 across scales); at the largest scale it
is the *only* region whose alignment rises. (iii) The ventral-stream trajectory flips
sign with scale (+0.21 → +0.04 → −0.39), i.e. the smallest model behaves qualitatively
unlike the largest — a warning for single-model claims about "brain-like" VLMs.
The pattern is family-general: in two further families (Qwen2-VL, LLaVA-OneVision) the
early-cortex fall again holds in 0/8 subjects and the lateral rise in 8/8 and 7/8, so
across all five models the only region whose alignment survives — and grows — through
the language stack is the lateral stream. A rise cannot be produced by pure loss of
visual information, so the language stack is not merely degrading vision; it reshapes
visual geometry, selectively, toward the lateral stream — the cortical territory
associated with actions, bodies, and social content.

---

## 2. Methods

### 2.1 Brain data

Natural Scenes Dataset (Allen et al. 2022): 7T whole-brain fMRI, 8 subjects, each viewing
up to 30,000 presentations of COCO images. We use the prepared betas (GLMsingle,
`betas_fithrf_GLMdenoise_RR`, version b3) in the `fsaverage` surface space, restricted to
the `streams` ROI atlas (early, midventral, ventral, midlateral, lateral, midparietal,
parietal).

**Stimulus set: 515, not 1000.** The "shared1000" images were seen three times by only
the four subjects who completed all 40 sessions. Exactly 515 images have ≥3 repeats in
all 8 subjects. Unequal repeat counts would make measurement noise differ per image, and
a noisier response pattern sits systematically further from everything else in an RDM —
a stimulus artefact indistinguishable from representational structure. We therefore use
the 515-image set (132,355 pairs); the same subset is used by Doerig et al. (2025).
Betas are z-scored per session, averaged over the 3 repeats, and restricted to vertices
valid in all subjects.

**Pipeline validation.** Split-half reliability computed from our own trial extraction
agrees with NSD's published `ncsnr` at r = 0.968 across 67,696 vertices; reliability is
higher in early visual cortex (+0.40) than anterior ventral regions (+0.21), as it must
be. [Figure S1]

### 2.2 Models

*Vision encoders (part A, validation and baselines):* DINOv2 ViT-B/14 and ViT-L/14
(self-supervised), ViT-B/16 (ImageNet supervised), ResNet-50 (supervised), CLIP ViT-B/16
(language-supervised).

*VLM scale ladder (part B):* SmolVLM-256M, SmolVLM-500M, SmolVLM-2.2B — one family, one
training recipe, three capacities. SigLIP vision tower → pixel-shuffle connector → LLM
(30/32/25 layers respectively).

*Other families [F4]:* Qwen2-VL-2B (native-resolution ViT, 32 blocks → merger →
Qwen2 LLM, 28 layers) and LLaVA-OneVision-0.5B (SigLIP, 26 blocks → MLP projector →
Qwen2 LLM, 24 layers).

### 2.3 Readouts along the stack

One continuous depth axis per model: every vision-encoder block, the projector, and at
every LLM layer the mean over *image-token positions only* (`imgmean`), plus the final
position (`lasttoken`). Pooling over all positions would average prompt words into the
"visual" representation and manufacture task effects; the image-token mask is derived
from the model's declared image-token id, and extraction refuses to run if none is found.

**Token pooling (trim).** ViT-family towers concentrate up to 15% (SigLIP) and 31.5%
(DINOv2-B) of total token norm into 1% of tokens at mid-depth, and the outlier share
tracks RSA almost deterministically (r = −0.78 ventral, +0.88 lateral). No NeuroAI
standard exists for handling these tokens. Our primary readout is therefore the mean
over tokens after dropping the top 1% by L2 norm (`trim`); all headline numbers were
verified robust to the choice among mean / trim / median. The apparent *negative*
alignment at encoder mid-depth under plain mean pooling is partly an outlier artefact
and is not interpreted.

**Precision (fp32).** bfloat16 — the reflexive choice for inference — moves RDMs by
0.02–0.05 Spearman against a bitwise-reproducible fp32 reference, the same order as the
between-model effects we report. This reproduces on both Apple MPS and CUDA tensor-core
backends (worst-layer RDM agreement with fp32: bf16 0.962, fp16 0.931 on an RTX 5070
Ti). All activations are extracted in float32.

**No tiling.** Image-splitting (SmolVLM tiling, LLaVA anyres) is disabled: the model
sees the whole visual field at once, as cortex does, and sequence lengths stay
comparable across families.

**Task conditions and prompt position.** Four conditions: no prompt (control), and
questions about objects, spatial layout, mood. Attention is causal, so image tokens can
only attend to what *precedes* them: with the conventional (image, question) order the
image-token readouts are bitwise identical across prompts — the modulation null would be
architectural, not empirical. The question is therefore placed *before* the image,
matching the human paradigm of instructing before stimulus. This property is re-verified
per model family (max |difference| across conditions must be exactly 0 with the question
after the image, and non-zero with it before).

### 2.4 RSA

RDMs: 1 − Pearson over features (fp64 accumulation). Model–brain comparison: Spearman
over the RDM upper triangle. Significance: permutation test over stimuli (5,000
permutations; floor p = 0.0002). Noise ceiling: split-half based ceiling per ROI
(Lage-Castellanos et al. 2018 conventions); all reported alignments are fractions of
this ceiling. Layer-trajectory statistics: per-subject linear slope of normalised RSA
across LLM layers, sign counts across the 8 subjects, two-sided Wilcoxon signed-rank
test on the slopes.

Low-level controls: RDMs from downsampled pixels, mean luminance, and RMS contrast.
Mean luminance alone reaches 0.218 of ceiling in early visual cortex — half of the best
model — so early-cortex alignment numbers are interpreted against this baseline, not
against zero.

---

## 3. Results

### 3.1 Vision encoders reproduce the known hierarchy (validation)

All five encoders show the layer-to-region hierarchy (early layers ↔ early cortex, late
layers ↔ ventral cortex), all 35 model×ROI combinations significant. Peak
noise-ceiling-normalised alignment: CLIP 0.425 (early), DINOv2-L 0.403, best ventral
0.293 (CLIP). Self-supervised DINOv2 does *not* exceed supervised ViT in ventral cortex —
consistent with training-diet-over-architecture findings (Conwell et al. 2024). The one
large localised effect: ResNet-50 collapses in ventral cortex (0.125 vs 0.24–0.29 for
all transformers) while matching them in early cortex. [Figure 1: encoder heatmaps]

### 3.2 Across the language stack, early-cortex alignment falls in every subject; lateral rises

The core figure tracks alignment along vision blocks → projector → LLM layers
(image-token mean). Slopes across LLM layers (×100, `trim` readout, per-subject signs,
Wilcoxon p):

| Region | SmolVLM-256M | SmolVLM-500M | SmolVLM-2.2B |
|---|---|---|---|
| early | −0.22 · 0/8 · p=.008 | −0.13 · 0/8 · p=.008 | −0.43 · 0/8 · p=.008 |
| midventral | −0.11 · 2/8 | −0.06 · 2/8 | −0.53 · 0/8 · p=.008 |
| ventral | +0.21 · 7/8 · p=.023 | +0.04 · 3/8 · p=.74 | −0.39 · 0/8 · p=.008 |
| lateral | +0.22 · 6/8 · p=.039 | +0.53 · 8/8 · p=.008 | +0.61 · 8/8 · p=.008 |

Early-cortex alignment falls monotonically through the language stack in 0/8 subjects ×
3 scales; the slope doubles at 2.2B. Lateral alignment rises, strengthening with scale;
at 2.2B it is the only region that rises. [Figure 2: stack profiles, three scales]

### 3.3 The ventral trajectory reverses with scale

The ventral slope goes +0.21 (7/8 subjects) → +0.04 (3/8) → −0.39 (0/8) across the
capacity ladder. A "the LLM continues the ventral hierarchy" story supported by the
256M model is contradicted by the 2.2B model. The reversal is readout-independent at
2.2B (0/8 under mean and trim, 1/8 under median). Conclusions about brain alignment of
"VLMs" drawn from a single model — particularly a small one — do not generalise even
within one family.

Absolute levels rise with scale: peak ventral alignment 0.306 at the LLM entry (llm.00)
for 2.2B, exceeding the best pure vision encoder in part A (CLIP, 0.293); peak early
alignment 0.232, marginally above the luminance baseline (0.218) for the first time.

### 3.4 Task modulation is real, replicable, and an order of magnitude too small to matter

With the question placed before the image, the task measurably shifts LLM-stage
alignment (permutation p ≤ 0.027 in all regions at 2.2B; negative control at the vision
tower: spread exactly 0.0000). But the effect is 1–3% of the alignment value itself
(spreads 0.0007–0.006 against values 0.03–0.35), does not grow with scale (2.2B spreads
are *smaller* than 500M's), and is absent in lateral cortex at the smaller scales. At
inference time, instructions barely reshape visual geometry in models of this class — a
controlled negative result given the design.

### 3.5 The effect generalises across model families

Same protocol on two architecturally distinct families — Qwen2-VL-2B (native-resolution
ViT trained end-to-end, merger projector, Qwen2 LLM) and LLaVA-OneVision-0.5B (frozen-
recipe SigLIP, MLP projector, Qwen2 LLM). Slopes across LLM layers (×100, `trim`):

| Region | Qwen2-VL-2B | LLaVA-OneVision-0.5B |
|---|---|---|
| early | −0.90 · 0/8 · p=.008 | −0.30 · 0/8 · p=.008 |
| midventral | −0.88 · 0/8 · p=.008 | −0.34 · 0/8 · p=.008 |
| ventral | −0.54 · 0/8 · p=.008 | −0.28 · 0/8 · p=.008 |
| lateral | +0.47 · 8/8 · p=.008 | +0.16 · 7/8 · p=.039 |

Early-cortex alignment falls through the language stack in 0/8 subjects in every model
of every family tested — five models, three families. Lateral-stream alignment rises in
both new families; in Qwen2-VL its peak (0.447 of ceiling) sits at the *last* LLM layer.
Robust to all three token poolings in both models. [Figure 4: f4_family_slopes]

Across families, the ventral pattern sharpens the within-family scale trend of §3.3:
in both new families ventral alignment falls (0/8) — including LLaVA-OneVision at 0.5B,
a scale at which SmolVLM still shows a ventral rise. The positive ventral slope is a
peculiarity of the smallest SmolVLM rungs, not a property of small VLMs in general. The
family-general pattern is: through the language stack, alignment falls in every measured
region *except* the lateral stream.

Absolute levels: Qwen2-VL peaks at 0.317 (early, vision tower) and 0.331 (ventral) of
ceiling — well above the luminance baseline (0.218), removing the caveat that applied
to the smallest models.

### 3.6 [F5] Encoding-model check

*Slot. Ridge from best encoder layer and best LLM layer into vertices, image-wise CV,
R²/noise ceiling by region; one table: do encoding conclusions agree with RSA?*

---

## 4. Discussion

**Summary.** *(one paragraph, write after F4)*

**Retargeting, not degradation.** Liu et al. (2025) show LLMs preserve less visual
information than the encoder provides — inviting a deflationary reading of our early-
cortex decline as simple information loss. Pure loss, however, cannot produce a *rise*:
lateral alignment increases through the same layers in 8/8 subjects. [F5/B8: linear
decodability of low- vs high-level image properties per LLM layer.]

**Not just object nouns?** Conwell et al. (2023) find caption-based language-model
alignment with high-level visual cortex is largely carried by object content. We built a
crop-aware COCO object-category RDM (an object counts only if its bounding-box centre
falls inside the NSD crop; 80 categories, Jaccard distance, plus an area-weighted
variant). The category RDM is itself selectively lateral: it reaches 0.365 of ceiling in
lateral cortex against ≈0 everywhere else — the object inventory genuinely lives in the
lateral stream's geometry, which makes this the right control to run. Partialling it out
of the model–brain relationship leaves the early-cortex fall untouched in every model
(0/8, p = .008), and the lateral rise survives in the three models where it was strong
(SmolVLM-500M +0.48, SmolVLM-2.2B +0.69 — *larger* than the plain +0.61, the category
component was partly masking the effect — and Qwen2-VL +0.44; all 8/8, p = .008). In the
two weakest models (SmolVLM-256M, LLaVA-OneVision-0.5B) the residual rise keeps its
direction but loses significance (p = .15, .11). The lateral gain of the stronger models
is therefore not reducible to a coarse object inventory; a caption-embedding partial —
the richer version of the objection — is left for future work.

**Degradation or retargeting, tested directly.** Linear decoding from every LLM layer's
image-token readout: COCO categories in the crop decode nearly perfectly from *every*
layer, including the last (mean AUC 0.96 → 0.97 across depth in both 2B models), while
low-level structure decays monotonically (luminance R² 0.37 → 0.26, Gabor-energy PCs
0.31 → 0.19 in Qwen2-VL). Pure information loss predicts joint decline; what we observe
is selective retention of object content with progressive shedding of low-level
structure — the decoding-level signature of retargeting, independent of the RSA result
it corroborates.

**Relation to language-space convergence.** Venhoff et al. (2025) locate visual-to-
language feature convergence at mid-to-late LLM depth. [B10 slot: compare our shift's
depth profile with theirs; if they coincide, discuss the non-neural reading explicitly.]

**Why lateral?** The lateral visual pathway has been argued to constitute a third
processing stream, distinct from the classic ventral "what" and dorsal "where" routes,
specialised for the perception of other agents: faces and bodies in motion, actions, and
social interactions (Pitcher & Ungerleider 2020). Within it, responses are organised
hierarchically — from low-level motion-sensitive territory (MT/EVC border) through
body- and object-selective areas (EBA/LOC) to superior temporal regions coding social
interaction as such (McMahon, Bonner & Isik 2023; McMahon & Isik 2024). Lateral
occipitotemporal cortex represents action categories generalising across the agent
performing them (Walbrin & Koldewyn 2019; Landsiedel et al. 2022), and recent work finds
its scene responses well described by verb-like, relational descriptors rather than
object inventories (Küçük et al. 2024). If language is, functionally, a compressed
description of *what matters about a scene to agents* — who is doing what to whom — then
a next-token objective should reshape visual tokens toward exactly these relational,
agent-centred distinctions, more than toward the fine-grained object identity of the
ventral stream or the local contrast structure of V1–V3. That is what we observe: the
language stack sheds early- and ventral-stream geometry while gaining lateral-stream
geometry, in every family tested. The reading is interpretive and correlational — we
measure geometric correspondence, not mechanism — but it makes a testable prediction:
the gain should be carried disproportionately by images containing agents and
interactions, and should survive partialling out object-category structure (see the
planned Conwell-style control). Notably, lateral cortex is also where the strongest
absolute alignment in this study lives (0.478 of ceiling at encoder mid-depth for
SmolVLM-2.2B; 0.447 at the *last* LLM layer of Qwen2-VL), an observation without a
satisfying explanation yet.

**Methodological contributions worth stating plainly.** (i) Reduced precision distorts
RDMs by the size of typical between-model effects, on two hardware backends; fp32 is a
validity condition, not pedantry. (ii) With causal attention, prompt position decides
whether task effects are architecturally possible at all; several published "no task
effect" nulls may be layout artefacts. (iii) High-norm outlier tokens carry alignment
effects nearly deterministically; the field has no standard, and we supply a measured,
defensible readout (`trim`).

**Limitations.** One dataset; 515 images; 8 subjects (though NSD's per-subject data
depth partially compensates); correlational geometry comparisons only — no causal claims
about the brain "working like" a VLM; families covered are SmolVLM (three scales),
Qwen2-VL-2B and LLaVA-OneVision-0.5B — no 7B+ models, excluded by the fp32 validity
requirement on 16 GB hardware; absolute alignment levels in early
cortex sit near a trivial luminance baseline at small scales, and analyses there are
interpreted as trajectories, not levels.

---

## Figures (planned)

1. **F1.** Part A validation: encoder layer × ROI heatmaps (5 models) + reliability/ncsnr scatter. *(exists: s3 heatmaps, s1 figures)*
2. **F2.** VLM stack profile: alignment along vision→projector→LLM, 7 ROIs × 3 scales. *(exists: vlm_stack_profile.png; restyle)*
3. **F3.** The slope table as a figure: early/ventral/lateral slopes vs scale, dots = subjects. *(to make; extends to F4 families)*
4. **F4.** [F4] Cross-family summary: early/lateral slopes, all families and scales. *(the preprint's headline figure)*
5. **S-figures.** Token-norm outlier profiles; readout robustness; task modulation; precision check; low-level controls.

## TODO

- [ ] F4 numbers + §3.5 + headline figure
- [ ] B8 decodability control (with F5)
- [ ] B9 object-category partial RSA
- [ ] B10 depth comparison vs Venhoff
- [ ] F5 encoding model + §3.6
- [ ] Verify every citation against the original paper (not abstracts) before submission
- [ ] LaTeX conversion (arXiv), bib file
