# NSD-RSA

**Representational alignment between self-supervised vision transformers and the human ventral visual stream.**

How well do the representations of modern self-supervised vision transformers (DINOv2 and
friends) match those of the human ventral visual stream — and how is that match distributed
across model layers and cortical regions?

- **Brain data:** [Natural Scenes Dataset](https://naturalscenesdataset.org) (NSD) — 7T fMRI,
  8 subjects, natural images from COCO. We use the **shared1000** subset (the 1000 images every
  subject saw).
- **Primary method:** representational similarity analysis (RSA), noise-ceiling normalised.
- **Convergent method:** voxelwise encoding models (ridge regression).

**Начать отсюда:** [`docs/CONTEXT.md`](docs/CONTEXT.md) — что это за проект и что уже
сделано, за пять минут. Запуск на машине с видеокартой —
[`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md). План текущей фазы — [`PHASE2.md`](PHASE2.md).

> Status: **core effect replicated across three VLM families.**
> All 8 subjects, 515 stimuli, 7 cortical regions, 5 vision encoders + 5 VLMs
> (SmolVLM 256M/500M/2.2B, Qwen2-VL-2B, LLaVA-OneVision-0.5B).
>
> *Pipeline validation:* our per-vertex split-half reliability agrees with NSD's own `ncsnr`
> at **r = 0.968** across 67,696 vertices; the layer-to-region hierarchy replicates in all
> 5 encoders; reliability is higher in early visual cortex (+0.40) than anterior ventral
> (+0.21), as it must be.
>
> *Core result:* passing through a VLM's language stack, alignment to early visual
> cortex falls monotonically — **0 of 8 subjects show a rise, in all five models of all
> three families** (p = 0.008 each) — while alignment to the lateral stream rises
> (6/8–8/8 per model). Across families, lateral is the only region whose alignment
> grows through the language stack. See [`paper/CLAIM.md`](paper/CLAIM.md) for the
> precise claim and [`figures/f4_family_slopes.png`](figures/f4_family_slopes.png) for
> the one-figure summary.

---

## Quick start

```bash
make setup
```

This installs [uv](https://docs.astral.sh/uv/) if missing, creates a Python 3.11 environment,
installs dependencies, and runs the environment check.

```bash
make check          # verify python, packages, compute device, disk, S3 reachability
make s0-estimate    # how many GB before you download anything
```

**Before downloading any brain data you must sign the NSD Data Access Agreement** —
see [docs/NSD_ACCESS.md](docs/NSD_ACCESS.md). This is a legal precondition, not a technical one.

## Pipeline

Each stage is one command. Stages are strictly ordered; each ends with figures and a journal entry.

| Stage | Command | What it does |
|---|---|---|
| S0 | `make s0-estimate` | Estimate NSD download volume, verify access |
| S1 | `make s1-sanity` | Load betas → `(1000 images × vertices)` per subject/ROI; sanity checks |
| S2 | `make s2-activations` | Extract + cache model activations, all layers, 1000 images |
| S3 | `make s3-rsa` | RDMs, RSA scores, permutation tests, noise-ceiling normalisation |
| S4 | `make s4-analysis` | Main analysis and figures |
| S5 | `make s5-encoding` | Ridge encoding model as a robustness check |

## Layout

```
src/nsd_rsa/     library code (importable, tested)
scripts/         one CLI per stage
configs/         yaml — every knob lives here, nothing hardcoded
notebooks/       exploration only, never pipeline logic
docs/            GLOSSARY, LAB_JOURNAL, NSD_ACCESS, IDEAS
paper/           preprint draft
tests/           unit tests (RDM math, loaders)
data/ cache/     gitignored — never committed
figures/         committed, all script-generated
```

## Models under comparison

| Model | Supervision | Why it's here |
|---|---|---|
| DINOv2 ViT-B/14, ViT-L/14 | self-supervised | the hypothesis under test |
| ViT-B/16 | supervised (ImageNet) | same architecture, different objective |
| ResNet-50 | supervised (ImageNet) | classic convolutional baseline |
| CLIP ViT-B/16 | language-supervised | does language grounding help or hurt? |

## Reproducibility

Fixed seeds, pinned versions, config-driven. Every figure in `figures/` is produced by a
script — none by hand.

## Citation

If you use NSD, cite:

> Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle, L.T., Nau, M.,
> Caron, B., Pestilli, F., Charest, I., Hutchinson, J.B., Naselaris, T., Kay, K. (2022).
> A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence.
> *Nature Neuroscience* 25, 116–126.

## License

MIT for the code. NSD data is **not** redistributed here and is governed by its own terms.
