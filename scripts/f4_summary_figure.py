#!/usr/bin/env python
"""F4 — the preprint's headline figure: LLM-layer slopes across families and scales.

One panel per cortical region. X: the five VLMs (SmolVLM ladder, then the two other
families). Y: per-subject slope of noise-ceiling-normalised RSA across LLM layers
(x100), one dot per subject, bar = subject mean. This is the whole claim in one image:
early falls everywhere, lateral rises everywhere, ventral flips with scale inside
SmolVLM and is negative in the other families.

Reads the per-model readout caches written by f1_readout_robustness.py
(cache/f1_readouts_<model>.npz), so run that first for every model shown.

Usage: python scripts/f4_summary_figure.py --config configs/vlm.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import average_repeats, common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.noise_ceiling import normalise_to_ceiling  # noqa: E402
from nsd_rsa.rdm import compare_rdms, compute_rdm  # noqa: E402

ROIS = ("early", "midventral", "ventral", "lateral")
MODELS = ("smolvlm_256m", "smolvlm_500m", "smolvlm_2b", "qwen2vl_2b", "llava_ov_05b")
LABELS = {
    "smolvlm_256m": "SmolVLM\n256M",
    "smolvlm_500m": "SmolVLM\n500M",
    "smolvlm_2b": "SmolVLM\n2.2B",
    "qwen2vl_2b": "Qwen2-VL\n2B",
    "llava_ov_05b": "LLaVA-OV\n0.5B",
}
POOL = "trim"  # primary readout, chosen in F1


def per_subject_slopes(npz: Path, brain: dict, subjects: list[str], ceilings: dict) -> dict:
    acts = np.load(npz)
    llm = sorted((k for k in acts.files if k.startswith("llm.") and k.endswith(f".{POOL}")),
                 key=lambda k: int(k.split(".")[1]))
    rdms = {k: compute_rdm(acts[k].astype(np.float64)) for k in llm}

    out: dict[str, list[float]] = {}
    for roi in ROIS:
        slopes = []
        for s in subjects:
            y = np.array([
                normalise_to_ceiling(compare_rdms(rdms[k], brain[roi][s]), ceilings[roi])
                for k in llm
            ])
            slopes.append(float(np.polyfit(np.arange(len(y)), y, 1)[0] * 100))
        out[roi] = slopes
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    idx = usable_images(cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"])
    valid = common_valid_vertices(paths["betas"], paths["cache"] / "valid_vertices.npy")
    ceilings = {k: v[0] for k, v in
                json.loads((paths["cache"] / "rsa_results.json").read_text())["ceilings"].items()}

    print("building brain RDMs")
    brain: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(paths["betas"].glob("*_shared_betas.h5")):
        data = load_subject(path)
        for roi in ROIS:
            patterns, _ = average_repeats(data, images=idx, roi=roi, valid=valid)
            brain.setdefault(roi, {})[data.subject] = compute_rdm(patterns.astype(np.float64))
        del data
    subjects = sorted(brain[ROIS[0]])

    slopes = {}
    for model in MODELS:
        npz = paths["cache"] / f"f1_readouts_{model}.npz"
        if not npz.exists():
            print(f"  missing {npz.name} — run f1_readout_robustness.py --models {model}")
            continue
        slopes[model] = per_subject_slopes(npz, brain, subjects, ceilings)
        print(f"  {model}: done")

    (paths["cache"] / "f4_summary.json").write_text(json.dumps(
        {"pool": POOL, "subjects": subjects, "slopes_x100": slopes}, indent=1))

    models = [m for m in MODELS if m in slopes]
    fig, axes = plt.subplots(1, len(ROIS), figsize=(3.4 * len(ROIS), 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, roi in zip(axes, ROIS, strict=False):
        for i, model in enumerate(models):
            vals = np.array(slopes[model][roi])
            colour = "#2a6f97" if i < 3 else "#c1121f" if i == 3 else "#e07a00"
            jitter = rng.uniform(-0.13, 0.13, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.75,
                       color=colour, zorder=3)
            ax.hlines(vals.mean(), i - 0.28, i + 0.28, color=colour, lw=2.5, zorder=4)
            n_pos = int((vals > 0).sum())
            ax.text(i, 0.02, f"{n_pos}/8", ha="center", fontsize=7, color="0.35",
                    transform=ax.get_xaxis_transform())
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABELS[m] for m in models], fontsize=7.5)
        ax.set_title(roi, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("slope of RSA/ceiling across LLM layers (x100)")
    fig.suptitle("Across three VLM families, early-cortex alignment falls through the "
                 "language stack and lateral-stream alignment rises", fontsize=11)
    fig.tight_layout()
    out = paths["figures"] / "f4_family_slopes.png"
    fig.savefig(out, dpi=170)
    print(f"figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
