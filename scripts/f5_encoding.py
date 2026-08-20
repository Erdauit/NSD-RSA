#!/usr/bin/env python
"""F5 — voxelwise encoding as an independent check of the RSA claim.

RSA compares representational *geometries*; an encoding model asks a different question:
how much of each vertex's actual response can be linearly predicted from the model's
features? If both methods show the same picture — early-cortex prediction falling across
LLM layers while lateral-stream prediction rises — the claim no longer depends on the
choice of method.

Design (deliberately minimal, per PHASE2):
  * features: image-token `trim` readouts of every LLM layer (from the F1 caches),
    for one 2B-class model of each family lineage
  * ridge regression per layer -> all vertices of the four streams ROIs,
    5-fold CV over images, per-vertex alpha via efficient leave-one-out GCV (RidgeCV)
  * out-of-fold R² per vertex, normalised by the ncsnr-derived noise ceiling
    (fraction of stimulus-driven variance at 3 repeats), median over vertices
  * per-subject slope of that quantity across LLM layers — the exact encoding
    analogue of the RSA slope table

Usage: python scripts/f5_encoding.py --config configs/vlm.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Windows consoles default to a legacy codepage that cannot print "²"; a crash at the
# final print after hours of computation is exactly how we learned this.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import average_repeats, common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.noise_ceiling import voxel_noise_ceiling  # noqa: E402

MODELS = ("smolvlm_2b", "qwen2vl_2b")
ROIS = ("early", "midventral", "ventral", "lateral")
POOL = "trim"
ALPHAS = np.logspace(1, 5, 9)
NC_FLOOR = 0.05  # drop vertices with <5% explainable variance: dividing by ~0 lies


def load_ncsnr(meta_dir: Path, subject: str, vertex_index: np.ndarray) -> np.ndarray:
    import nibabel as nib

    parts = [
        np.asarray(nib.load(meta_dir / subject / f"{hemi}.ncsnr.mgh").get_fdata()).ravel()
        for hemi in ("lh", "rh")
    ]
    return np.concatenate(parts)[vertex_index]


def out_of_fold_r2(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Per-vertex R² from cross-validated ridge predictions."""
    preds = np.empty_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        model.fit((X[tr] - mu) / sd, Y[tr])
        preds[te] = model.predict((X[te] - mu) / sd)
    ss_res = ((Y - preds) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    idx = usable_images(cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"])
    valid = common_valid_vertices(paths["betas"], paths["cache"] / "valid_vertices.npy")

    features: dict[str, dict[str, np.ndarray]] = {}
    for model in args.models:
        acts = np.load(paths["cache"] / f"f1_readouts_{model}.npz")
        llm = sorted((k for k in acts.files if k.startswith("llm.") and k.endswith(f".{POOL}")),
                     key=lambda k: int(k.split(".")[1]))
        features[model] = {k: acts[k].astype(np.float64) for k in llm}
        print(f"{model}: {len(llm)} LLM layers, {features[model][llm[0]].shape[1]} dims")

    # scores[model][roi] -> (n_subjects, n_layers) median normalised R²
    scores: dict[str, dict[str, list[list[float]]]] = {
        m: {r: [] for r in ROIS} for m in args.models
    }
    subjects = []
    for path in sorted(paths["betas"].glob("*_shared_betas.h5")):
        data = load_subject(path)
        subjects.append(data.subject)
        ncsnr = load_ncsnr(paths["meta"], data.subject, data.vertex_index)
        t0 = time.time()
        for roi in ROIS:
            Y, _ = average_repeats(data, images=idx, roi=roi, valid=valid)
            Y = Y.astype(np.float64)
            from nsd_rsa.loaders import _vertex_selector

            nc = voxel_noise_ceiling(ncsnr[_vertex_selector(data, roi, valid)], 3) / 100.0
            keep = nc >= NC_FLOOR
            Y, nc = Y[:, keep], nc[keep]

            for model in args.models:
                profile = []
                for k, X in features[model].items():
                    r2 = out_of_fold_r2(X, Y, seed=cfg.get("seed", 0))
                    profile.append(float(np.nanmedian(np.clip(r2, 0, None) / nc)))
                scores[model][roi].append(profile)
        print(f"  {data.subject}: {time.time()-t0:.0f}s "
              f"({sum(len(f) for f in features.values())} layers x {len(ROIS)} ROIs)")
        # Incremental checkpoint: hours of computation must survive any later crash.
        (paths["cache"] / "f5_encoding_partial.json").write_text(
            json.dumps({"subjects": subjects, "scores": scores}, indent=1))
        del data

    # ---- slopes across LLM layers, per subject ----
    summary: dict = {"subjects": subjects, "scores": scores}
    for model in args.models:
        for roi in ROIS:
            arr = np.asarray(scores[model][roi])  # (subjects, layers)
            slopes = np.array([np.polyfit(np.arange(arr.shape[1]), row, 1)[0] for row in arr])
            summary.setdefault("slopes", {})[f"{model}|{roi}"] = {
                "slope_x100": float(slopes.mean() * 100),
                "n_positive": int((slopes > 0).sum()),
                "p": float(wilcoxon(slopes).pvalue),
                "per_subject_x100": [float(s * 100) for s in slopes],
            }
    # Persist BEFORE printing: results must not depend on the console behaving.
    (paths["cache"] / "f5_encoding.json").write_text(json.dumps(summary, indent=1))

    print("\n" + "=" * 86)
    print("ENCODING SLOPE across LLM layers — median R²/NC, x100; subjects rising; Wilcoxon p")
    print("=" * 86)
    print(f"{'model':<15}{'ROI':<12}{'slope':>9}{'rising':>9}{'p':>9}{'first':>9}{'last':>9}")
    for model in args.models:
        for roi in ROIS:
            arr = np.asarray(scores[model][roi])
            s = summary["slopes"][f"{model}|{roi}"]
            print(f"{model:<15}{roi:<12}{s['slope_x100']:>+9.3f}{s['n_positive']:>7}/8"
                  f"{s['p']:>9.4f}{arr[:, 0].mean():>9.3f}{arr[:, -1].mean():>9.3f}")

    # ---- profile figure ----
    fig, axes = plt.subplots(1, len(ROIS), figsize=(3.3 * len(ROIS), 3.4), sharex=False)
    for ax, roi in zip(axes, ROIS, strict=False):
        for model, colour in zip(args.models, ("#2a6f97", "#c1121f"), strict=False):
            arr = np.asarray(scores[model][roi])
            x = np.arange(arr.shape[1])
            ax.plot(x, arr.mean(0), color=colour, lw=2, label=model)
            ax.fill_between(x, arr.mean(0) - arr.std(0) / np.sqrt(8),
                            arr.mean(0) + arr.std(0) / np.sqrt(8), color=colour, alpha=0.2)
        ax.set_title(roi, fontsize=10)
        ax.set_xlabel("LLM layer")
    axes[0].set_ylabel("median R² / noise ceiling")
    axes[0].legend(fontsize=8)
    fig.suptitle("Encoding model across the language stack (5-fold CV ridge, trim readout)",
                 fontsize=11)
    fig.tight_layout()
    out = paths["figures"] / "f5_encoding_profile.png"
    fig.savefig(out, dpi=170)
    print(f"\nfigure -> {out}\nresults -> cache/f5_encoding.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
