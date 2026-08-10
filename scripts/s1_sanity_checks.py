#!/usr/bin/env python
"""S1 — data sanity checks and figures.

Research hygiene before analysis: confirm the data are what we think they are, and
reproduce a fact we already know to be true. If a pipeline cannot recover a known
result, no novel result it produces should be believed.

The known fact we target: **split-half reliability is higher in early visual cortex
than in anterior ventral regions.** Early cortex responds to low-level image properties
in a stable, stimulus-locked way; anterior regions are more affected by attention,
memory and task state, so their responses to a repeated image vary more. Any pipeline
that gets this backwards has a labelling or ordering bug.

Usage: make s1-sanity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.loaders import load_subject, split_half  # noqa: E402
from nsd_rsa.noise_ceiling import split_half_reliability, voxel_noise_ceiling  # noqa: E402
from nsd_rsa.rois import STREAM_LABELS, VENTRAL_HIERARCHY  # noqa: E402

# Anatomical order: posterior/low-level on the left, anterior/high-level on the right.
ROI_ORDER = ["early", "midventral", "ventral", "midlateral", "lateral", "midparietal", "parietal"]
PALETTE = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(ROI_ORDER)))


def load_ncsnr(meta_dir: Path, subject: str, vertex_index: np.ndarray) -> np.ndarray | None:
    """NSD's per-vertex signal-to-noise estimate, restricted to our kept vertices."""
    import nibabel as nib

    parts = []
    for hemi in ("lh", "rh"):
        p = meta_dir / subject / f"{hemi}.ncsnr.mgh"
        if not p.exists():
            return None
        parts.append(np.squeeze(np.asarray(nib.load(str(p)).dataobj)).astype(np.float64))
    return np.concatenate(parts)[vertex_index]


def analyse_subject(path: Path, meta_dir: Path, seed: int) -> dict:
    data = load_subject(path)
    out: dict = {"subject": data.subject, "roi_counts": data.roi_counts()}

    print(f"\n=== {data.subject} ===")
    print(f"  betas matrix        : {data.betas.shape}  ({data.betas.nbytes/1e6:.0f} MB)")
    print(f"  images / trials     : {data.n_images} / {len(data.image)}")

    v = data.betas.ravel()
    finite = np.isfinite(v)
    out["beta_stats"] = {
        "mean": float(v[finite].mean()),
        "std": float(v[finite].std()),
        "pct_nonfinite": float(100 * (~finite).mean()),
        "p1": float(np.percentile(v[finite], 1)),
        "p99": float(np.percentile(v[finite], 99)),
    }
    s = out["beta_stats"]
    print(f"  beta distribution   : mean={s['mean']:+.3f}  std={s['std']:.3f}  "
          f"1-99% [{s['p1']:+.2f}, {s['p99']:+.2f}]  non-finite={s['pct_nonfinite']:.4f}%")

    # --- the key check: split-half reliability per ROI ---
    rel: dict[str, float] = {}
    for roi in ROI_ORDER:
        mask = data.roi_mask(roi)
        if mask.sum() == 0:
            continue
        a, b, _ = split_half(data, roi=roi, seed=seed)
        r = split_half_reliability(a, b)
        rel[roi] = float(np.nanmean(r))
    out["reliability"] = rel

    print("  split-half reliability (Spearman-Brown corrected):")
    for roi in ROI_ORDER:
        if roi in rel:
            print(f"      {roi:<12} {rel[roi]:+.3f}   n={data.roi_counts()[roi]:>6,} vertices")

    # --- ncsnr-derived ceiling, an independent estimate of the same thing ---
    ncsnr = load_ncsnr(meta_dir, data.subject, data.vertex_index)
    if ncsnr is not None:
        nc = voxel_noise_ceiling(ncsnr, n_repeats=3)
        out["noise_ceiling"] = {
            roi: float(np.nanmean(nc[data.roi_mask(roi)]))
            for roi in ROI_ORDER
            if data.roi_mask(roi).sum() > 0
        }
        print("  NSD ncsnr noise ceiling (% explainable variance, 3 repeats):")
        for roi, val in out["noise_ceiling"].items():
            print(f"      {roi:<12} {val:5.1f}%")

    out["_data"] = data
    out["_ncsnr"] = ncsnr
    return out


def figure_beta_distribution(results: list[dict], fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for res in results:
        data = res["_data"]
        sample = data.betas[:: max(1, len(data.betas) // 200)].ravel()
        sample = sample[np.isfinite(sample)]
        axes[0].hist(sample, bins=200, range=(-10, 10), histtype="step", density=True,
                     label=res["subject"], linewidth=1.4)

    axes[0].set_xlabel("beta (% signal change)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Single-trial beta distribution")
    axes[0].legend(fontsize=8)
    axes[0].axvline(0, color="k", lw=0.5, ls=":")

    res = results[0]
    data = res["_data"]
    for roi, colour in zip(ROI_ORDER, PALETTE, strict=False):
        mask = data.roi_mask(roi)
        if mask.sum() == 0:
            continue
        sample = data.betas[:: max(1, len(data.betas) // 100)][:, mask].ravel()
        sample = sample[np.isfinite(sample)]
        axes[1].hist(sample, bins=150, range=(-8, 8), histtype="step", density=True,
                     label=roi, color=colour, linewidth=1.3)
    axes[1].set_xlabel("beta (% signal change)")
    axes[1].set_title(f"By ROI ({res['subject']})")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    out = fig_dir / "s1_beta_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_roi_counts(results: list[dict], fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(len(results), 1)
    x = np.arange(len(ROI_ORDER))
    for i, res in enumerate(results):
        counts = [res["roi_counts"].get(r, 0) for r in ROI_ORDER]
        ax.bar(x + i * width, counts, width, label=res["subject"])
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(ROI_ORDER, rotation=20, ha="right")
    ax.set_ylabel("vertices (fsaverage)")
    ax.set_title("ROI size per subject — streams atlas")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "s1_roi_vertex_counts.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_reliability(results: list[dict], fig_dir: Path) -> Path:
    """The sanity check that matters: does reliability fall from early to anterior cortex?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    width = 0.8 / max(len(results), 1)
    x = np.arange(len(ROI_ORDER))
    for i, res in enumerate(results):
        vals = [res["reliability"].get(r, np.nan) for r in ROI_ORDER]
        axes[0].bar(x + i * width, vals, width, label=res["subject"])
    axes[0].set_xticks(x + width * (len(results) - 1) / 2)
    axes[0].set_xticklabels(ROI_ORDER, rotation=20, ha="right")
    axes[0].set_ylabel("split-half reliability (r)")
    axes[0].set_title("Reliability by ROI")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)

    for res in results:
        vals = [res["reliability"].get(r, np.nan) for r in VENTRAL_HIERARCHY]
        axes[1].plot(VENTRAL_HIERARCHY, vals, "o-", label=res["subject"], linewidth=2)
    axes[1].set_ylabel("split-half reliability (r)")
    axes[1].set_title("Ventral hierarchy: expect a decline from early to ventral")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = fig_dir / "s1_split_half_reliability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def verdict(results: list[dict]) -> bool:
    """State plainly whether the known-fact check passed, per subject."""
    print("\n=== KNOWN-FACT CHECK: reliability(early) > reliability(ventral)? ===")
    passed = 0
    for res in results:
        rel = res["reliability"]
        if "early" not in rel or "ventral" not in rel:
            continue
        ok = rel["early"] > rel["ventral"]
        passed += ok
        print(f"  {res['subject']}: early={rel['early']:+.3f}  ventral={rel['ventral']:+.3f}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  {passed}/{len(results)} subjects reproduce the expected ordering")
    return passed == len(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)

    files = sorted(paths["betas"].glob("*_shared_betas.h5"))
    if not files:
        print("No extracted betas found. Run: python scripts/s1_download_betas.py")
        return 1

    print(f"Found {len(files)} subject file(s): {[f.stem.split('_')[0] for f in files]}")
    print(f"ROI label map: {STREAM_LABELS}")

    results = [analyse_subject(f, paths["meta"], cfg.get("seed", 0)) for f in files]

    fig_dir = paths["figures"]
    made = [
        figure_beta_distribution(results, fig_dir),
        figure_roi_counts(results, fig_dir),
        figure_reliability(results, fig_dir),
    ]

    ok = verdict(results)

    print("\nFigures written:")
    for p in made:
        print(f"  {p}")
    if not ok:
        print("\nWARNING: the known-fact check did not pass for every subject. "
              "Investigate before trusting downstream results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
