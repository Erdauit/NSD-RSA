#!/usr/bin/env python
"""S3 — the RSA core: model layers against brain regions.

The problem RSA solves: a model layer has 768 dimensions, a brain ROI has 19,065
vertices, and there is no correspondence between a dimension and a vertex. You cannot
compare the spaces directly. So instead of comparing the spaces, compare the *geometry
of the stimuli inside* them.

For each representation, compute the pairwise distance between every pair of the 515
images — that is the RDM. Its size depends only on the number of images, so the model's
RDM and the brain's RDM are both 515x515 and directly comparable. If both systems place
the same images near each other, they encode the same structure, whatever coordinates
they use.

Everything is normalised to the noise ceiling. Without that, an ROI with poor SNR looks
like a region models fail to explain, when in fact nothing could explain it.

Outputs a layer x ROI heatmap per model, plus a results table.

Usage: make s3-rsa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.noise_ceiling import normalise_to_ceiling, rsa_noise_ceiling  # noqa: E402
from nsd_rsa.rdm import permutation_test  # noqa: E402
from nsd_rsa.rsa import (  # noqa: E402
    RDMBank,
    brain_rdm_bank,
    compare_banks,
    layer_depth_index,
    model_rdm_bank,
)


def build_brain_banks(
    paths: dict, cfg: dict, images: np.ndarray, valid: np.ndarray
) -> tuple[dict[str, RDMBank], list[str]]:
    """One RDM bank per subject, over the configured ROIs."""
    rois = cfg["rois"]["order"]
    banks, found = {}, []
    for path in sorted(paths["betas"].glob("*_shared_betas.h5")):
        subject = path.stem.split("_")[0]
        t0 = time.time()
        data = load_subject(path)
        banks[subject] = brain_rdm_bank(
            data, rois, images, metric=cfg["rdm"]["metric"], valid=valid
        )
        found.append(subject)
        print(f"  {subject}: {len(rois)} ROI RDMs over {len(images)} images "
              f"({time.time()-t0:.0f}s)")
        del data
    return banks, found


def build_model_banks(paths: dict, cfg: dict, images: np.ndarray) -> dict[str, RDMBank]:
    banks = {}
    size = cfg["models"]["input_size"]
    for name in cfg["models"]["names"]:
        path = paths["cache"] / "activations" / f"{name}_{size}.h5"
        if not path.exists():
            print(f"  {name}: no cache at {path}, skipping")
            continue
        t0 = time.time()
        banks[name] = model_rdm_bank(path, images, metric=cfg["rdm"]["metric"])
        print(f"  {name}: {len(banks[name])} readout RDMs ({time.time()-t0:.0f}s)")
    return banks


def compute_ceilings(brain: dict[str, RDMBank], rois: list[str]) -> dict[str, tuple[float, float]]:
    """Noise ceiling per ROI, across subjects.

    Upper bound: each subject's RDM against the mean of all subjects (including
    themselves, so it flatters). Lower bound: against the mean of the others
    (leave-one-out, so it is conservative). A model landing between them is doing as well
    as these data allow.
    """
    subjects = sorted(brain)
    return {
        roi: rsa_noise_ceiling(np.vstack([brain[s].rdms[i] for s in subjects]))
        for i, roi in enumerate(rois)
    }


def figure_heatmap(
    model: str, labels: list[str], scores: np.ndarray, rois: list[str], fig_dir: Path
) -> Path:
    """Layer x ROI heatmap of ceiling-normalised RSA."""
    fig, ax = plt.subplots(figsize=(1.1 * len(rois) + 3.5, 0.28 * len(labels) + 2.2))
    vmax = np.nanmax(np.abs(scores)) if np.isfinite(scores).any() else 1.0
    im = ax.imshow(scores, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"{model}: RSA normalised to noise ceiling", fontsize=10)
    fig.colorbar(im, ax=ax, label="RSA / noise ceiling (lower bound)")
    fig.tight_layout()
    out = fig_dir / f"s3_heatmap_{model}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_hierarchy(results: dict, rois: list[str], fig_dir: Path) -> Path:
    """Best-matching layer depth per ROI — the hierarchy claim, in one plot."""
    hierarchy = [r for r in rois if r in ("early", "midventral", "ventral")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for model, res in results.items():
        depth = [res["best_depth"][rois.index(r)] for r in hierarchy]
        axes[0].plot(hierarchy, depth, "o-", label=model, linewidth=2)
        peak = [res["best_norm"][rois.index(r)] for r in hierarchy]
        axes[1].plot(hierarchy, peak, "o-", label=model, linewidth=2)

    axes[0].set_ylabel("relative depth of best-matching layer (0=first, 1=last)")
    axes[0].set_title("Does the best layer move deeper along the ventral stream?")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel("best RSA / noise ceiling")
    axes[1].set_title("How much of the explainable structure is captured?")
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = fig_dir / "s3_hierarchy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rsa.yaml")
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    rois = cfg["rois"]["order"]

    # --- stimulus set ---
    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    images = usable_images(
        cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"]
    )
    print(f"stimulus set: {len(images)} images with >= {cfg['stimuli']['min_repeats']} "
          f"repeats in all {len(cfg['stimuli']['subjects'])} subjects")
    print(f"              -> {len(images)*(len(images)-1)//2:,} stimulus pairs per RDM\n")

    # Vertices with data in every subject. NSD's slab acquisition misses the posterior
    # edge of V1 in some subjects, and an ROI must mean the same cortical locations in
    # everyone or the noise ceiling would partly measure coverage differences.
    valid = common_valid_vertices(paths["betas"], paths["cache"] / "valid_vertices.npy")
    print(f"valid vertices: {valid.sum():,} of {len(valid):,} "
          f"({(~valid).sum()} dropped as missing in at least one subject)\n")

    print("building brain RDMs")
    brain, subjects = build_brain_banks(paths, cfg, images, valid)
    if len(subjects) < 3:
        print(f"\nNeed at least 3 subjects for a noise ceiling; have {len(subjects)}. "
              "Let the download finish.")
        return 1

    print("\nbuilding model RDMs")
    models = build_model_banks(paths, cfg, images)
    if not models:
        print("No model activations found. Run: make s2-activations")
        return 1

    # --- noise ceiling ---
    print(f"\nnoise ceiling over {len(subjects)} subjects ({', '.join(subjects)})")
    ceilings = compute_ceilings(brain, rois)
    for roi in rois:
        lo, hi = ceilings[roi]
        print(f"  {roi:<12} lower={lo:.3f}  upper={hi:.3f}")

    # --- RSA ---
    print("\nRSA: model readouts x ROIs, averaged over subjects")
    results = {}
    for model, bank in models.items():
        per_subject = np.stack([compare_banks(bank, brain[s]) for s in subjects])
        raw = per_subject.mean(axis=0)          # (n_readouts, n_rois)
        sem = per_subject.std(axis=0) / np.sqrt(len(subjects))

        norm = np.empty_like(raw)
        for j, roi in enumerate(rois):
            lo, _ = ceilings[roi]
            norm[:, j] = [normalise_to_ceiling(v, lo) for v in raw[:, j]]

        depth = layer_depth_index(bank.labels)
        best_idx = np.nanargmax(norm, axis=0)
        results[model] = {
            "labels": bank.labels,
            "raw": raw,
            "sem": sem,
            "norm": norm,
            "per_subject": per_subject,
            "best_idx": best_idx,
            "best_norm": norm[best_idx, np.arange(len(rois))],
            "best_raw": raw[best_idx, np.arange(len(rois))],
            "best_label": [bank.labels[i] for i in best_idx],
            "best_depth": depth[best_idx],
        }
        print(f"  {model}: peak normalised RSA = {np.nanmax(norm):.3f}")

    # --- significance on the headline cell per (model, ROI) ---
    if not args.skip_permutation:
        n_perm = cfg["inference"]["n_permutations"]
        print(f"\npermutation test on the best readout per (model, ROI), {n_perm} permutations")
        for model, res in results.items():
            pvals = []
            for j in range(len(rois)):
                model_rdm = models[model].rdms[res["best_idx"][j]]
                group_rdm = np.mean([brain[s].rdms[j] for s in subjects], axis=0)
                _, p = permutation_test(model_rdm, group_rdm, n_perm=n_perm, seed=cfg["seed"])
                pvals.append(p)
            res["best_p"] = pvals
            sig = sum(p < cfg["inference"]["alpha"] for p in pvals)
            print(f"  {model}: {sig}/{len(rois)} ROIs significant at "
                  f"p < {cfg['inference']['alpha']}")

    # --- figures and table ---
    print()
    for model, res in results.items():
        p = figure_heatmap(model, res["labels"], res["norm"], rois, paths["figures"])
        print(f"  {p}")
    print(f"  {figure_hierarchy(results, rois, paths['figures'])}")

    print("\n" + "=" * 96)
    print("BEST READOUT PER ROI  (normalised = RSA / noise-ceiling lower bound)")
    print("=" * 96)
    header = f"{'model':<15}{'ROI':<13}{'best readout':<22}{'depth':>6}{'raw':>8}{'norm':>8}{'p':>9}"
    print(header)
    for model, res in results.items():
        for j, roi in enumerate(rois):
            p = res.get("best_p", [float('nan')] * len(rois))[j]
            print(f"{model:<15}{roi:<13}{res['best_label'][j]:<22}"
                  f"{res['best_depth'][j]:>6.2f}{res['best_raw'][j]:>8.3f}"
                  f"{res['best_norm'][j]:>8.3f}{p:>9.4f}")

    # --- persist ---
    out = paths["cache"] / "rsa_results.json"
    payload = {
        "n_images": int(len(images)),
        "subjects": subjects,
        "rois": rois,
        "ceilings": {k: list(v) for k, v in ceilings.items()},
        "models": {
            m: {
                "labels": r["labels"],
                "raw": r["raw"].tolist(),
                "norm": r["norm"].tolist(),
                "sem": r["sem"].tolist(),
                "best_label": r["best_label"],
                "best_depth": r["best_depth"].tolist(),
                "best_raw": r["best_raw"].tolist(),
                "best_norm": r["best_norm"].tolist(),
                "best_p": r.get("best_p"),
            }
            for m, r in results.items()
        },
    }
    out.write_text(json.dumps(payload, indent=1))
    np.savez_compressed(
        paths["cache"] / "rsa_per_subject.npz",
        **{m: r["per_subject"] for m, r in results.items()},
        subjects=np.array(subjects),
        rois=np.array(rois),
    )
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
