#!/usr/bin/env python
"""S3 — low-level control baselines.

The question this answers: how much of a model's brain alignment is explained by trivial
image properties that require no learning at all?

Early visual cortex responds to luminance and contrast. So do the early layers of any
vision model, because those properties dominate the input. If a model's RSA score in
early cortex is no better than what raw brightness achieves, the score says nothing about
representation — it says the model, like the brain, is sensitive to how bright a picture
is. Any RSA claim has to clear this bar, and a reviewer will ask for it.

Baselines, roughly in decreasing triviality:
  mean_luminance  — ONE number per image: its average brightness
  rms_contrast    — ONE number per image: its standard deviation
  pixels_gray     — downsampled greyscale image, i.e. raw retinal-ish input
  canny_edges     — pooled density of Canny edges: where the contours are
  gabor_energy    — energy of a Gabor filter bank (4 spatial frequencies x 6
                    orientations, pooled on a grid). This is not arbitrary: oriented
                    Gabor energy is the textbook model of V1 simple/complex cells, so
                    it is the natural hand-crafted competitor for `early`.

Usage: make s3-control
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.noise_ceiling import normalise_to_ceiling  # noqa: E402
from nsd_rsa.rdm import compute_rdm  # noqa: E402
from nsd_rsa.rsa import RDMBank, brain_rdm_bank, compare_banks  # noqa: E402


def scalar_rdm(values: np.ndarray) -> np.ndarray:
    """RDM from one number per image: distance is the absolute difference."""
    v = np.asarray(values, dtype=np.float64).ravel()
    full = np.abs(v[:, None] - v[None, :])
    from scipy.spatial.distance import squareform

    np.fill_diagonal(full, 0.0)
    return squareform(full, checks=False)


from nsd_rsa.lowlevel import edge_features, gabor_features  # noqa: E402


def build_control_bank(images_path: Path, images: np.ndarray) -> RDMBank:
    im = np.load(images_path, mmap_mode="r")[images].astype(np.float32)
    gray = im.mean(axis=3)
    downsampled = gray[:, ::8, ::8].reshape(len(images), -1).astype(np.float64)
    print("  computing edge and Gabor features")
    return RDMBank(
        ["pixels_gray", "mean_luminance", "rms_contrast", "canny_edges", "gabor_energy"],
        np.vstack([
            compute_rdm(downsampled),
            scalar_rdm(gray.mean(axis=(1, 2))),
            scalar_rdm(gray.std(axis=(1, 2))),
            compute_rdm(edge_features(gray)),
            compute_rdm(gabor_features(gray)),
        ]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rsa.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    rois = cfg["rois"]["order"]

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    images = usable_images(cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"])
    valid = common_valid_vertices(paths["betas"], paths["cache"] / "valid_vertices.npy")

    control = build_control_bank(paths["stimuli"] / "shared1000_images.npy", images)

    brain = {}
    for path in sorted(paths["betas"].glob("*_shared_betas.h5")):
        data = load_subject(path)
        brain[path.stem.split("_")[0]] = brain_rdm_bank(
            data, rois, images, metric=cfg["rdm"]["metric"], valid=valid
        )
        del data
    subjects = sorted(brain)

    results_path = paths["cache"] / "rsa_results.json"
    if not results_path.exists():
        print("Run scripts/s3_rsa.py first.")
        return 1
    res = json.loads(results_path.read_text())
    ceilings = {k: v[0] for k, v in res["ceilings"].items()}

    raw = np.stack([compare_banks(control, brain[s]) for s in subjects]).mean(axis=0)
    norm = np.array([
        [normalise_to_ceiling(raw[i, j], ceilings[r]) for j, r in enumerate(rois)]
        for i in range(len(control))
    ])

    print(f"stimulus set: {len(images)} images, {len(subjects)} subjects")
    print("\n=== LOW-LEVEL CONTROLS (normalised to noise ceiling) ===")
    print(f"{'predictor':<18}" + "".join(f"{r:>13}" for r in rois))
    for i, lab in enumerate(control.labels):
        print(f"{lab:<18}" + "".join(f"{v:>13.3f}" for v in norm[i]))

    print("\n=== BEST MODEL PER ROI, for comparison ===")
    print(f"{'model':<18}" + "".join(f"{r:>13}" for r in rois))
    for model, r in res["models"].items():
        print(f"{model:<18}" + "".join(f"{v:>13.3f}" for v in r["best_norm"]))

    print("\n=== MARGIN OVER THE STRONGEST TRIVIAL BASELINE ===")
    print("How many times better than the best low-level control. Below ~2x, the model is")
    print("not clearly doing more than tracking brightness or contrast.")
    floor = np.nanmax(norm, axis=0)
    print(f"{'model':<18}" + "".join(f"{r:>13}" for r in rois))
    for model, r in res["models"].items():
        ratios = np.array(r["best_norm"]) / np.maximum(floor, 1e-6)
        print(f"{model:<18}" + "".join(f"{v:>13.1f}" for v in ratios))
    print(f"{'(control floor)':<18}" + "".join(f"{v:>13.3f}" for v in floor))

    out = paths["cache"] / "lowlevel_control.json"
    out.write_text(json.dumps({
        "rois": rois,
        "labels": control.labels,
        "raw": raw.tolist(),
        "norm": norm.tolist(),
    }, indent=1))
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
