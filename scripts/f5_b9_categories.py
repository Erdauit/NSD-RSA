#!/usr/bin/env python
"""B9 — does the lateral-stream rise survive partialling out COCO object categories?

The deflationary reading (Conwell et al. 2023): language-model alignment with high-level
visual cortex is largely carried by *object content* — a coarse inventory of nouns. If
our lateral-stream rise through the LLM stack is nothing but "the model lists the
objects", then partialling an object-category RDM out of the model-brain relationship
should flatten the rise. If the rise survives, the language stack is adding structure
beyond the inventory.

Construction:
  * NSD images are crops of COCO images (`cropBox` in nsd_stim_info_merged.csv), so an
    object counts only if its bbox centre falls inside the crop.
  * category vector per image: 80-dim multi-hot (plus an area-weighted variant for a
    sanity check).
  * category RDM: Jaccard distance between multi-hot vectors (area variant: 1-Pearson).
  * partial Spearman rho(model, brain | category): rank-transform all three condensed
    RDMs, residualise model and brain on category, correlate the residuals.
  * the deliverable: per-subject slopes of ceiling-normalised alignment across LLM
    layers, plain vs partial, per model and ROI.

Usage: python scripts/f5_b9_categories.py --config configs/vlm.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import rankdata, wilcoxon  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import average_repeats, common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.noise_ceiling import normalise_to_ceiling  # noqa: E402
from nsd_rsa.rdm import compare_rdms, compute_rdm  # noqa: E402

MODELS = ("smolvlm_256m", "smolvlm_500m", "smolvlm_2b", "qwen2vl_2b", "llava_ov_05b")
ROIS = ("early", "midventral", "ventral", "lateral")
POOL = "trim"


def category_vectors(meta_dir: Path, nsd_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(multi-hot, area-fraction) category matrices for the given nsdIds, crop-aware."""
    info = pd.read_csv(meta_dir / "nsd_stim_info_merged.csv", index_col="nsdId")

    anns: dict[str, dict] = {}
    sizes: dict[str, dict] = {}
    for split in ("train2017", "val2017"):
        blob = json.loads((meta_dir / "coco" / "annotations" / f"instances_{split}.json")
                          .read_text(encoding="utf-8"))
        sizes[split] = {im["id"]: (im["height"], im["width"]) for im in blob["images"]}
        by_img: dict[int, list] = {}
        for a in blob["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        anns[split] = by_img
        del blob

    cat_ids = sorted({a["category_id"] for split in anns.values() for lst in split.values()
                      for a in lst})
    cat_index = {c: i for i, c in enumerate(cat_ids)}

    hot = np.zeros((len(nsd_ids), len(cat_ids)))
    area = np.zeros_like(hot)
    missing = 0
    for row, nsd_id in enumerate(nsd_ids):
        # shared1000_nsd_ids.npy stores the 1-based 73k ids from the experiment design;
        # the CSV's nsdId index is 0-based. Off by one here silently attaches every
        # image to its neighbour's objects — caught once via below-chance decoding AUC.
        rec = info.loc[int(nsd_id) - 1]
        if not bool(rec["shared1000"]):
            raise ValueError(
                f"nsdId {int(nsd_id)-1} is not flagged shared1000 — id convention broke"
            )
        split, coco_id = rec["cocoSplit"], int(rec["cocoId"])
        top, bottom, left, right = ast.literal_eval(rec["cropBox"])
        h, w = sizes[split][coco_id]
        r0, r1 = top * h, (1 - bottom) * h
        c0, c1 = left * w, (1 - right) * w
        crop_area = max((r1 - r0) * (c1 - c0), 1.0)
        items = anns[split].get(coco_id, [])
        if not items:
            missing += 1
        for a in items:
            x, y, bw, bh = a["bbox"]
            cx, cy = x + bw / 2, y + bh / 2
            if not (c0 <= cx < c1 and r0 <= cy < r1):
                continue
            j = cat_index[a["category_id"]]
            hot[row, j] = 1.0
            area[row, j] += a["area"] / crop_area
    if missing:
        print(f"  note: {missing} images had no instance annotations at all")
    return hot, area


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y with z partialled out (all condensed RDMs)."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    zc = np.column_stack([rz, np.ones_like(rz)])
    rx_res = rx - zc @ np.linalg.lstsq(zc, rx, rcond=None)[0]
    ry_res = ry - zc @ np.linalg.lstsq(zc, ry, rcond=None)[0]
    denom = np.linalg.norm(rx_res) * np.linalg.norm(ry_res)
    return float(rx_res @ ry_res / denom) if denom > 0 else float("nan")


def slopes_table(per_subject: np.ndarray) -> tuple[float, int, float]:
    slopes = np.array([np.polyfit(np.arange(len(row)), row, 1)[0] for row in per_subject.T])
    return float(slopes.mean() * 100), int((slopes > 0).sum()), float(wilcoxon(slopes).pvalue)


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
    nsd_ids = np.load(paths["stimuli"] / "shared1000_nsd_ids.npy")[idx]
    valid = common_valid_vertices(paths["betas"], paths["cache"] / "valid_vertices.npy")
    ceilings = {k: v[0] for k, v in
                json.loads((paths["cache"] / "rsa_results.json").read_text())["ceilings"].items()}

    print("building category RDMs (crop-aware)")
    hot, area = category_vectors(paths["meta"], nsd_ids)
    n_per_img = hot.sum(1)
    print(f"  {hot.shape[1]} categories; objects per image: "
          f"median {np.median(n_per_img):.0f}, zero-object images: {(n_per_img==0).sum()}")
    np.savez_compressed(paths["cache"] / "coco_categories.npz",
                        multihot=hot, area=area, nsd_ids=nsd_ids)
    cat_rdm = pdist(hot, metric="jaccard")
    # Cosine with an epsilon column: an image with zero objects in the crop has an
    # all-zero area vector, for which correlation distance is undefined.
    cat_rdm_area = pdist(np.column_stack([area, np.full(len(area), 1e-6)]), metric="cosine")

    print("building brain RDMs")
    brain: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(paths["betas"].glob("*_shared_betas.h5")):
        data = load_subject(path)
        for roi in ROIS:
            patterns, _ = average_repeats(data, images=idx, roi=roi, valid=valid)
            brain.setdefault(roi, {})[data.subject] = compute_rdm(patterns.astype(np.float64))
        del data
    subjects = sorted(brain[ROIS[0]])

    # How much does the category RDM itself explain? Context for everything below.
    print("\ncategory RDM alone (normalised to ceiling):")
    for roi in ROIS:
        vals = [normalise_to_ceiling(compare_rdms(cat_rdm, brain[roi][s]), ceilings[roi])
                for s in subjects]
        print(f"  {roi:<12}{np.mean(vals):.3f}")

    results: dict = {}
    print("\n" + "=" * 96)
    print("LLM SLOPES x100 — plain vs partial(category), per-subject rise counts, Wilcoxon p")
    print("=" * 96)
    print(f"{'model':<15}{'ROI':<12}{'plain':>21}{'partial':>21}{'partial(area)':>21}")
    for model in args.models:
        acts = np.load(paths["cache"] / f"f1_readouts_{model}.npz")
        llm = sorted((k for k in acts.files if k.startswith("llm.") and k.endswith(f".{POOL}")),
                     key=lambda k: int(k.split(".")[1]))
        rdms = {k: compute_rdm(acts[k].astype(np.float64)) for k in llm}

        for roi in ROIS:
            plain = np.empty((len(llm), len(subjects)))
            part = np.empty_like(plain)
            part_a = np.empty_like(plain)
            for li, k in enumerate(llm):
                for si, s in enumerate(subjects):
                    b = brain[roi][s]
                    plain[li, si] = normalise_to_ceiling(
                        compare_rdms(rdms[k], b), ceilings[roi])
                    part[li, si] = normalise_to_ceiling(
                        partial_spearman(rdms[k], b, cat_rdm), ceilings[roi])
                    part_a[li, si] = normalise_to_ceiling(
                        partial_spearman(rdms[k], b, cat_rdm_area), ceilings[roi])
            cells = []
            row: dict = {}
            for name, mat in (("plain", plain), ("partial", part), ("partial_area", part_a)):
                sl, npos, p = slopes_table(mat)
                cells.append(f"{sl:>+9.3f} {npos}/8 p={p:.3f}")
                row[name] = {"slope_x100": sl, "n_positive": npos, "p": p}
            results.setdefault(model, {})[roi] = row
            print(f"{model:<15}{roi:<12}" + "".join(f"{c:>21}" for c in cells))

    out = paths["cache"] / "f5_b9_categories.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
