#!/usr/bin/env python
"""F7 — the literature-grounded image feature battery (docs/FEATURE_BATTERY.md).

Computes tier-1 features for the 515 analysis images and caches them:

  cache/feature_battery.csv    scalar features, one row per image
  cache/feature_battery.npz    array-valued families: gist (512), ps (710), hue (10)

Families and sources are documented in docs/FEATURE_BATTERY.md; every feature here has a
published lineage (Groen 2013; Portilla-Simoncelli via Henderson 2023; Oliva-Torralba;
Rosenholtz 2007; Long 2018; Pennock 2021/2025; COCO annotations).

Usage: python scripts/f7_feature_battery.py --config configs/vlm.yaml
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.lowlevel import (  # noqa: E402
    edge_features,
    feature_congestion_proxy,
    fourier_stats,
    gist_features,
    orientation_stats,
    subband_entropy,
    weibull_contrast,
)


def instance_features(meta_dir: Path, nsd_ids: np.ndarray) -> pd.DataFrame:
    """Crop-aware object features from COCO instance annotations (same conventions as
    f5_b9_categories.py, including the 1-based/0-based id fix and its guard)."""
    info = pd.read_csv(meta_dir / "nsd_stim_info_merged.csv", index_col="nsdId")
    anns: dict[str, dict] = {}
    sizes: dict[str, dict] = {}
    person_ids: set[int] = set()
    for split in ("train2017", "val2017"):
        blob = json.loads((meta_dir / "coco" / "annotations" / f"instances_{split}.json")
                          .read_text(encoding="utf-8"))
        sizes[split] = {im["id"]: (im["height"], im["width"]) for im in blob["images"]}
        person_ids = {c["id"] for c in blob["categories"] if c["name"] == "person"}
        by_img: dict[int, list] = {}
        for a in blob["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        anns[split] = by_img
        del blob

    rows = []
    boxes_all = []
    for nsd_id in nsd_ids:
        rec = info.loc[int(nsd_id) - 1]
        if not bool(rec["shared1000"]):
            raise ValueError("id convention broke — see f5_b9_categories.py")
        split, coco_id = rec["cocoSplit"], int(rec["cocoId"])
        top, bottom, left, right = ast.literal_eval(rec["cropBox"])
        h, w = sizes[split][coco_id]
        r0, r1 = top * h, (1 - bottom) * h
        c0, c1 = left * w, (1 - right) * w
        crop_h, crop_w = max(r1 - r0, 1.0), max(c1 - c0, 1.0)
        crop_area = crop_h * crop_w
        n = 0
        cats: set[int] = set()
        aspects, areas = [], []
        person_area = 0.0
        boxes = []  # (row0, row1, col0, col1) as fractions of the crop
        for a in anns[split].get(coco_id, []):
            x, y, bw, bh = a["bbox"]
            cx, cy = x + bw / 2, y + bh / 2
            if not (c0 <= cx < c1 and r0 <= cy < r1):
                continue
            n += 1
            cats.add(a["category_id"])
            if bh > 1:
                aspects.append(bw / bh)
            areas.append(a["area"] / crop_area)
            if a["category_id"] in person_ids:
                person_area += a["area"] / crop_area
            boxes.append((
                np.clip((y - r0) / crop_h, 0, 1), np.clip((y + bh - r0) / crop_h, 0, 1),
                np.clip((x - c0) / crop_w, 0, 1), np.clip((x + bw - c0) / crop_w, 0, 1),
            ))
        rows.append({
            "n_instances": n,
            "n_categories": len(cats),
            "area_covered": float(np.clip(sum(areas), 0, 1)),
            "mean_bbox_aspect": float(np.mean(aspects)) if aspects else 1.0,
            "max_bbox_area": float(max(areas)) if areas else 0.0,
            "person_area": float(np.clip(person_area, 0, 1)),
        })
        boxes_all.append(boxes)
    return pd.DataFrame(rows), boxes_all


def object_background_warmth(im: np.ndarray, boxes) -> tuple[float, float]:
    """Warm-hue share inside the union of object bboxes vs outside (Pennock 2025:
    object pixels are warmer than backgrounds; bboxes approximate masks)."""
    import matplotlib.colors as mcolors

    hsv = mcolors.rgb_to_hsv(im / 255.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    w = s * v
    warm = ((h < 0.125) | (h > 0.875)).astype(float)
    mask = np.zeros(im.shape[:2], dtype=bool)
    H, W = mask.shape
    for r0, r1, c0, c1 in boxes:
        mask[int(r0 * H):max(int(r1 * H), int(r0 * H) + 1),
             int(c0 * W):max(int(c1 * W), int(c0 * W) + 1)] = True

    def share(m):
        ww = w[m]
        return float((warm[m] * ww).sum() / ww.sum()) if ww.sum() > 1e-6 else np.nan

    return share(mask), share(~mask)


def ps_statistics(gray: np.ndarray) -> np.ndarray:
    """Portilla-Simoncelli texture statistics via plenoptic (Henderson 2023 lineage)."""
    import torch
    from plenoptic.models import PortillaSimoncelli
    from skimage.transform import resize

    ps = PortillaSimoncelli((256, 256))
    out = np.empty((len(gray), 710), dtype=np.float32)
    with torch.no_grad():
        for i, g in enumerate(gray):
            small = resize(g / 255.0, (256, 256), anti_aliasing=True)
            t = torch.from_numpy(small).float()[None, None]
            out[i] = ps(t).numpy().ravel()
            if (i + 1) % 100 == 0:
                print(f"    ps {i+1}/{len(gray)}", flush=True)
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
    nsd_ids = np.load(paths["stimuli"] / "shared1000_nsd_ids.npy")[idx]
    images = np.load(paths["stimuli"] / "shared1000_images.npy", mmap_mode="r")[idx]
    N = len(idx)

    print(f"{N} images — computing the tier-1 battery")
    t0 = time.time()

    print("  objects (crop-aware bboxes)")
    obj, boxes_all = instance_features(paths["meta"], nsd_ids)

    print("  scalars: spectral, clutter, shape, color")
    rows = []
    for i in range(N):
        im = np.asarray(images[i])
        g = im.mean(2)
        g01 = g / 255.0
        slope, intercept = fourier_stats(g01)
        ce, sc = weibull_contrast(g01)
        rect, oent, curv = orientation_stats(g01)
        buf = io.BytesIO()
        from PIL import Image as PILImage
        PILImage.fromarray(im).save(buf, "JPEG", quality=75)
        ow, bw = object_background_warmth(im, boxes_all[i])
        rows.append({
            "fourier_slope": slope, "fourier_intercept": intercept,
            "weibull_ce": ce, "weibull_sc": sc,
            "edge_density": float(edge_features(g[None]).mean()),
            "subband_entropy": subband_entropy(g01),
            "jpeg_kb": buf.tell() / 1024,
            "fc_proxy": feature_congestion_proxy(im),
            "rectilinearity": rect, "orientation_entropy": oent, "curvature_proxy": curv,
            "luminance": float(g.mean()), "contrast": float(g.std()),
            "object_warmth": ow, "background_warmth": bw,
        })
        if (i + 1) % 100 == 0:
            print(f"    scalars {i+1}/{N}", flush=True)
    scal = pd.concat([pd.DataFrame(rows), obj], axis=1)

    print("  gist (512 dims)")
    gray_all = np.asarray(images).mean(3)
    gist = gist_features(gray_all)

    print("  portilla-simoncelli (710 dims)")
    ps = ps_statistics(gray_all)

    hue = np.load(paths["cache"] / "color_features.npy")  # from notebook 06

    scal.to_csv(paths["cache"] / "feature_battery.csv", index=False)
    np.savez_compressed(paths["cache"] / "feature_battery.npz",
                        gist=gist, ps=ps, hue=hue, image_index=idx)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"scalars: {scal.shape[1]} features -> cache/feature_battery.csv")
    print("arrays : gist (512), ps (710), hue (10) -> cache/feature_battery.npz")
    print(scal.describe().loc[["mean", "std"]].T.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
