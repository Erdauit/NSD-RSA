#!/usr/bin/env python
"""F8 — tier-2 norms: THINGSplus concept properties and ResMem memorability.

THINGSplus (Stoinski et al. 2023): concept-level property ratings (1-7 scale) mapped
onto the 80 COCO categories by name (direct match + a documented synonym table;
`person` is averaged over THINGS `man`/`woman`). Per-image value = mean over the
categories present in the crop. THINGSplus has no direct real-world-size norm in the
property table, so `heavy` serves as the size proxy and is labeled as such.

Memorability: ResMem (Needell & Bainbridge 2022), predicted for each of the 515 images.

Output: cache/tier2_norms.csv

Usage: python scripts/f8_tier2_norms.py --config configs/vlm.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402

# COCO name -> THINGS Word where names differ. Every entry checked against the
# THINGSplus word list; person handled separately.
SYNONYMS = {
    "bicycle": "bike", "stop sign": "road sign", "handbag": "purse", "skis": "ski",
    "sports ball": "ball", "tennis racket": "racket", "wine glass": "wineglass",
    "hot dog": "hotdog", "dining table": "table", "tv": "television",
    "remote": "remote control", "cell phone": "cellphone", "hair drier": "hairdryer",
    "potted plant": "plant",
}
PROPS = ["lives", "manmade", "natural", "moves", "heavy", "grasp", "pleasant", "arousal"]


def things_category_norms(meta_dir: Path, coco_names: list[str]) -> pd.DataFrame:
    P = pd.read_csv(meta_dir / "thingsplus" / "property-ratings.tsv", sep="\t", index_col=0)
    P["word_l"] = P["Word"].str.lower()
    by_word = P.groupby("word_l")[[f"property_{p}_mean" for p in PROPS]].mean()

    rows, unmapped = {}, []
    for name in coco_names:
        key = SYNONYMS.get(name, name)
        if name == "person":
            sub = by_word.loc[[w for w in ("man", "woman") if w in by_word.index]]
            rows[name] = sub.mean()
        elif key in by_word.index:
            rows[name] = by_word.loc[key]
        else:
            unmapped.append(name)
    if unmapped:
        print(f"  UNMAPPED categories (excluded from norms): {unmapped}")
    out = pd.DataFrame(rows).T
    out.columns = PROPS
    return out


def resmem_scores(images: np.ndarray, device: str) -> np.ndarray:
    import torch
    from PIL import Image
    from resmem import ResMem, transformer

    model = ResMem(pretrained=True).eval().to(device)
    out = np.empty(len(images))
    with torch.no_grad():
        for i in range(len(images)):
            x = transformer(Image.fromarray(np.asarray(images[i]))).unsqueeze(0).to(device)
            out[i] = float(model(x).cpu())
            if (i + 1) % 100 == 0:
                print(f"    resmem {i+1}/{len(images)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    device = cfg.get("device", "cpu")

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    idx = usable_images(cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"])
    images = np.load(paths["stimuli"] / "shared1000_images.npy", mmap_mode="r")[idx]

    import json
    blob = json.loads((paths["meta"] / "coco" / "annotations" / "instances_val2017.json")
                      .read_text(encoding="utf-8"))
    cat_ids = sorted(c["id"] for c in blob["categories"])
    names = [ {c["id"]: c["name"] for c in blob["categories"]}[cid] for cid in cat_ids ]

    print("THINGSplus category norms")
    norms = things_category_norms(paths["meta"], names)
    print(f"  mapped {len(norms)}/{len(names)} categories; scale 1-7")

    hot = np.load(paths["cache"] / "coco_categories.npz")["multihot"]
    mapped_cols = [j for j, n in enumerate(names) if n in norms.index]
    M = norms.loc[[names[j] for j in mapped_cols]].values          # (n_mapped, props)
    H = hot[:, mapped_cols]                                        # (N, n_mapped)
    denom = H.sum(1, keepdims=True)
    per_image = np.where(denom > 0, H @ M / np.maximum(denom, 1), np.nan)
    T = pd.DataFrame(per_image, columns=[f"things_{p}" for p in PROPS])
    T.loc[:, "things_heavy_note"] = "size proxy"
    print(f"  images with no mapped category: {int((denom == 0).sum())}")

    print("ResMem memorability")
    T["memorability"] = resmem_scores(images, device)

    out = paths["cache"] / "tier2_norms.csv"
    T.drop(columns=["things_heavy_note"]).to_csv(out, index=False)
    print(f"\n-> {out}")
    print(T.drop(columns=["things_heavy_note"]).describe().loc[["mean", "std", "min", "max"]].T.round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
