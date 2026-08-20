#!/usr/bin/env python
"""B8 — degradation or retargeting? What the language stack keeps, layer by layer.

Liu et al. (2025) show LLMs preserve less visual information than the encoder supplies,
inviting a deflationary reading of our early-cortex decline: maybe the language stack
just *loses* visual information wholesale. The RSA answer is already suggestive — a rise
in lateral alignment cannot come from pure loss — but this makes the control direct:
linearly decode image properties from every LLM layer's image-token readout.

  low-level targets : mean luminance, RMS contrast (ridge R², 5-fold CV),
                      Gabor-energy PCs (mean R² over the top 10 components)
  high-level target : COCO object categories in the crop (ridge scores -> ROC-AUC,
                      mean over categories present in >= 5% of images)

Pure degradation predicts everything falls together. Retargeting predicts the low-level
targets fall while the high-level target holds or rises.

Usage: python scripts/f5_b8_decodability.py --config configs/vlm.yaml
  (needs cache/coco_categories.npz from f5_b9_categories.py)
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
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.lowlevel import gabor_features  # noqa: E402

MODELS = ("smolvlm_2b", "qwen2vl_2b")
POOL = "trim"
ALPHAS = np.logspace(1, 5, 9)


def oof_predict(X: np.ndarray, Y: np.ndarray, seed: int) -> np.ndarray:
    preds = np.empty_like(Y, dtype=np.float64)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        m = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        m.fit((X[tr] - mu) / sd, Y[tr])
        preds[te] = m.predict((X[te] - mu) / sd).reshape(len(te), -1)
    return preds


def r2_cols(Y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    ss_res = ((Y - preds) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    paths = resolve_paths(cfg)

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    idx = usable_images(cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"])
    im = np.load(paths["stimuli"] / "shared1000_images.npy", mmap_mode="r")[idx].astype(np.float32)
    gray = im.mean(axis=3)

    # ---- targets ----
    lum = gray.mean(axis=(1, 2))[:, None]
    con = gray.std(axis=(1, 2))[:, None]

    gab_cache = paths["cache"] / "gabor_features.npy"
    if gab_cache.exists():
        gab = np.load(gab_cache)
    else:
        print("computing Gabor features (cached afterwards)")
        gab = gabor_features(gray)
        np.save(gab_cache, gab)
    gz = (gab - gab.mean(0)) / (gab.std(0) + 1e-8)
    _, _, vt = np.linalg.svd(gz, full_matrices=False)
    gab_pcs = gz @ vt[:10].T

    cats = np.load(paths["cache"] / "coco_categories.npz")["multihot"]
    common = cats.mean(0) >= 0.03  # >= ~15 positive images: enough for a stable AUC
    cats = cats[:, common]
    print(f"targets: luminance, contrast, 10 Gabor PCs, {cats.shape[1]} common categories")

    results: dict = {}
    for model in args.models:
        acts = np.load(paths["cache"] / f"f1_readouts_{model}.npz")
        llm = sorted((k for k in acts.files if k.startswith("llm.") and k.endswith(f".{POOL}")),
                     key=lambda k: int(k.split(".")[1]))
        prof: dict[str, list[float]] = {"luminance": [], "contrast": [], "gabor": [],
                                        "categories_auc": []}
        for k in llm:
            X = acts[k].astype(np.float64)
            prof["luminance"].append(float(r2_cols(lum, oof_predict(X, lum, seed))[0]))
            prof["contrast"].append(float(r2_cols(con, oof_predict(X, con, seed))[0]))
            prof["gabor"].append(float(np.nanmean(r2_cols(gab_pcs, oof_predict(X, gab_pcs, seed)))))
            scores = oof_predict(X, cats, seed)
            aucs = [roc_auc_score(cats[:, j], scores[:, j]) for j in range(cats.shape[1])]
            prof["categories_auc"].append(float(np.mean(aucs)))
            print(f"  {model} {k}: lum {prof['luminance'][-1]:.3f}  "
                  f"gabor {prof['gabor'][-1]:.3f}  catAUC {prof['categories_auc'][-1]:.3f}",
                  flush=True)
        results[model] = {"layers": llm, **prof}

    (paths["cache"] / "f5_b8_decodability.json").write_text(json.dumps(results, indent=1))

    fig, axes = plt.subplots(1, len(args.models), figsize=(5.4 * len(args.models), 3.8),
                             squeeze=False)
    for ax, model in zip(axes[0], args.models, strict=False):
        r = results[model]
        x = np.arange(len(r["layers"]))
        ax2 = ax.twinx()
        ax.plot(x, r["luminance"], label="luminance R²", color="#2a6f97")
        ax.plot(x, r["contrast"], label="contrast R²", color="#61a5c2")
        ax.plot(x, r["gabor"], label="Gabor PCs R²", color="#89c2d9")
        ax2.plot(x, r["categories_auc"], label="categories AUC", color="#c1121f", lw=2)
        ax.set_xlabel("LLM layer")
        ax.set_ylabel("R² (low-level)")
        ax2.set_ylabel("AUC (categories)", color="#c1121f")
        ax.set_title(model, fontsize=10)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7)
    fig.suptitle("What the language stack keeps: low-level structure vs object content",
                 fontsize=11)
    fig.tight_layout()
    out = paths["figures"] / "f5_b8_decodability.png"
    fig.savefig(out, dpi=170)
    print(f"\nfigure -> {out}\nresults -> cache/f5_b8_decodability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
