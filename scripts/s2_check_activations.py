#!/usr/bin/env python
"""S2 — sanity checks on the cached activations.

Same discipline as S1: before using these representations to make claims about the brain,
confirm they are well-formed and reproduce something already known about deep networks.

The known fact we target: **representations change gradually with depth.** Adjacent
blocks of a trained transformer should produce similar RDMs, and blocks far apart should
not. A network whose layer 1 and layer 12 looked alike would either be untrained or
badly hooked; either way its brain alignment would be meaningless.

Produces the layer x layer RDM-similarity heatmap, which doubles as a paper figure.

Usage: make s2-check
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
from nsd_rsa.models import ModelSpec  # noqa: E402
from nsd_rsa.rdm import compare_rdms, compute_rdm  # noqa: E402

MIN_DEPTH_GRADIENT = 0.15  # adjacent-minus-distant similarity we require


def cache_file(paths: dict, spec: ModelSpec) -> Path:
    return paths["cache"] / "activations" / f"{spec.name}_{spec.input_size}.h5"


def check_wellformed(path: Path) -> dict:
    """No NaNs, no constant readouts, sensible scale."""
    import h5py

    problems, stats = [], {}
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        for k in keys:
            a = f[k][:]
            if not np.isfinite(a).all():
                problems.append(f"{k}: non-finite values")
            elif a.std() == 0:
                problems.append(f"{k}: constant — the hook probably captured nothing")
            stats[k] = float(a.std())
        n_images = int(f.attrs.get("n_images", -1))
    return {"keys": keys, "problems": problems, "stds": stats, "n_images": n_images}


def depth_profile(path: Path, pool: str) -> tuple[list[str], np.ndarray]:
    """Layer x layer RDM similarity for one pooling readout."""
    import h5py

    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if k.endswith(f".{pool}"))
        if not keys:
            return [], np.zeros((0, 0))
        rdms = [compute_rdm(f[k][:].astype(np.float64)) for k in keys]

    n = len(rdms)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = compare_rdms(rdms[i], rdms[j])
    return [k.split(".")[0] for k in keys], sim


def gradient_score(sim: np.ndarray) -> tuple[float, float]:
    """Mean similarity of adjacent layers vs layers at least half the depth apart."""
    n = len(sim)
    if n < 4:
        return float("nan"), float("nan")
    adjacent = np.mean([sim[i, i + 1] for i in range(n - 1)])
    far = max(2, n // 2)
    distant = np.mean([sim[i, j] for i in range(n) for j in range(n) if abs(i - j) >= far])
    return float(adjacent), float(distant)


def figure_depth_similarity(results: list[dict], fig_dir: Path) -> Path:
    usable = [r for r in results if r["sim"].size > 0]
    fig, axes = plt.subplots(1, len(usable), figsize=(3.4 * len(usable), 3.6), squeeze=False)
    for ax, res in zip(axes[0], usable, strict=False):
        im = ax.imshow(res["sim"], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{res['name']}\n({res['supervision']})", fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046, label="RDM similarity")
    fig.suptitle("Representations change gradually with depth (readout: CLS / GAP)", fontsize=10)
    fig.tight_layout()
    out = fig_dir / "s2_layer_similarity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    specs = [ModelSpec.from_config(d) for d in cfg["models"]]

    results, all_ok = [], True
    for spec in specs:
        path = cache_file(paths, spec)
        if not path.exists():
            print(f"{spec.name}: not extracted, skipping")
            continue

        info = check_wellformed(path)
        pool = "cls" if spec.kind == "vit" else "gap"
        layers, sim = depth_profile(path, pool)
        adjacent, distant = gradient_score(sim)

        print(f"\n=== {spec.name} ({spec.supervision}, {spec.input_size}px) ===")
        print(f"  {len(info['keys'])} readouts over {info['n_images']} images")
        if info["problems"]:
            all_ok = False
            for p in info["problems"]:
                print(f"  PROBLEM: {p}")
        else:
            print("  well-formed: no non-finite values, no constant readouts")
        stds = np.array(list(info["stds"].values()))
        print(f"  activation std across readouts: {stds.min():.3f} .. {stds.max():.3f}")

        if np.isfinite(adjacent):
            gap = adjacent - distant
            ok = gap >= MIN_DEPTH_GRADIENT
            all_ok &= ok
            print(f"  RDM similarity, adjacent layers : {adjacent:.3f}")
            print(f"  RDM similarity, distant layers  : {distant:.3f}")
            print(f"  depth gradient {gap:+.3f} -> {'PASS' if ok else 'FAIL'} "
                  f"(need >= {MIN_DEPTH_GRADIENT})")

        results.append({
            "name": spec.name, "supervision": spec.supervision,
            "sim": sim, "layers": layers,
        })

    if results:
        out = figure_depth_similarity(results, paths["figures"])
        print(f"\nFigure: {out}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL — investigate before using these'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
