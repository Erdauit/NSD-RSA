#!/usr/bin/env python
"""S2 — extract and cache layer-wise model activations for the 1000 shared images.

Activations are computed once and written to HDF5. Nothing downstream ever recomputes
them, so RSA experiments are cheap to iterate on.

Rough sizes (1000 images, float32):
    dinov2_vitb14   12 blocks x {cls, patchmean} x 768   ~ 74 MB
    dinov2_vitl14   24 blocks x {cls, patchmean} x 1024  ~ 197 MB
    vit_b16         12 blocks x {cls, patchmean} x 768   ~ 74 MB
    clip_vitb16     12 blocks x {cls, patchmean} x 768   ~ 74 MB
    resnet50         4 stages x {gap} x (256..2048)      ~ 15 MB

Usage: make s2-activations
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.models import (  # noqa: E402
    ModelSpec,
    build_model,
    extract_activations,
    pick_device,
)


def cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / "activations" / f"{name}.h5"


def run_model(spec: ModelSpec, images: np.ndarray, cfg: dict, paths: dict, device) -> Path:
    import h5py

    out = cache_path(paths["cache"], spec.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with h5py.File(out, "r") as f:
            print(f"  cached: {out.name} ({len(f.keys())} layer readouts)")
        return out

    print(f"\n=== {spec.name} ({spec.supervision}) ===")
    print(f"  {spec.note}")
    t0 = time.time()
    model, transform, data_cfg = build_model(spec, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  timm: {spec.timm_name}")
    print(f"  input {data_cfg['input_size']}, {n_params/1e6:.0f}M params, loaded in {time.time()-t0:.0f}s")

    t0 = time.time()
    acts = extract_activations(
        model, spec, images, transform, device, batch_size=cfg.get("batch_size", 32)
    )
    dt = time.time() - t0

    total = sum(a.nbytes for a in acts.values())
    print(f"  {len(acts)} readouts, {total/1e6:.0f} MB, {dt:.0f}s "
          f"({len(images)/dt:.1f} img/s)")

    with h5py.File(out, "w") as f:
        for key, arr in sorted(acts.items()):
            f.create_dataset(key, data=arr.astype(np.float32), compression="lzf")
        f.attrs["model"] = spec.name
        f.attrs["timm_name"] = spec.timm_name
        f.attrs["kind"] = spec.kind
        f.attrs["supervision"] = spec.supervision
        f.attrs["n_images"] = len(images)

    del model
    print(f"  -> {out} ({out.stat().st_size/1e6:.0f} MB)")
    return out


def print_dimension_table(paths: dict, specs: list[ModelSpec]) -> None:
    """The S2 deliverable: a table of what we cached and at what dimensionality."""
    import h5py

    print("\n" + "=" * 74)
    print("ACTIVATION CACHE — dimensions")
    print("=" * 74)
    print(f"{'model':<16}{'supervision':<17}{'readouts':>9}{'depth':>7}{'dim range':>16}{'MB':>7}")
    for spec in specs:
        p = cache_path(paths["cache"], spec.name)
        if not p.exists():
            print(f"{spec.name:<16}{'(not extracted)':<17}")
            continue
        with h5py.File(p, "r") as f:
            keys = sorted(f.keys())
            dims = [f[k].shape[1] for k in keys]
            depth = len({k.split(".")[0] for k in keys})
        print(f"{spec.name:<16}{spec.supervision:<17}{len(keys):>9}{depth:>7}"
              f"{f'{min(dims)}-{max(dims)}':>16}{p.stat().st_size/1e6:>7.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="subset of registry names")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)

    stim = paths["stimuli"] / "shared1000_images.npy"
    if not stim.exists():
        print(f"Stimuli not found at {stim}\nRun: python scripts/s2_fetch_stimuli.py")
        return 1
    images = np.load(stim)
    if args.limit:
        images = images[: args.limit]
    print(f"stimuli: {images.shape} {images.dtype}")

    device = pick_device(cfg.get("device", "auto"))
    print(f"device: {device}")

    specs = [ModelSpec.from_config(d) for d in cfg["models"]]
    if args.models:
        specs = [s for s in specs if s.name in args.models]
        if not specs:
            print(f"No registry entries matched {args.models}")
            return 1

    for spec in specs:
        run_model(spec, images, cfg, paths, device)

    print_dimension_table(paths, specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
