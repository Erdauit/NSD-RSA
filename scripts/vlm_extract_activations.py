#!/usr/bin/env python
"""Extract activations from the whole VLM stack, for every prompt condition.

Produces one cache per (model, prompt), so a run can be interrupted and resumed — the
2.2B model alone takes hours on an M5.

Readout naming gives a single continuous depth axis:
    vision.00 .. vision.NN   the vision encoder's blocks
    projector                what actually enters the language model
    llm.KK.imgmean           image-token positions at LLM layer KK
    llm.KK.lasttoken         the final position, which the model reads out to answer

Usage: make vlm-extract
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.vlm import PROMPTS, VLMSpec, build_vlm, extract_stack  # noqa: E402


def cache_path(cache_dir: Path, model: str, prompt: str, position: str = "before") -> Path:
    return cache_dir / "vlm_activations" / f"{model}__{prompt}__{position}.h5"


def already_done(path: Path, n_images: int) -> bool:
    """A short cache from an interrupted or --limit run must not pass as complete."""
    if not path.exists():
        return False
    import h5py

    try:
        with h5py.File(path, "r") as f:
            return int(f.attrs.get("n_images", -1)) == n_images
    except OSError:
        return False  # truncated file from a kill mid-write


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    device = cfg.get("device", "mps")
    tiling = bool(cfg.get("tiling", False))
    position = cfg.get("prompt_position", "before")

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    image_idx = usable_images(
        cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"]
    )
    all_images = np.load(paths["stimuli"] / "shared1000_images.npy", mmap_mode="r")
    images = np.asarray(all_images[image_idx])
    if args.limit:
        images = images[: args.limit]

    specs = [VLMSpec.from_config(d) for d in cfg["models"]]
    if args.models:
        specs = [s for s in specs if s.name in args.models]
    prompt_names = args.prompts or cfg["prompts"]

    print(f"stimuli : {images.shape} (analysis subset of the shared1000)")
    print(f"device  : {device}, float32, tiling={tiling}, prompt_position={position}")
    print(f"models  : {[s.name for s in specs]}")
    print(f"prompts : {prompt_names}")

    out_dir = paths["cache"] / "vlm_activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    import h5py

    grand_start = time.time()
    for spec in specs:
        todo = [p for p in prompt_names
                if not already_done(cache_path(paths["cache"], spec.name, p, position), len(images))]
        if not todo:
            print(f"\n=== {spec.name}: all prompts cached ===")
            continue

        print(f"\n=== {spec.name} ===")
        print(f"  {spec.note}")
        t0 = time.time()
        model, processor = build_vlm(spec, device, tiling=tiling)
        print(f"  {spec.n_params:.2f}B params, loaded in {time.time()-t0:.0f}s")

        for prompt_name in todo:
            prompt = PROMPTS[prompt_name]
            label = f'"{prompt}"' if prompt else "(no text — control)"
            print(f"\n  prompt [{prompt_name}] {label}")

            t0 = time.time()
            acts = extract_stack(
                model, processor, images, prompt, device, prompt_position=position
            )
            dt = time.time() - t0

            total_mb = sum(a.nbytes for a in acts.values()) / 1e6
            print(f"    {len(acts)} readouts, {total_mb:.0f} MB, {dt/60:.1f} min "
                  f"({dt/len(images):.2f} s/img)")

            out = cache_path(paths["cache"], spec.name, prompt_name, position)
            tmp = out.with_suffix(".h5.part")
            with h5py.File(tmp, "w") as f:
                for key, arr in sorted(acts.items()):
                    f.create_dataset(key, data=arr, compression="lzf")
                f.attrs["model"] = spec.name
                f.attrs["hf_name"] = spec.hf_name
                f.attrs["prompt_name"] = prompt_name
                f.attrs["prompt"] = prompt
                f.attrs["prompt_position"] = position
                f.attrs["n_images"] = len(images)
                f.attrs["dtype"] = "float32"
                f.attrs["tiling"] = tiling
            tmp.replace(out)  # atomic: a kill mid-write cannot leave a valid-looking cache
            print(f"    -> {out.name}")

        del model

    print(f"\ntotal wall clock: {(time.time()-grand_start)/60:.0f} min")

    print("\n" + "=" * 78)
    print("VLM ACTIVATION CACHE")
    print("=" * 78)
    print(f"{'model':<16}{'prompt':<10}{'readouts':>9}{'vision':>8}{'llm':>6}{'MB':>8}")
    for spec in specs:
        for prompt_name in prompt_names:
            p = cache_path(paths["cache"], spec.name, prompt_name, position)
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                keys = list(f.keys())
                n_vision = len([k for k in keys if k.startswith("vision.")])
                n_llm = len({k.split(".")[1] for k in keys if k.startswith("llm.")})
            print(f"{spec.name:<16}{prompt_name:<10}{len(keys):>9}{n_vision:>8}{n_llm:>6}"
                  f"{p.stat().st_size/1e6:>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
