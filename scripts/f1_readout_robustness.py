#!/usr/bin/env python
"""F1 — do the headline numbers survive a change of token pooling?

The diagnostic (scripts/f1_token_norms.py) established that SmolVLM's vision tower puts
up to 15% of all token norm into 1% of tokens at mid-depth, and that the outlier share
tracks alignment almost deterministically: r = -0.73 with early cortex, -0.78 with
ventral, +0.88 with lateral, replicated across both model scales.

That last number is the problem. Our largest reported effect — lateral alignment peaking
at 0.445 — sits at the outlier-peak layer. If it is carried by the scratchpad tokens
rather than by the visual representation, it is an artefact.

So this recomputes the same quantities under three poolings and prints them side by side:
  mean    the original readout
  trim    mean after dropping the top 1% of tokens by norm
  median  elementwise median across tokens

Reported per pooling: the vision-tower profile against each ROI, and the slope of
alignment across LLM layers with per-subject signs. Nothing is concluded here; the table
is the deliverable.

Usage: make f1-robustness
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import wilcoxon  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import average_repeats, common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.models import pool_tokens  # noqa: E402
from nsd_rsa.noise_ceiling import normalise_to_ceiling  # noqa: E402
from nsd_rsa.rdm import compare_rdms, compute_rdm  # noqa: E402
from nsd_rsa.vlm import VLMSpec, build_vlm, image_token_mask, locate_stack  # noqa: E402

POOLINGS = ("mean", "trim", "median")
ROIS = ("early", "midventral", "ventral", "lateral")


@torch.no_grad()
def extract_three_poolings(spec: VLMSpec, images: np.ndarray, device: str) -> dict:
    """Vision blocks and LLM image-token positions, under all three poolings."""
    from PIL import Image

    model, processor = build_vlm(spec, device, tiling=False)
    stack = locate_stack(model)

    captured: dict[str, torch.Tensor] = {}
    handles = [
        block.register_forward_hook(
            lambda _m, _i, out, k=f"vision.{i:02d}": captured.__setitem__(
                k, (out[0] if isinstance(out, tuple) else out).detach()
            )
        )
        for i, block in enumerate(stack["vision_blocks"])
    ]

    chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=True
    )

    acc: dict[str, list[np.ndarray]] = {}
    t0 = time.time()
    try:
        for n, im in enumerate(images):
            inputs = processor(
                text=chat, images=[Image.fromarray(np.asarray(im))], return_tensors="pt"
            ).to(device)
            hidden = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
            mask = image_token_mask(inputs["input_ids"], model, processor)[0]

            for key, tensor in captured.items():
                for pool, vec in pool_tokens(tensor.reshape(1, -1, tensor.shape[-1])).items():
                    acc.setdefault(f"{key}.{pool}", []).append(vec[0].cpu().numpy())
            captured.clear()

            for layer, h in enumerate(hidden):
                toks = h[0][mask].unsqueeze(0)
                for pool, vec in pool_tokens(toks).items():
                    acc.setdefault(f"llm.{layer:02d}.{pool}", []).append(vec[0].cpu().numpy())

            if (n + 1) % 100 == 0:
                print(f"      {n+1}/{len(images)}  ({(time.time()-t0)/(n+1):.2f} s/img)", flush=True)
    finally:
        for h in handles:
            h.remove()
        del model

    return {k: np.stack(v).astype(np.float32) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--models", nargs="*", default=["smolvlm_256m", "smolvlm_500m"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    device = cfg.get("device", "mps")

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    idx = usable_images(cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"])
    images = np.asarray(np.load(paths["stimuli"] / "shared1000_images.npy", mmap_mode="r")[idx])
    valid = common_valid_vertices(paths["betas"], paths["cache"] / "valid_vertices.npy")
    ceilings = {k: v[0] for k, v in
                json.loads((paths["cache"] / "rsa_results.json").read_text())["ceilings"].items()}

    print("building brain RDMs")
    brain: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(paths["betas"].glob("*_shared_betas.h5")):
        data = load_subject(path)
        for roi in ROIS:
            patterns, _ = average_repeats(data, images=idx, roi=roi, valid=valid)
            brain.setdefault(roi, {})[data.subject] = compute_rdm(patterns.astype(np.float64))
        del data
    subjects = sorted(brain[ROIS[0]])
    print(f"  {len(subjects)} subjects, {len(idx)} images\n")

    results: dict = {}
    for name in args.models:
        spec = next(VLMSpec.from_config(d) for d in cfg["models"] if d["name"] == name)
        print(f"=== {name}: extracting three poolings ===", flush=True)
        acts = extract_three_poolings(spec, images, device)

        cache = paths["cache"] / f"f1_readouts_{name}.npz"
        np.savez_compressed(cache, **acts, image_index=idx)
        print(f"  cached -> {cache.name}")

        rdms = {k: compute_rdm(v.astype(np.float64)) for k, v in acts.items()}
        results[name] = {}

        for pool in POOLINGS:
            vision = sorted(k for k in rdms if k.startswith("vision.") and k.endswith(f".{pool}"))
            llm = sorted((k for k in rdms if k.startswith("llm.") and k.endswith(f".{pool}")),
                         key=lambda k: int(k.split(".")[1]))
            entry: dict = {"vision": {}, "llm_slope": {}}

            for roi in ROIS:
                prof = np.array([
                    normalise_to_ceiling(
                        float(np.mean([compare_rdms(rdms[k], brain[roi][s]) for s in subjects])),
                        ceilings[roi],
                    ) for k in vision
                ])
                entry["vision"][roi] = {
                    "peak": float(np.nanmax(prof)),
                    "at": vision[int(np.nanargmax(prof))].split(".")[1],
                    "min": float(np.nanmin(prof)),
                }

                slopes = []
                for s in subjects:
                    y = np.array([
                        normalise_to_ceiling(compare_rdms(rdms[k], brain[roi][s]), ceilings[roi])
                        for k in llm
                    ])
                    slopes.append(np.polyfit(np.arange(len(y)), y, 1)[0])
                slopes = np.array(slopes)
                entry["llm_slope"][roi] = {
                    "slope": float(slopes.mean() * 100),
                    "n_positive": int((slopes > 0).sum()),
                    "p": float(wilcoxon(slopes).pvalue),
                }
            results[name][pool] = entry

    out = paths["cache"] / "f1_readout_robustness.json"
    # Merge, don't clobber: partial runs (--models X) must not erase other models' rows.
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(results)
    results = merged
    out.write_text(json.dumps(results, indent=1))

    print("\n" + "=" * 88)
    print("VISION TOWER — peak alignment, and the minimum (the collapse)")
    print("=" * 88)
    print(f"{'model':<15}{'ROI':<12}" + "".join(f"{p:>22}" for p in POOLINGS))
    for name, by_pool in results.items():
        for roi in ROIS:
            cells = "".join(
                f"{by_pool[p]['vision'][roi]['peak']:>9.3f} @{by_pool[p]['vision'][roi]['at']:<4}"
                f"{by_pool[p]['vision'][roi]['min']:>8.3f}"
                for p in POOLINGS
            )
            print(f"{name:<15}{roi:<12}{cells}")
    print("  (each cell: peak @ block, then the minimum over blocks)")

    print("\n" + "=" * 88)
    print("LLM SLOPE — x100, and how many of 8 subjects show a rise")
    print("=" * 88)
    print(f"{'model':<15}{'ROI':<12}" + "".join(f"{p:>20}" for p in POOLINGS))
    for name, by_pool in results.items():
        for roi in ROIS:
            cells = "".join(
                f"{by_pool[p]['llm_slope'][roi]['slope']:>11.3f}"
                f"{by_pool[p]['llm_slope'][roi]['n_positive']:>6}/8"
                for p in POOLINGS
            )
            print(f"{name:<15}{roi:<12}{cells}")

    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
