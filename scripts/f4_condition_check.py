#!/usr/bin/env python
"""F4 — mandatory pre-flight for every new VLM family: does the prompt reach the tokens?

Attention is causal, so image tokens can only attend to what precedes them. On SmolVLM we
learned this the expensive way: with the usual (image, then question) layout the
image-token readouts were bitwise identical across prompts, and four task conditions
would have produced four copies and a guaranteed null. Different chat templates lay the
sequence out differently, so this must be re-verified per family, not assumed.

For each model this runs a few images under two prompt conditions and reports, at every
LLM layer, the max |difference| of the image-token mean readout:

  prompt_position=before  ->  readouts MUST differ (the task can reach the tokens)
  prompt_position=after   ->  readouts MUST be bitwise identical (causal-attention sanity;
                              if they differ here, the template put text before the image
                              behind our back and "after" is not what it claims)

Usage:
    python scripts/f4_condition_check.py --config configs/vlm.yaml --models qwen2vl_2b llava_ov_05b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.vlm import PROMPTS, VLMSpec, build_vlm, extract_stack  # noqa: E402


def llm_imgmean_keys(acts: dict[str, np.ndarray]) -> list[str]:
    return sorted(k for k in acts if k.startswith("llm.") and k.endswith(".imgmean"))


def max_diff_by_layer(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    keys = llm_imgmean_keys(a)
    assert keys == llm_imgmean_keys(b), "readout sets differ between conditions"
    return {k: float(np.abs(a[k] - b[k]).max()) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--prompt", default="objects", choices=[k for k in PROMPTS if k != "none"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    device = cfg.get("device", "mps")
    tiling = bool(cfg.get("tiling", False))

    images = np.asarray(
        np.load(paths["stimuli"] / "shared1000_images.npy", mmap_mode="r")[: args.n_images]
    )
    prompt = PROMPTS[args.prompt]
    print(f"{args.n_images} images, condition pair: none vs {args.prompt!r}, fp32, "
          f"tiling={tiling}\n")

    all_ok = True
    for name in args.models:
        spec = next(VLMSpec.from_config(d) for d in cfg["models"] if d["name"] == name)
        print(f"=== {name} ({spec.hf_name}) ===")
        model, processor = build_vlm(spec, device, tiling=tiling)

        verdicts = {}
        for position, want_effect in (("before", True), ("after", False)):
            base = extract_stack(model, processor, images, "", device,
                                 progress_every=0, prompt_position=position)
            task = extract_stack(model, processor, images, prompt, device,
                                 progress_every=0, prompt_position=position)
            diffs = max_diff_by_layer(base, task)
            worst = max(diffs.values())
            mid = list(diffs.values())[len(diffs) // 2]
            n_img_tokens = None  # informational only; mask size varies per model
            if want_effect:
                ok = worst > 0.0
                verdicts["before"] = ok
                print(f"  before: max|diff| over layers = {worst:.6f} "
                      f"(mid-depth {mid:.6f}) -> {'OK, prompt reaches tokens' if ok else 'FAIL: identical'}")
            else:
                ok = worst == 0.0
                verdicts["after"] = ok
                print(f"  after : max|diff| over layers = {worst:.6f} "
                      f"-> {'OK, bitwise identical as causality demands' if ok else 'FAIL: template reordered content'}")
            del base, task
        del model, processor

        model_ok = all(verdicts.values())
        all_ok &= model_ok
        print(f"  VERDICT: {'USABLE for the task design' if model_ok else 'NOT USABLE — fix before extracting'}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
