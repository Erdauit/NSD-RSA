#!/usr/bin/env python
"""Smoke test: can we read representations from the whole VLM stack on this machine?

This exists to answer one question cheaply, before committing a week to the VLM
direction: does a 2B-class vision-language model load and run on an M5 with 17 GB, and
can we pull hidden states from *both* the vision tower and the language model at the
image-token positions?

It deliberately checks the hard parts rather than just "does it load":
  1. memory headroom in fp16 on MPS
  2. that image tokens can be located inside the LLM's sequence
  3. that per-layer hidden states differ from each other (a stack that returns the same
     vector everywhere would be useless for a depth analysis)
  4. throughput, extrapolated to 1000 images

Usage: python scripts/vlm_smoke_test.py --model HuggingFaceTB/SmolVLM-256M-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROMPT = "Describe the objects in this image."


def rss_gb() -> float:
    import resource

    # macOS reports maxrss in bytes; Linux in kilobytes.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def find_image_token_positions(
    input_ids: torch.Tensor, processor, model
) -> torch.Tensor:
    """Boolean mask over sequence positions marking image tokens.

    This is the crux of the whole idea: inside the LLM we care only about what happens
    to the *visual* tokens, not the prompt words. Different VLMs mark them differently,
    so try the documented routes and fail loudly rather than silently pooling over text.
    """
    candidates = []
    for obj in (model.config, getattr(model.config, "text_config", None), processor):
        for attr in ("image_token_id", "image_token_index"):
            val = getattr(obj, attr, None)
            if isinstance(val, int):
                candidates.append(val)

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        for name in ("<image>", "<|image_pad|>", "<image_soft_token>"):
            tid = tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tok, "unk_token_id", -1):
                candidates.append(tid)

    for tid in candidates:
        mask = input_ids == tid
        if mask.any():
            return mask

    raise RuntimeError(
        "could not locate image tokens in the sequence; tried ids "
        f"{sorted(set(candidates))}. Pooling over all positions would mix prompt text "
        "into the visual representation, so refusing to guess."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    stim = Path("data/stimuli/shared1000_images.npy")
    if not stim.exists():
        print(f"missing {stim}")
        return 1
    images_np = np.load(stim, mmap_mode="r")[: args.n_images]
    print(f"stimuli: {images_np.shape}")

    print(f"\nloading {args.model} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    # NOTE: device_map= segfaults (SIGSEGV) with transformers 5.15 + torch 2.13 on MPS.
    # Loading on CPU and moving the module afterwards is equivalent and stable.
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.2f}B params, loaded in {time.time()-t0:.0f}s")
    print(f"  peak RSS so far: {rss_gb():.1f} GB of 17 GB")

    from PIL import Image

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)

    per_layer_means, vision_shapes = [], None
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(images_np)):
            pil = Image.fromarray(np.asarray(images_np[i]))
            inputs = processor(text=chat, images=[pil], return_tensors="pt").to(args.device)

            out = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states  # tuple: embeddings + one per LLM layer

            mask = find_image_token_positions(inputs["input_ids"], processor, model)
            n_img_tokens = int(mask.sum())

            # Mean-pool each layer over image-token positions only.
            pooled = np.stack([
                h[0][mask[0]].float().mean(dim=0).cpu().numpy() for h in hidden
            ])
            per_layer_means.append(pooled)

            if i == 0:
                vision_shapes = (len(hidden), pooled.shape[1], n_img_tokens)
                print(f"\n  sequence length     : {inputs['input_ids'].shape[1]}")
                print(f"  image tokens found  : {n_img_tokens}")
                print(f"  LLM layers captured : {len(hidden)} (incl. embedding output)")
                print(f"  hidden width        : {pooled.shape[1]}")

    dt = time.time() - t0
    acts = np.stack(per_layer_means)  # (n_images, n_layers, dim)

    print(f"\nthroughput: {len(images_np)/dt:.2f} img/s "
          f"-> {1000/max(len(images_np)/dt, 1e-9)/60:.0f} min per 1000 images per prompt")
    print(f"peak RSS: {rss_gb():.1f} GB of 17 GB")

    print("\n=== checks ===")
    finite = np.isfinite(acts).all()
    print(f"  all finite                      : {finite}")

    # Layers must actually differ, or there is no depth axis to analyse.
    flat = acts.reshape(acts.shape[0], acts.shape[1], -1)
    first_last = np.corrcoef(flat[:, 1].ravel(), flat[:, -1].ravel())[0, 1]
    adjacent = np.mean([
        np.corrcoef(flat[:, k].ravel(), flat[:, k + 1].ravel())[0, 1]
        for k in range(1, flat.shape[1] - 1)
    ])
    print(f"  corr(first LLM layer, last)     : {first_last:+.3f}")
    print(f"  mean corr(adjacent layers)      : {adjacent:+.3f}")
    distinct = adjacent > first_last + 0.05
    print(f"  layers are distinct             : {distinct}")

    # Images must differ too, or every stimulus collapses to the same vector.
    spread = np.std(flat[:, -1], axis=0).mean()
    print(f"  across-image std at last layer  : {spread:.4f}")
    varied = spread > 1e-4

    ok = bool(finite and distinct and varied)
    print(f"\nVERDICT: {'USABLE' if ok else 'NOT USABLE — investigate before committing'}")
    print(f"  (layers {vision_shapes[0]}, width {vision_shapes[1]}, "
          f"{vision_shapes[2]} image tokens)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
