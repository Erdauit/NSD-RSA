#!/usr/bin/env python
"""Does reduced-precision inference preserve representational geometry?

Running large models in float16 or bfloat16 is the default habit, and for generation it
is harmless — the sampled text barely changes. RSA is a different question. We are not
reading the model's output, we are measuring the *geometry* of its hidden states, and
comparing that geometry between models whose scores differ by a few hundredths.

So the question is not "does the model still work in bf16" but "does bf16 move the RDM by
less than the effects we are trying to measure". That has to be measured, not assumed.

Measured on SmolVLM-2.2B, 60 NSD images, image tokens mean-pooled per layer:

    control: fp32 vs fp32   RDM corr 1.000000  (bitwise identical — MPS is deterministic,
                                                so any disagreement below is real)
    float16  vs fp32        RDM corr 0.934 - 0.969
    bfloat16 vs fp32        RDM corr 0.955 - 0.976

A 0.95 agreement sounds high until you compare it to the effects in play: the gap between
two models' normalised RSA scores in ventral cortex was 0.04. Precision noise of this size
could manufacture or erase such a gap. Hence the pipeline runs in float32.

Usage: python scripts/vlm_precision_check.py --model HuggingFaceTB/SmolVLM-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa.rdm import compare_rdms, compute_rdm  # noqa: E402

PROMPT = "Describe the objects in this image."
# Below this, precision noise is comparable to the between-model effects we report.
MIN_SAFE_AGREEMENT = 0.999


def extract(model_name: str, dtype, device: str, images: np.ndarray, tiling: bool):
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    processor.image_processor.do_image_splitting = tiling
    # device_map= segfaults on MPS with transformers 5.15 + torch 2.13; move afterwards.
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype)
    model.eval()
    model = model.to(device)

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_token_id = model.config.image_token_id

    acts = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(images)):
            inputs = processor(
                text=chat, images=[Image.fromarray(np.asarray(images[i]))], return_tensors="pt"
            ).to(device)
            hidden = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
            mask = (inputs["input_ids"] == image_token_id)[0]
            acts.append(np.stack([h[0][mask].float().mean(0).cpu().numpy() for h in hidden]))
    elapsed = time.time() - t0

    del model
    if device == "mps":
        torch.mps.empty_cache()
    return np.stack(acts), elapsed


def agreement(a: np.ndarray, b: np.ndarray) -> list[tuple[int, float, float]]:
    n_layers = a.shape[1]
    probes = sorted({1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    out = []
    for layer in probes:
        r = compare_rdms(
            compute_rdm(a[:, layer].astype(np.float64)),
            compute_rdm(b[:, layer].astype(np.float64)),
        )
        out.append((layer, r, float(np.abs(a[:, layer] - b[:, layer]).max())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tiling", action="store_true", help="keep the processor's image splitting")
    args = ap.parse_args()

    images = np.load("data/stimuli/shared1000_images.npy", mmap_mode="r")[: args.n_images]
    print(f"{args.model}, {len(images)} images, tiling={args.tiling}\n")

    reference, t_ref = extract(args.model, torch.float32, args.device, images, args.tiling)
    print(f"float32   {t_ref/len(images):5.2f} s/img")

    # Control first. If float32 does not reproduce itself, nothing below means anything.
    control, _ = extract(args.model, torch.float32, args.device, images, args.tiling)
    identical = np.array_equal(reference, control)
    print(f"\ncontrol — float32 run twice: bitwise identical = {identical}")
    if not identical:
        print("  Computation is non-deterministic; precision comparisons are meaningless.")
        for layer, r, d in agreement(reference, control):
            print(f"    layer {layer:>3}: RDM corr {r:.6f}  max|diff| {d:.5f}")
        return 1

    worst = 1.0
    for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        acts, elapsed = extract(args.model, dtype, args.device, images, args.tiling)
        print(f"\n{name}  {elapsed/len(images):5.2f} s/img  ({t_ref/elapsed:.1f}x faster)")
        print(f"  {'layer':>6}  {'RDM corr vs fp32':>18}  {'max|diff|':>10}")
        for layer, r, d in agreement(acts, reference):
            print(f"  {layer:>6}  {r:>18.6f}  {d:>10.5f}")
            worst = min(worst, r)

    print(f"\nworst agreement across probed layers: {worst:.4f}")
    if worst >= MIN_SAFE_AGREEMENT:
        print("VERDICT: reduced precision is safe here.")
        return 0
    print(
        f"VERDICT: NOT safe. Reduced precision moves the RDM by more than "
        f"{1 - worst:.1%}, comparable to the between-model differences this project "
        "reports. Run the analysis in float32."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
