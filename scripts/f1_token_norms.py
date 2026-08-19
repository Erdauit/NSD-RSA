#!/usr/bin/env python
"""F1 diagnostic — do high-norm outlier tokens explain the mid-encoder RSA collapse?

Trained ViTs develop a handful of patch tokens whose L2 norm is one to two orders of
magnitude above the rest (Darcet et al., "Vision Transformers Need Registers", ICLR 2024).
They appear around the middle of the network, in low-information image regions, and carry
global scratch information rather than anything about their own patch.

Why that matters here: our readout is a mean over tokens, and a mean is dominated by the
largest terms. With 256 tokens of which three carry 50x the norm, over a third of the
pooled vector is contributed by tokens that no longer describe the picture. The resulting
RDM would reflect the model's scratchpad rather than its visual representation — which is
one candidate explanation for the negative RSA we measured in mid-encoder blocks.

This script only measures. It reports, per layer:
    median token norm, the ratio of the largest norm to the median, and the share of all
    norm mass carried by the top 1% of tokens.

The prediction under test: the layers where RSA collapses are the layers where the top-1%
share spikes. A flat norm profile would falsify the hypothesis and send us looking
elsewhere.

Usage: make f1-norms
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.models import ModelSpec, build_model, enumerate_layers, pick_device  # noqa: E402

TOP_FRACTION = 0.01


def tokens_from(output: torch.Tensor, kind: str, n_prefix: int) -> torch.Tensor:
    """Reduce a layer output to (batch, tokens, features).

    For a ViT we drop the prefix tokens (CLS and any registers): they are supposed to be
    global, so counting them as outliers would confirm the hypothesis by construction.
    For a CNN the spatial positions play the role of tokens.
    """
    if kind == "vit":
        return output[:, n_prefix:, :]
    return output.flatten(2).transpose(1, 2)


def norm_stats(tokens: torch.Tensor) -> dict[str, float]:
    """Per-image statistics of token norms, averaged over the batch."""
    norms = tokens.float().norm(dim=-1)                       # (batch, tokens)
    n_top = max(1, int(round(norms.shape[1] * TOP_FRACTION)))
    top = norms.topk(n_top, dim=1).values

    median = norms.median(dim=1).values
    total = norms.sum(dim=1)
    return {
        "median_norm": float(median.mean()),
        "max_over_median": float((norms.max(dim=1).values / median.clamp_min(1e-9)).mean()),
        "top1pct_share": float((top.sum(dim=1) / total.clamp_min(1e-9)).mean()),
        "n_tokens": int(norms.shape[1]),
        "n_top": n_top,
    }


@torch.no_grad()
def profile_model(spec: ModelSpec, images: np.ndarray, device, batch_size: int = 8) -> dict:
    from PIL import Image

    model, transform, _ = build_model(spec, device)
    layers = enumerate_layers(model, spec)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1)) if spec.kind == "vit" else 0

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name):
        def fn(_m, _i, out):
            captured[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    for name, module in layers:
        handles.append(module.register_forward_hook(hook(name)))

    per_layer: dict[str, list[dict]] = {}
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            x = torch.stack([transform(Image.fromarray(im)) for im in batch]).to(device)
            model(x)
            for name, _ in layers:
                toks = tokens_from(captured[name], spec.kind, n_prefix)
                per_layer.setdefault(name, []).append(norm_stats(toks))
            captured.clear()
    finally:
        for h in handles:
            h.remove()
    del model

    out = {}
    for name, chunks in per_layer.items():
        out[name] = {k: float(np.mean([c[k] for c in chunks])) for k in chunks[0]}
    return out


def figure(profiles: dict, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap = plt.get_cmap("tab10")

    for i, (name, prof) in enumerate(sorted(profiles.items())):
        keys = sorted(prof)
        depth = np.linspace(0, 1, len(keys))
        share = [prof[k]["top1pct_share"] * 100 for k in keys]
        ratio = [prof[k]["max_over_median"] for k in keys]
        axes[0].plot(depth, share, "o-", color=cmap(i), label=name, lw=1.6, ms=3)
        axes[1].plot(depth, ratio, "o-", color=cmap(i), label=name, lw=1.6, ms=3)

    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_ylabel("% of total norm mass in the top 1% of tokens")
    axes[0].set_xlabel("relative depth")
    axes[0].set_title("If tokens were uniform this would sit at the dotted line (1%)")
    axes[0].legend(fontsize=7)

    axes[1].set_yscale("log")
    axes[1].set_ylabel("largest token norm / median")
    axes[1].set_xlabel("relative depth")
    axes[1].set_title("How extreme the most extreme token is")
    axes[1].legend(fontsize=7)

    fig.suptitle("F1 diagnostic: high-norm outlier tokens across depth", fontsize=11)
    fig.tight_layout()
    out = fig_dir / "f1_token_norm_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--n-images", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)

    images = np.load(paths["stimuli"] / "shared1000_images.npy", mmap_mode="r")[: args.n_images]
    images = np.asarray(images)
    device = pick_device(cfg.get("device", "auto"))
    print(f"{len(images)} images, device {device}\n")

    profiles = {}
    for spec in [ModelSpec.from_config(d) for d in cfg["models"]]:
        print(f"=== {spec.name} ===", flush=True)
        prof = profile_model(spec, images, device)
        profiles[spec.name] = prof

        keys = sorted(prof)
        print(f"  {'layer':<10}{'tokens':>8}{'median|x|':>11}{'max/med':>10}{'top1% share':>13}")
        for k in keys:
            p = prof[k]
            print(f"  {k:<10}{p['n_tokens']:>8.0f}{p['median_norm']:>11.2f}"
                  f"{p['max_over_median']:>10.1f}{p['top1pct_share']*100:>12.1f}%")
        peak = max(keys, key=lambda k: prof[k]["top1pct_share"])
        print(f"  -> worst layer: {peak} "
              f"({prof[peak]['top1pct_share']*100:.1f}% of norm in top 1%)\n", flush=True)

    out = paths["cache"] / "f1_token_norms.json"
    out.write_text(json.dumps(profiles, indent=1))
    print(f"figure: {figure(profiles, paths['figures'])}")
    print(f"data:   {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
