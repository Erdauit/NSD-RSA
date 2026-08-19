"""Model registry and layer-wise activation extraction.

Adding a model is one entry in configs/models.yaml. Everything here is generic over
timm models, so DINOv2, supervised ViT, CLIP's image tower and ResNet-50 all go through
one code path — which matters, because any difference in preprocessing or pooling
between models would show up as a difference in brain alignment and be indistinguishable
from a real effect.

**Why we extract CLS and mean-pooled patch tokens separately.**
A ViT block outputs one vector per token: a CLS token plus one per image patch. The CLS
token is trained to be a summary — for DINOv2 it is the target of the self-distillation
objective, so it carries whatever the training task needed. The mean over patch tokens is
a different summary: it weights every spatial location equally and keeps texture and
layout information that the CLS token may have discarded. They can therefore align with
different parts of cortex — patch-mean plausibly closer to early, retinotopic areas, CLS
closer to anterior object-selective ones. Collapsing them into one number would hide
exactly the effect we are looking for, so we keep both and report both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class ModelSpec:
    """One row of the registry."""

    name: str                      # our short name, used in filenames
    timm_name: str                 # timm identifier including pretrained tag
    kind: str                      # 'vit' or 'cnn' — decides how layers are enumerated
    supervision: str               # 'self-supervised' | 'supervised' | 'language'
    note: str = ""
    input_size: int | None = None  # override the checkpoint's default resolution
    layers: list[str] = field(default_factory=list)  # filled in at build time

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(
            name=d["name"],
            timm_name=d["timm_name"],
            kind=d["kind"],
            supervision=d.get("supervision", "unknown"),
            note=d.get("note", ""),
            input_size=d.get("input_size"),
        )


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(spec: ModelSpec, device: torch.device):
    """Instantiate a pretrained model plus its own preprocessing transform.

    Normalisation statistics always come from timm's config for that checkpoint, never a
    shared default — feeding a model the wrong mean/std quietly degrades its
    representations.

    Resolution is different, and is a deliberate choice rather than a detail. timm's
    default for DINOv2 is 518x518 but 224x224 for supervised ViT-B/16 and CLIP. Comparing
    them at their own defaults would confound the thing we are testing: a difference in
    brain alignment could come from seeing the image at 518 rather than from the training
    objective. `input_size` in the registry forces a common resolution so the pairs differ
    in exactly one thing.
    """
    import timm

    kwargs: dict[str, Any] = {"pretrained": True, "num_classes": 0}
    # Only ViTs need img_size at construction: their positional embeddings are tied to a
    # fixed token grid, so timm has to interpolate them. A CNN is fully convolutional and
    # accepts any input, so its resolution is set purely by the transform — passing
    # img_size to a ResNet is a TypeError.
    if spec.input_size is not None and spec.kind == "vit":
        kwargs["img_size"] = spec.input_size

    model = timm.create_model(spec.timm_name, **kwargs)
    model.eval().to(device)

    cfg = timm.data.resolve_data_config({}, model=model)
    if spec.input_size is not None:
        cfg["input_size"] = (3, spec.input_size, spec.input_size)
        cfg["crop_pct"] = 1.0
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform, cfg


def enumerate_layers(model, spec: ModelSpec) -> list[tuple[str, torch.nn.Module]]:
    """The modules whose outputs we record.

    For ViTs: every transformer block, giving a clean depth axis to compare against the
    cortical hierarchy. For CNNs: the four residual stages, which is the coarse
    equivalent.
    """
    if spec.kind == "vit":
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise AttributeError(f"{spec.name}: expected a `blocks` attribute on a ViT")
        return [(f"block{i:02d}", b) for i, b in enumerate(blocks)]

    if spec.kind == "cnn":
        out = []
        for i in range(1, 5):
            layer = getattr(model, f"layer{i}", None)
            if layer is not None:
                out.append((f"stage{i}", layer))
        if not out:
            raise AttributeError(f"{spec.name}: no layer1..layer4 found")
        return out

    raise ValueError(f"unknown model kind {spec.kind!r}")


def _pool(output: torch.Tensor, kind: str) -> dict[str, torch.Tensor]:
    """Reduce a layer output to fixed-length vectors, one per image.

    ViT block output is (batch, tokens, dim); CNN stage output is (batch, ch, h, w).
    Both are reduced to (batch, features) because an RDM needs one vector per stimulus.
    """
    if kind == "vit":
        if output.ndim != 3:
            raise ValueError(f"expected (B, T, D) from a ViT block, got {tuple(output.shape)}")
        return {"cls": output[:, 0, :], "patchmean": output[:, 1:, :].mean(dim=1)}

    if output.ndim != 4:
        raise ValueError(f"expected (B, C, H, W) from a CNN stage, got {tuple(output.shape)}")
    return {"gap": output.mean(dim=(2, 3))}


@torch.no_grad()
def extract_activations(
    model,
    spec: ModelSpec,
    images: np.ndarray,
    transform,
    device: torch.device,
    batch_size: int = 32,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Run images through the model, returning {f"{layer}.{pool}": (n_images, dim)}.

    Images arrive as uint8 (N, H, W, 3) — the raw NSD stimulus array.
    """
    from PIL import Image

    layers = enumerate_layers(model, spec)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inp, out):
            captured[layer_name] = out.detach()
        return hook

    for layer_name, module in layers:
        handles.append(module.register_forward_hook(make_hook(layer_name)))

    collected: dict[str, list[np.ndarray]] = {}
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([transform(Image.fromarray(im)) for im in batch]).to(device)
            model(tensors)

            for layer_name, _ in layers:
                for pool_name, vec in _pool(captured[layer_name], spec.kind).items():
                    key = f"{layer_name}.{pool_name}"
                    collected.setdefault(key, []).append(vec.float().cpu().numpy())
            captured.clear()

            if progress and (start // batch_size) % 5 == 0:
                print(f"    {min(start + batch_size, len(images)):4d}/{len(images)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def pool_tokens(tokens: torch.Tensor, trim_quantile: float = 0.01) -> dict[str, torch.Tensor]:
    """Three ways to collapse (batch, tokens, features) into one vector per image.

    A mean is dominated by its largest terms, and trained ViTs put one to two orders of
    magnitude more norm into a handful of "outlier" tokens that carry global scratch
    information rather than anything about their own patch (Darcet et al. 2024). Measured
    on SmolVLM's SigLIP tower, the top 1% of tokens carry up to 15% of all norm mass at
    mid-depth — and RSA against early and ventral cortex collapses in exactly those layers
    (r = -0.73 and -0.78 between outlier share and alignment).

    So the pooling choice is not a detail, and we compute all three rather than assume:
      mean   — what we used originally, dominated by outliers where they exist
      trim   — mean after dropping the top `trim_quantile` of tokens by norm
      median — elementwise median, robust but a different statistic entirely
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected (batch, tokens, features), got {tuple(tokens.shape)}")
    n_tokens = tokens.shape[1]
    k = max(1, int(round(n_tokens * trim_quantile)))
    if k >= n_tokens:
        raise ValueError(f"trim_quantile {trim_quantile} would drop all {n_tokens} tokens")

    tokens = tokens.float()
    norms = tokens.norm(dim=-1)
    drop = norms.topk(k, dim=1).indices
    keep = torch.ones_like(norms, dtype=torch.bool).scatter_(1, drop, False)

    return {
        "mean": tokens.mean(dim=1),
        "trim": (tokens * keep.unsqueeze(-1)).sum(dim=1) / keep.sum(dim=1, keepdim=True),
        "median": tokens.median(dim=1).values,
    }
