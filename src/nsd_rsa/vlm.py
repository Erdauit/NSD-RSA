"""Reading representations from every stage of a vision-language model.

The point of this module is the part nobody measures. Brain-alignment work stops at the
vision encoder, but in a VLM the visual tokens then travel through a language model for
another twenty to thirty layers. This extracts a single continuous depth axis across the
whole stack:

    vision.00 .. vision.NN  ->  projector  ->  llm.00 .. llm.MM

Two decisions here are load-bearing and were established by measurement, not preference:

**float32.** bfloat16 is the reflexive choice and is harmless for text generation, but we
read the *geometry* of hidden states and compare models whose scores differ by ~0.04.
Measured on SmolVLM-2.2B, bf16 moves the RDM by 0.02-0.05 against an fp32 reference that
reproduces itself bitwise. That is the same size as the effects we report.
See scripts/vlm_precision_check.py.

**No image tiling.** SmolVLM splits an image into tiles by default, giving a 1563-token
sequence instead of 103 — 33 s/img instead of ~1-4 in float32. It is also the cleaner
comparison: cortex sees the whole visual field at once, not high-resolution fragments.

Pooling. For the LLM we keep two readouts per layer, for the same reason S2 keeps CLS and
patch-mean separately: they can align with different cortex.
  `imgmean`   — mean over image-token positions: what the visual content became
  `lasttoken` — the final position, which is what the model actually reads out to answer

**Prompt position is not cosmetic.** Attention is causal, so an image token can only
attend to what precedes it. With the usual layout (image, then question) the image tokens
literally cannot see the question, and `imgmean` is bitwise identical across prompts —
verified: max difference 0.000000. Running four prompts that way would produce four
identical copies and a guaranteed "no task modulation" null that is an artefact of
architecture, not a result.

Placing the question first makes top-down modulation measurable (max difference rises to
~0.9 by mid-depth), and it is the correct analogue of the human experiment: you tell a
subject what to look for *before* showing the picture. `after` remains available for the
standard-usage configuration, where only `lasttoken` can vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Prompt conditions. Human visual cortex is modulated by task; a VLM lets us set the task
# literally. `none` is the control: the image with no question attached.
PROMPTS: dict[str, str] = {
    "none": "",
    "objects": "Describe the objects in this image.",
    "spatial": "Describe the spatial layout of this scene.",
    "mood": "Describe the mood and atmosphere of this image.",
}


@dataclass
class VLMSpec:
    name: str
    hf_name: str
    note: str = ""
    n_params: float = 0.0

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> VLMSpec:
        return cls(name=d["name"], hf_name=d["hf_name"], note=d.get("note", ""))


@dataclass
class StackReadouts:
    """Activations for one (model, prompt), keyed by readout name."""

    data: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, key: str, vec: np.ndarray) -> None:
        self.data.setdefault(key, []).append(vec)

    def finalise(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v).astype(np.float32) for k, v in self.data.items()}


def build_vlm(spec: VLMSpec, device: str, tiling: bool = False):
    """Load a VLM in float32 with tiling disabled.

    NOTE: `device_map=` segfaults (SIGSEGV) under transformers 5.15 + torch 2.13 on MPS,
    so the model is loaded on CPU and moved afterwards. Equivalent, and stable.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(spec.hf_name)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = tiling

    model = AutoModelForImageTextToText.from_pretrained(spec.hf_name, dtype=torch.float32)
    model.eval()
    model = model.to(device)
    spec.n_params = sum(p.numel() for p in model.parameters()) / 1e9
    return model, processor


def locate_stack(model) -> dict[str, Any]:
    """Find the vision blocks, the projector and the LLM blocks.

    Kept explicit and fail-loud: silently missing the vision tower would leave us
    analysing only the language half while believing we covered the stack.
    """
    inner = getattr(model, "model", model)
    vision = getattr(inner, "vision_model", None)
    connector = getattr(inner, "connector", None)
    text = getattr(inner, "text_model", None)

    if vision is None or text is None:
        raise AttributeError(
            f"{type(model).__name__}: expected .model.vision_model and .model.text_model. "
            "This model's layout differs; add a mapping rather than guessing."
        )

    vision_blocks = None
    for module in vision.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 1:
            vision_blocks = module
            break
    if vision_blocks is None:
        raise AttributeError("could not find the vision encoder's block list")

    return {
        "vision_blocks": vision_blocks,
        "connector": connector,
        "n_llm_layers": len(text.layers),
    }


def image_token_mask(input_ids: torch.Tensor, model, processor) -> torch.Tensor:
    """Positions holding image tokens.

    Refuses to guess. Pooling over every position would average the prompt's words into
    what we then call a visual representation — and since the prompt differs between our
    conditions, that would fabricate exactly the task effect we are testing for.
    """
    candidates: list[int] = []
    for obj in (model.config, getattr(model.config, "text_config", None), processor):
        for attr in ("image_token_id", "image_token_index"):
            val = getattr(obj, attr, None)
            if isinstance(val, int):
                candidates.append(val)

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        for name in ("<image>", "<|image_pad|>", "<image_soft_token>"):
            tid = tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0:
                candidates.append(tid)

    for tid in dict.fromkeys(candidates):
        mask = input_ids == tid
        if mask.any():
            return mask

    raise RuntimeError(
        f"no image tokens found (tried ids {sorted(set(candidates))}). Refusing to pool "
        "over all positions, which would mix prompt text into the visual representation."
    )


@torch.no_grad()
def extract_stack(
    model,
    processor,
    images: np.ndarray,
    prompt: str,
    device: str,
    progress_every: int = 50,
    prompt_position: str = "before",
) -> dict[str, np.ndarray]:
    """Run every image and collect readouts across the full stack.

    `prompt_position` decides whether the question precedes the image. See the module
    docstring: with "after", image-token readouts cannot depend on the prompt at all.
    """
    from PIL import Image

    stack = locate_stack(model)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def fn(_m, _inp, out):
            captured[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    for i, block in enumerate(stack["vision_blocks"]):
        handles.append(block.register_forward_hook(hook(f"vision.{i:02d}")))
    if stack["connector"] is not None:
        handles.append(stack["connector"].register_forward_hook(hook("projector")))

    if prompt_position not in ("before", "after"):
        raise ValueError(f"prompt_position must be 'before' or 'after', got {prompt_position!r}")
    content: list[dict] = [{"type": "image"}]
    if prompt:
        text = {"type": "text", "text": prompt}
        content = [text, {"type": "image"}] if prompt_position == "before" else [
            {"type": "image"}, text
        ]
    chat = processor.apply_chat_template([{"role": "user", "content": content}],
                                         add_generation_prompt=True)

    out = StackReadouts()
    try:
        for idx in range(len(images)):
            pil = Image.fromarray(np.asarray(images[idx]))
            inputs = processor(text=chat, images=[pil], return_tensors="pt").to(device)

            result = model(**inputs, output_hidden_states=True, use_cache=False)
            mask = image_token_mask(inputs["input_ids"], model, processor)[0]

            # Vision tower and projector: mean over tokens (and over sub-images, if any).
            for key, tensor in captured.items():
                flat = tensor.reshape(-1, tensor.shape[-1])
                out.add(key, flat.float().mean(0).cpu().numpy())
            captured.clear()

            # LLM: one readout per layer, two poolings.
            for layer, hidden in enumerate(result.hidden_states):
                out.add(f"llm.{layer:02d}.imgmean",
                        hidden[0][mask].float().mean(0).cpu().numpy())
                out.add(f"llm.{layer:02d}.lasttoken",
                        hidden[0][-1].float().cpu().numpy())

            if progress_every and (idx + 1) % progress_every == 0:
                print(f"      {idx+1}/{len(images)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    return out.finalise()
