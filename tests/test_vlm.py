"""Tests for the VLM stack extraction.

The two things tested here are the ones that would silently invalidate the study rather
than crash it:

  * pooling over the wrong token positions would average prompt text into what we call a
    "visual representation" — and since the prompt differs between conditions, that would
    manufacture the very task effect we are testing for;
  * putting the question after the image makes image-token readouts provably independent
    of it, so four prompt conditions would produce four identical copies and a null result
    caused by causal attention rather than by the model.
"""

import numpy as np
import pytest
import torch

from nsd_rsa.vlm import (
    PROMPTS,
    StackReadouts,
    VLMSpec,
    _move_prompt_before_image,
    image_token_mask,
    locate_stack,
)

# --- prompt conditions ---------------------------------------------------------


def test_control_prompt_is_empty():
    """The control must carry no text at all, not a neutral sentence — otherwise it is
    another task condition rather than a baseline."""
    assert PROMPTS["none"] == ""


def test_task_prompts_are_distinct_and_nonempty():
    tasks = [v for k, v in PROMPTS.items() if k != "none"]
    assert len(tasks) == len(set(tasks))
    assert all(t.strip() for t in tasks)


# --- spec ----------------------------------------------------------------------


def test_spec_from_config():
    spec = VLMSpec.from_config({"name": "m", "hf_name": "org/m", "note": "n"})
    assert (spec.name, spec.hf_name, spec.note) == ("m", "org/m", "n")


# --- readout accumulation ------------------------------------------------------


def test_readouts_stack_in_insertion_order():
    r = StackReadouts()
    for i in range(3):
        r.add("llm.00.imgmean", np.full(4, float(i)))
    out = r.finalise()
    assert out["llm.00.imgmean"].shape == (3, 4)
    assert out["llm.00.imgmean"].dtype == np.float32
    assert list(out["llm.00.imgmean"][:, 0]) == [0.0, 1.0, 2.0]


def test_readouts_keep_keys_separate():
    r = StackReadouts()
    r.add("vision.00", np.ones(2))
    r.add("llm.05.lasttoken", np.zeros(3))
    out = r.finalise()
    assert set(out) == {"vision.00", "llm.05.lasttoken"}
    assert out["vision.00"].shape == (1, 2)


# --- locating the stack --------------------------------------------------------


class _Fake:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def modules(self):
        yield self
        for v in vars(self).values():
            if isinstance(v, torch.nn.Module):
                yield from v.modules()


def _fake_model(with_vision=True):
    vision_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    vision = _Fake(encoder=vision_blocks) if with_vision else None
    text = _Fake(layers=torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(6)]))
    return _Fake(model=_Fake(vision_model=vision, connector=torch.nn.Linear(2, 2),
                             text_model=text))


def test_locate_stack_finds_blocks_and_counts_layers():
    found = locate_stack(_fake_model())
    assert len(found["vision_blocks"]) == 4
    assert found["n_llm_layers"] == 6
    assert found["connector"] is not None


def _fake_qwen2vl():
    """Qwen2-VL layout: .model.visual (blocks + merger inside) and .model.language_model."""
    visual = _Fake(
        blocks=torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(5)]),
        merger=torch.nn.Linear(2, 2),
    )
    text = _Fake(layers=torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(7)]))
    return _Fake(model=_Fake(visual=visual, language_model=text))


def _fake_llava():
    """LLaVA layout: vision_tower / multi_modal_projector / language_model."""
    vision = _Fake(encoder=torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)]))
    text = _Fake(layers=torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(5)]))
    return _Fake(model=_Fake(vision_tower=vision, multi_modal_projector=torch.nn.Linear(2, 2),
                             language_model=text))


def test_locate_stack_finds_qwen2vl_layout():
    """The projector lives inside the vision module (PatchMerger), not beside it."""
    found = locate_stack(_fake_qwen2vl())
    assert len(found["vision_blocks"]) == 5
    assert found["n_llm_layers"] == 7
    assert isinstance(found["connector"], torch.nn.Linear)


def test_locate_stack_finds_llava_layout():
    found = locate_stack(_fake_llava())
    assert len(found["vision_blocks"]) == 3
    assert found["n_llm_layers"] == 5
    assert found["connector"] is not None


def test_locate_stack_refuses_unknown_layout():
    """Silently missing the vision tower would leave us analysing only the language half
    while believing the whole stack was covered."""
    with pytest.raises(AttributeError, match="vision_model"):
        locate_stack(_fake_model(with_vision=False))


# --- prompt-before-image reordering -------------------------------------------


def test_reorder_moves_prompt_ahead_of_hardcoded_image():
    """LLaVA-OneVision's template pins <image> first regardless of content order; the
    rendered string must be repaired or 'before' silently becomes 'after'."""
    chat = "<|im_start|>user <image>\nDescribe the objects.<|im_end|><|im_start|>assistant\n"
    fixed = _move_prompt_before_image(chat, "Describe the objects.")
    assert fixed.index("Describe the objects.") < fixed.index("<image>")
    assert fixed.count("Describe the objects.") == 1
    assert fixed.count("<image>") == 1


def test_reorder_is_noop_when_template_honours_order():
    chat = "<|im_start|>user Describe.\n<image><|im_end|>"
    assert _move_prompt_before_image(chat, "Describe.") == chat


def test_reorder_refuses_without_known_placeholder():
    with pytest.raises(RuntimeError, match="placeholder"):
        _move_prompt_before_image("user says things", "says")


# --- image token location ------------------------------------------------------


class _Tok:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.unk_token_id = -1

    def convert_tokens_to_ids(self, name):
        return self.mapping.get(name, -1)


def test_finds_image_tokens_via_config_id():
    model = _Fake(config=_Fake(image_token_id=42, text_config=None))
    processor = _Fake(tokenizer=_Tok())
    ids = torch.tensor([[1, 42, 42, 7]])
    mask = image_token_mask(ids, model, processor)
    assert mask.tolist() == [[False, True, True, False]]


def test_finds_image_tokens_via_tokenizer_when_config_silent():
    model = _Fake(config=_Fake(text_config=None))
    processor = _Fake(tokenizer=_Tok({"<image>": 99}))
    ids = torch.tensor([[99, 3, 99]])
    assert image_token_mask(ids, model, processor).sum().item() == 2


def test_refuses_when_no_image_tokens_present():
    """Must fail loudly. Falling back to pooling over every position would mix the prompt
    into the visual representation, and the prompt is exactly what we vary."""
    model = _Fake(config=_Fake(image_token_id=42, text_config=None))
    processor = _Fake(tokenizer=_Tok())
    with pytest.raises(RuntimeError, match="Refusing to pool"):
        image_token_mask(torch.tensor([[1, 2, 3]]), model, processor)


def test_mask_selects_only_image_positions():
    model = _Fake(config=_Fake(image_token_id=5, text_config=None))
    processor = _Fake(tokenizer=_Tok())
    ids = torch.tensor([[5, 5, 8, 9]])
    hidden = torch.tensor([[[1.0], [3.0], [100.0], [200.0]]])
    mask = image_token_mask(ids, model, processor)[0]
    pooled = hidden[0][mask].mean(0).item()
    assert pooled == pytest.approx(2.0)  # text positions must not contribute
