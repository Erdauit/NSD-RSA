#!/usr/bin/env python
"""F2 — does the task the model was given change its alignment to cortex?

Four prompt conditions were extracted with the question placed *before* the image, so
causal attention lets it reach the visual tokens at all. The question now is whether the
resulting differences are larger than chance.

WHY THE OBVIOUS TEST IS WRONG. An RDM over 515 images has 132,355 cells, and it is
tempting to treat those as 132,355 observations. They are not independent: every cell is
a function of two image representations, so perturbing one image moves 514 cells at once.
The effective sample size is the number of images, not the number of pairs. A test that
ignores this reports p-values many orders of magnitude too small — it would call every
condition difference significant, including ones that are pure noise.

WHAT WE DO INSTEAD. The exchangeable unit under the null "the task makes no difference"
is the condition label itself, within a subject. So for each subject we take one summary
number per condition — mean ceiling-normalised RSA across the LLM layers — shuffle the
four condition labels within that subject, and rebuild the statistic. Subjects are never
mixed, because between-subject differences are real and not what we are testing.

The statistic is the spread (max - min) across condition means. It asks a single question:
do the conditions separate more than random relabelling would produce?

BUILT-IN NEGATIVE CONTROL. The vision tower runs before the language model and never sees
the prompt, so its readouts are identical across conditions by construction. If the test
reports an effect there, the test is broken rather than the finding being real.

Usage: make f2-task
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

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import average_repeats, common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.models import pool_tokens  # noqa: E402
from nsd_rsa.noise_ceiling import normalise_to_ceiling  # noqa: E402
from nsd_rsa.rdm import compare_rdms, compute_rdm  # noqa: E402
from nsd_rsa.vlm import PROMPTS, VLMSpec, build_vlm, image_token_mask, locate_stack  # noqa: E402

ROIS = ("early", "midventral", "ventral", "lateral")
POOL = "trim"          # F1 established this as the primary readout
N_PERM = 10000


@torch.no_grad()
def extract(spec: VLMSpec, images, prompt: str, device: str, cache: Path) -> dict:
    """Vision blocks and LLM image-token readouts for one prompt condition."""
    if cache.exists():
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}

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

    content = [{"type": "text", "text": prompt}, {"type": "image"}] if prompt else [
        {"type": "image"}
    ]
    chat = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
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
                pooled = pool_tokens(tensor.reshape(1, -1, tensor.shape[-1]))[POOL]
                acc.setdefault(key, []).append(pooled[0].cpu().numpy())
            captured.clear()

            for layer, h in enumerate(hidden):
                pooled = pool_tokens(h[0][mask].unsqueeze(0))[POOL]
                acc.setdefault(f"llm.{layer:02d}", []).append(pooled[0].cpu().numpy())

            if (n + 1) % 200 == 0:
                print(f"      {n+1}/{len(images)} ({(time.time()-t0)/(n+1):.2f} s/img)", flush=True)
    finally:
        for h in handles:
            h.remove()
        del model

    out = {k: np.stack(v).astype(np.float32) for k, v in acc.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def within_subject_permutation(values: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    """values: (n_subjects, n_conditions). Returns (observed spread, p).

    Shuffles condition labels independently within each subject, which is exactly the
    null "the task makes no difference" — and preserves each subject's own level, which
    is a real effect we are not testing.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.ptp(values.mean(axis=0)))
    count = 0
    for _ in range(n_perm):
        permuted = np.stack([rng.permutation(row) for row in values])
        if np.ptp(permuted.mean(axis=0)) >= observed:
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--models", nargs="*", default=["smolvlm_256m", "smolvlm_500m"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 0)
    set_seed(seed)
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
    conditions = list(PROMPTS)
    print(f"  {len(subjects)} subjects, {len(idx)} images, readout '{POOL}'\n")

    results: dict = {}
    for name in args.models:
        spec = next(VLMSpec.from_config(d) for d in cfg["models"] if d["name"] == name)
        print(f"=== {name} ===", flush=True)

        rdms_by_cond = {}
        for cond in conditions:
            cache = paths["cache"] / "f2_readouts" / f"{name}__{cond}__{POOL}.npz"
            print(f"  [{cond}]", flush=True)
            acts = extract(spec, images, PROMPTS[cond], device, cache)
            rdms_by_cond[cond] = {
                k: compute_rdm(v.astype(np.float64)) for k, v in acts.items()
            }

        keys = sorted(rdms_by_cond[conditions[0]])
        llm = sorted((k for k in keys if k.startswith("llm.")),
                     key=lambda k: int(k.split(".")[1]))
        vision = sorted(k for k in keys if k.startswith("vision."))

        results[name] = {}
        for family, layer_keys in (("llm", llm), ("vision", vision)):
            results[name][family] = {}
            for roi in ROIS:
                # (subjects x conditions) of mean normalised RSA across this layer family
                table = np.array([
                    [
                        np.mean([
                            normalise_to_ceiling(
                                compare_rdms(rdms_by_cond[c][k], brain[roi][s]), ceilings[roi]
                            ) for k in layer_keys
                        ])
                        for c in conditions
                    ]
                    for s in subjects
                ])
                spread, p = within_subject_permutation(table, N_PERM, seed)
                results[name][family][roi] = {
                    "means": {c: float(m) for c, m in zip(conditions, table.mean(axis=0), strict=True)},
                    "spread": spread,
                    "p": p,
                    "vs_control": {
                        c: float(np.mean(table[:, i] - table[:, 0]))
                        for i, c in enumerate(conditions)
                    },
                    "n_subjects_above_control": {
                        c: int((table[:, i] > table[:, 0]).sum())
                        for i, c in enumerate(conditions) if c != "none"
                    },
                }

    out = paths["cache"] / "f2_task_modulation.json"
    out.write_text(json.dumps(results, indent=1))

    for family, title in (("vision", "NEGATIVE CONTROL — vision tower (cannot see the prompt)"),
                          ("llm", "LLM LAYERS — where the task can act")):
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)
        head = "".join(f"{c:>10}" for c in conditions)
        print(f"{'model':<15}{'ROI':<12}{head}{'spread':>9}{'p':>9}")
        for name, fams in results.items():
            for roi in ROIS:
                r = fams[family][roi]
                cells = "".join(f"{r['means'][c]:>10.4f}" for c in conditions)
                print(f"{name:<15}{roi:<12}{cells}{r['spread']:>9.4f}{r['p']:>9.4f}")

    print("\n" + "=" * 92)
    print("EACH TASK MINUS THE NO-QUESTION CONTROL (LLM layers), and subjects above control")
    print("=" * 92)
    for name, fams in results.items():
        for roi in ROIS:
            r = fams["llm"][roi]
            parts = "  ".join(
                f"{c}: {r['vs_control'][c]:+.4f} ({r['n_subjects_above_control'][c]}/8)"
                for c in conditions if c != "none"
            )
            print(f"  {name:<15}{roi:<12}{parts}")

    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
