#!/usr/bin/env python
"""RSA against the full VLM stack: where does cortical alignment live, and does the task
change it?

Two questions, two figures.

1. **Stack profile.** Alignment as a function of position along
   vision.00 -> ... -> projector -> llm.00 -> ... -> llm.NN, one line per cortical region.
   Brain-alignment work stops at the vision encoder; this asks what the language model
   does to the visual geometry over the following twenty-odd layers.

2. **Task modulation.** The same profile under different questions, all asked *before*
   the image so that causal attention lets them influence the visual tokens at all.
   Human visual cortex is modulated by task; this asks whether a VLM's alignment is a
   fixed property of the model or a property of the model in a task state.

Everything is normalised to the noise ceiling computed in S3, so numbers are directly
comparable to the vision-encoder results already in the paper.

Usage: make vlm-rsa
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import load_expdesign, usable_images  # noqa: E402
from nsd_rsa.loaders import common_valid_vertices, load_subject  # noqa: E402
from nsd_rsa.noise_ceiling import normalise_to_ceiling  # noqa: E402
from nsd_rsa.rsa import brain_rdm_bank, compare_banks, model_rdm_bank  # noqa: E402

READOUT = re.compile(r"^(vision)\.(\d+)$|^(projector)$|^llm\.(\d+)\.(imgmean|lasttoken)$")


def stack_position(label: str, n_vision: int) -> tuple[int, str] | None:
    """Map a readout name onto one ordered axis through the whole model.

    Returns (position, series). `series` separates the two LLM poolings so they are drawn
    as distinct curves rather than interleaved into one jagged line.
    """
    m = READOUT.match(label)
    if not m:
        return None
    if m.group(1):
        return int(m.group(2)), "stack"
    if m.group(3):
        return n_vision, "stack"
    return n_vision + 1 + int(m.group(4)), m.group(5)


def load_stack_scores(path: Path, brain: dict, subjects: list[str], images, ceilings, rois):
    """RSA of every readout in one cache against every ROI, normalised, averaged over
    subjects."""
    import h5py

    bank = model_rdm_bank(path, images)
    per_subject = np.stack([compare_banks(bank, brain[s]) for s in subjects])
    raw = per_subject.mean(axis=0)

    norm = np.empty_like(raw)
    for j, roi in enumerate(rois):
        norm[:, j] = [normalise_to_ceiling(v, ceilings[roi][0]) for v in raw[:, j]]

    with h5py.File(path, "r") as f:
        n_vision = len([k for k in f.keys() if k.startswith("vision.")])
    return bank.labels, raw, norm, per_subject, n_vision


def series_from(labels, values, n_vision):
    """Split readouts into ordered curves keyed by series name."""
    out: dict[str, list[tuple[int, float]]] = {}
    for i, lab in enumerate(labels):
        placed = stack_position(lab, n_vision)
        if placed is None:
            continue
        pos, series = placed
        out.setdefault(series, []).append((pos, values[i]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def figure_stack_profile(results, rois, n_vision_by_model, fig_dir, prompt="none"):
    models = sorted(results)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), squeeze=False)
    colours = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(rois)))

    for ax, model in zip(axes[0], models, strict=False):
        res = results[model][prompt]
        n_vision = n_vision_by_model[model]
        for j, (roi, colour) in enumerate(zip(rois, colours, strict=False)):
            curves = series_from(res["labels"], res["norm"][:, j], n_vision)
            if "stack" in curves and "imgmean" in curves:
                joined = np.vstack([curves["stack"], curves["imgmean"]])
                ax.plot(joined[:, 0], joined[:, 1], "-", color=colour, label=roi, lw=1.8)
        ax.axvline(n_vision, color="k", ls=":", lw=1.2)
        ax.text(n_vision, ax.get_ylim()[1], " projector", fontsize=7, va="top")
        ax.set_xlabel("position in stack  (vision blocks | projector | LLM layers)")
        ax.set_ylabel("RSA / noise ceiling")
        ax.set_title(model, fontsize=10)
        ax.axhline(0, color="k", lw=0.5)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Where cortical alignment lives along the VLM stack (prompt: {prompt})",
                 fontsize=11)
    fig.tight_layout()
    out = fig_dir / "vlm_stack_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_task_modulation(results, rois, n_vision_by_model, fig_dir):
    """Does the question change where alignment sits?"""
    models = sorted(results)
    fig, axes = plt.subplots(len(models), len(rois), figsize=(2.6 * len(rois), 2.5 * len(models)),
                             squeeze=False, sharex="row")
    for r, model in enumerate(models):
        n_vision = n_vision_by_model[model]
        prompts = sorted(results[model])
        for c, roi in enumerate(rois):
            ax = axes[r][c]
            j = rois.index(roi)
            for prompt in prompts:
                res = results[model][prompt]
                curves = series_from(res["labels"], res["norm"][:, j], n_vision)
                if "stack" in curves and "imgmean" in curves:
                    joined = np.vstack([curves["stack"], curves["imgmean"]])
                    ax.plot(joined[:, 0], joined[:, 1], lw=1.4, label=prompt)
            ax.axvline(n_vision, color="k", ls=":", lw=0.8)
            if r == 0:
                ax.set_title(roi, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{model}\nRSA / ceiling", fontsize=8)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Task modulation: same stack, different question (asked before the image)",
                 fontsize=11)
    fig.tight_layout()
    out = fig_dir / "vlm_task_modulation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--rsa-config", default="configs/rsa.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rsa_cfg = load_config(args.rsa_config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    rois = rsa_cfg["rois"]["order"]
    position = cfg.get("prompt_position", "before")

    results_path = paths["cache"] / "rsa_results.json"
    if not results_path.exists():
        print("Run scripts/s3_rsa.py first — the noise ceilings come from there.")
        return 1
    ceilings = json.loads(results_path.read_text())["ceilings"]

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    images = usable_images(cfg["stimuli"]["subjects"], design, cfg["stimuli"]["min_repeats"])
    valid = common_valid_vertices(paths["betas"], paths["cache"] / "valid_vertices.npy")

    print(f"stimulus set: {len(images)} images")
    print("building brain RDMs")
    brain = {}
    for p in sorted(paths["betas"].glob("*_shared_betas.h5")):
        data = load_subject(p)
        brain[p.stem.split("_")[0]] = brain_rdm_bank(
            data, rois, images, metric=rsa_cfg["rdm"]["metric"], valid=valid
        )
        del data
    subjects = sorted(brain)
    print(f"  {len(subjects)} subjects\n")

    act_dir = paths["cache"] / "vlm_activations"
    results: dict[str, dict] = {}
    n_vision_by_model: dict[str, int] = {}

    for spec in cfg["models"]:
        model = spec["name"]
        for prompt in cfg["prompts"]:
            path = act_dir / f"{model}__{prompt}__{position}.h5"
            if not path.exists():
                continue
            labels, raw, norm, per_subject, n_vision = load_stack_scores(
                path, brain, subjects, images, ceilings, rois
            )
            results.setdefault(model, {})[prompt] = {
                "labels": labels, "raw": raw, "norm": norm, "per_subject": per_subject
            }
            n_vision_by_model[model] = n_vision
            best = int(np.nanargmax(norm[:, rois.index("ventral")]))
            print(f"  {model:<16}{prompt:<9} peak in ventral: {norm[best, rois.index('ventral')]:.3f} "
                  f"at {labels[best]}")

    if not results:
        print("No VLM activations found. Run: make vlm-extract")
        return 1

    fig_dir = paths["figures"]
    print()
    print(f"  {figure_stack_profile(results, rois, n_vision_by_model, fig_dir)}")
    print(f"  {figure_task_modulation(results, rois, n_vision_by_model, fig_dir)}")

    print("\n" + "=" * 94)
    print("PEAK ALIGNMENT BY REGION AND TASK  (normalised to the S3 noise ceiling)")
    print("=" * 94)
    print(f"{'model':<15}{'prompt':<9}{'ROI':<12}{'peak':>7}{'at readout':>22}{'stack %':>9}")
    summary = {}
    for model, by_prompt in results.items():
        n_vision = n_vision_by_model[model]
        for prompt, res in by_prompt.items():
            for j, roi in enumerate(rois):
                col = res["norm"][:, j]
                best = int(np.nanargmax(col))
                placed = stack_position(res["labels"][best], n_vision)
                n_positions = max(
                    (stack_position(lab, n_vision) or (0, ""))[0] for lab in res["labels"]
                )
                depth = placed[0] / max(n_positions, 1) if placed else float("nan")
                print(f"{model:<15}{prompt:<9}{roi:<12}{col[best]:>7.3f}"
                      f"{res['labels'][best]:>22}{depth:>9.2f}")
                summary[f"{model}|{prompt}|{roi}"] = {
                    "peak": float(col[best]), "readout": res["labels"][best]
                }

    out = paths["cache"] / "vlm_rsa_results.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
