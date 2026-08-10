"""Loading the extracted betas and turning trials into per-image response patterns.

The file on disk holds one row per *presentation* (3000 for a complete subject). What
every analysis actually wants is one row per *image*. Collapsing those is where the
repeat structure gets used — for averaging, and for splitting the data in half to
measure how much reproducible signal is there at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nsd_rsa.rois import STREAM_LABELS


@dataclass
class SubjectBetas:
    """Per-trial betas for one subject, plus the labels needed to interpret them."""

    subject: str
    betas: np.ndarray      # (n_trials, n_vertices) float32
    image: np.ndarray      # (n_trials,) shared-image slot 0..999
    repeat: np.ndarray     # (n_trials,) 0/1/2
    session: np.ndarray    # (n_trials,) 1-based
    vertex_index: np.ndarray   # (n_vertices,) index into the full fsaverage surface
    roi_label: np.ndarray      # (n_vertices,) streams label value
    lh_size: int

    @property
    def n_images(self) -> int:
        return int(self.image.max()) + 1

    def roi_mask(self, roi: str) -> np.ndarray:
        inv = {v: k for k, v in STREAM_LABELS.items()}
        if roi not in inv:
            raise KeyError(f"unknown ROI {roi!r}")
        return self.roi_label == inv[roi]

    def roi_counts(self) -> dict[str, int]:
        return {name: int((self.roi_label == val).sum()) for val, name in STREAM_LABELS.items()}


def load_subject(path: str | Path) -> SubjectBetas:
    import h5py

    with h5py.File(path, "r") as f:
        return SubjectBetas(
            subject=f.attrs["subject"],
            betas=f["betas"][:],
            image=f["image"][:],
            repeat=f["repeat"][:],
            session=f["session"][:],
            vertex_index=f["vertex_index"][:],
            roi_label=f["roi_label"][:],
            lh_size=int(f.attrs["lh_size"]),
        )


def average_repeats(
    data: SubjectBetas, images: np.ndarray | None = None, roi: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse presentations into one response pattern per image.

    fMRI single-trial noise is large relative to the stimulus-driven signal, so a single
    beta is a poor estimate. Averaging k repeats cuts the noise standard deviation by
    about sqrt(k) while leaving the signal untouched — the same reason you average
    several noisy measurements of the same quantity.

    Returns (patterns, images_kept) where patterns is (n_images, n_vertices), laid out
    exactly like a batch of model activations so the RDM code treats both identically.
    """
    sel_v = slice(None) if roi is None else data.roi_mask(roi)
    betas = data.betas[:, sel_v]

    images = np.unique(data.image) if images is None else np.asarray(images)
    out = np.empty((len(images), betas.shape[1]), dtype=np.float32)
    for i, img in enumerate(images):
        rows = data.image == img
        if not rows.any():
            raise ValueError(f"image {img} has no trials for {data.subject}")
        out[i] = betas[rows].mean(axis=0)
    return out, images


def split_half(
    data: SubjectBetas, images: np.ndarray | None = None, roi: str | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each image's repeats into two independent halves and average within each.

    This is the workhorse of fMRI data quality assessment. Two halves built from disjoint
    presentations of the same images share only the stimulus-driven signal; anything they
    agree on is reproducible, and anything they don't is noise. Correlating them therefore
    measures signal directly, without needing a model.

    Images with fewer than 2 available repeats cannot be split and are dropped.

    Returns (half_a, half_b, images_kept).
    """
    rng = np.random.default_rng(seed)
    sel_v = slice(None) if roi is None else data.roi_mask(roi)
    betas = data.betas[:, sel_v]

    images = np.unique(data.image) if images is None else np.asarray(images)
    keep, a_rows, b_rows = [], [], []
    for img in images:
        idx = np.where(data.image == img)[0]
        if len(idx) < 2:
            continue
        perm = rng.permutation(idx)
        cut = len(perm) // 2
        a_rows.append(perm[:cut])
        b_rows.append(perm[cut : 2 * cut] if len(perm) % 2 == 0 else perm[cut:])
        keep.append(img)

    if not keep:
        raise ValueError(f"{data.subject}: no image has 2+ repeats, cannot split")

    a = np.stack([betas[r].mean(axis=0) for r in a_rows]).astype(np.float32)
    b = np.stack([betas[r].mean(axis=0) for r in b_rows]).astype(np.float32)
    return a, b, np.asarray(keep)
