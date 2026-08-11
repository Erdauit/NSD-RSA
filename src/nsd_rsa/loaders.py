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


def common_valid_vertices(betas_dir: str | Path, cache: str | Path | None = None) -> np.ndarray:
    """Vertices with finite data in EVERY subject.

    NSD's functional acquisition is a slab centred on visual cortex, not a whole-brain
    scan. For some subjects the most posterior edge of V1 falls outside it, so those
    fsaverage vertices have no data and arrive as NaN. Measured across all eight
    subjects: 350 such vertices in subj06 and 37 in subj08, all in `early` — 0.57% of the
    67,696 we keep.

    They must be dropped from every subject, not just the affected ones. An ROI has to
    mean the same set of cortical locations in each subject, or the noise ceiling — which
    is built by comparing subjects' RDMs — would partly measure differences in scanner
    coverage rather than differences in representation.
    """
    import h5py

    betas_dir = Path(betas_dir)
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        return np.load(cache)

    files = sorted(betas_dir.glob("*_shared_betas.h5"))
    if not files:
        raise FileNotFoundError(f"no subject files in {betas_dir}")

    valid: np.ndarray | None = None
    for path in files:
        with h5py.File(path, "r") as f:
            betas = f["betas"]
            ok = np.ones(betas.shape[1], dtype=bool)
            for start in range(0, betas.shape[0], 500):
                ok &= np.isfinite(betas[start : start + 500]).all(axis=0)
        valid = ok if valid is None else (valid & ok)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, valid)
    return valid


def _vertex_selector(
    data: SubjectBetas, roi: str | None, valid: np.ndarray | None
) -> np.ndarray:
    """Boolean column selector combining the ROI mask with the cross-subject valid mask."""
    mask = np.ones(data.betas.shape[1], dtype=bool) if roi is None else data.roi_mask(roi)
    if valid is not None:
        if len(valid) != data.betas.shape[1]:
            raise ValueError(
                f"valid mask has {len(valid)} entries but {data.subject} has "
                f"{data.betas.shape[1]} vertices"
            )
        mask = mask & valid
    if not mask.any():
        raise ValueError(f"{data.subject}/{roi}: no vertices left after masking")
    return mask


def average_repeats(
    data: SubjectBetas,
    images: np.ndarray | None = None,
    roi: str | None = None,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse presentations into one response pattern per image.

    fMRI single-trial noise is large relative to the stimulus-driven signal, so a single
    beta is a poor estimate. Averaging k repeats cuts the noise standard deviation by
    about sqrt(k) while leaving the signal untouched — the same reason you average
    several noisy measurements of the same quantity.

    Returns (patterns, images_kept) where patterns is (n_images, n_vertices), laid out
    exactly like a batch of model activations so the RDM code treats both identically.
    """
    betas = data.betas[:, _vertex_selector(data, roi, valid)]

    images = np.unique(data.image) if images is None else np.asarray(images)
    out = np.empty((len(images), betas.shape[1]), dtype=np.float32)
    for i, img in enumerate(images):
        rows = data.image == img
        if not rows.any():
            raise ValueError(f"image {img} has no trials for {data.subject}")
        out[i] = betas[rows].mean(axis=0)
    return out, images


def split_half(
    data: SubjectBetas,
    images: np.ndarray | None = None,
    roi: str | None = None,
    seed: int = 0,
    valid: np.ndarray | None = None,
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
    betas = data.betas[:, _vertex_selector(data, roi, valid)]

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
