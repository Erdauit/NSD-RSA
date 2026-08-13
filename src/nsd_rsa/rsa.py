"""Stage-level RSA: banks of RDMs and the all-pairs comparison between them.

`rdm.py` holds the primitives. This module is about doing them at scale: 124 model
readouts against 56 brain RDMs is ~7000 Spearman correlations over 132,355-element
vectors, and the naive loop is minutes of rank-sorting the same vectors over and over.

The trick: Spearman correlation is just Pearson correlation on ranks. So rank each RDM
once, z-score it, and every pairwise correlation becomes a dot product — the whole
comparison matrix is a single matmul. Same numbers, ~100x faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nsd_rsa.rdm import _quiet_matmul, compute_rdm, rank_transform


@dataclass
class RDMBank:
    """A set of RDMs sharing one stimulus set, plus their labels."""

    labels: list[str]
    rdms: np.ndarray  # (n_items, n_pairs) float32

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_pairs(self) -> int:
        return self.rdms.shape[1]

    def subset(self, keep: list[str]) -> RDMBank:
        idx = [self.labels.index(k) for k in keep]
        return RDMBank([self.labels[i] for i in idx], self.rdms[idx])


def rank_zscore(rdms: np.ndarray) -> np.ndarray:
    """Rank-transform each row, then z-score it.

    After this, the Pearson correlation between two rows is their dot product divided by
    the vector length — which is exactly their Spearman correlation. Rows with zero
    variance (a degenerate RDM) become zeros, so they correlate 0 with everything rather
    than producing NaN and poisoning an entire heatmap row.
    """
    ranked = np.vstack([rank_transform(r) for r in np.atleast_2d(rdms)]).astype(np.float64)
    ranked -= ranked.mean(axis=1, keepdims=True)
    sd = ranked.std(axis=1, keepdims=True)
    out = np.divide(ranked, sd, out=np.zeros_like(ranked), where=sd > 0)
    return out


def compare_banks(a: RDMBank, b: RDMBank) -> np.ndarray:
    """All-pairs Spearman correlation. Returns (len(a), len(b))."""
    if a.n_pairs != b.n_pairs:
        raise ValueError(
            f"RDM length mismatch: {a.n_pairs} vs {b.n_pairs} — the two banks were built "
            "on different stimulus sets"
        )
    za, zb = rank_zscore(a.rdms), rank_zscore(b.rdms)
    with _quiet_matmul():
        out = (za @ zb.T) / za.shape[1]
    return np.clip(out, -1.0, 1.0)


def model_rdm_bank(
    path, images: np.ndarray, metric: str = "correlation", dtype=np.float32
) -> RDMBank:
    """Build one RDM per cached readout, restricted to `images` (shared-image slots).

    Caches differ in what they hold: the S2 vision-encoder caches store all 1000 shared
    images in slot order, while the VLM caches store only the analysis subset. A cache
    that records an `image_index` dataset is taken at its word and its rows are looked up;
    otherwise rows are assumed to be slot-ordered.

    That distinction is why this is not a plain fancy-index. Indexing a 515-row subset
    cache with slot numbers happened to raise IndexError here, but on a cache with 1000
    rows in a different order it would have silently correlated the wrong pictures.

    Readouts are kept separate rather than concatenated: a ViT block contributes both its
    CLS token and its mean-pooled patch tokens, and those can align with different cortex.
    """
    import h5py

    images = np.asarray(images)
    labels, rows = [], []
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if k != "image_index")

        if "image_index" in f:
            stored = f["image_index"][:]
            lookup = {int(img): i for i, img in enumerate(stored)}
            missing = [int(i) for i in images if int(i) not in lookup]
            if missing:
                raise KeyError(
                    f"{path}: cache lacks {len(missing)} requested images "
                    f"(first few: {missing[:5]}). It was built on a different subset."
                )
            take = np.array([lookup[int(i)] for i in images])
        else:
            n_rows = f[keys[0]].shape[0]
            if images.max() >= n_rows:
                raise IndexError(
                    f"{path}: image slot {images.max()} requested but the cache has only "
                    f"{n_rows} rows and carries no image_index to map them. Re-extract "
                    "so the cache records which images it holds."
                )
            take = images

        for key in keys:
            acts = f[key][:][take].astype(np.float64)
            labels.append(key)
            rows.append(compute_rdm(acts, metric=metric).astype(dtype))
    return RDMBank(labels, np.vstack(rows))


def brain_rdm_bank(
    data,
    rois: list[str],
    images: np.ndarray,
    metric: str = "correlation",
    dtype=np.float32,
    valid: np.ndarray | None = None,
) -> RDMBank:
    """Build one RDM per ROI for a subject, on the same stimulus subset.

    `images` are shared-image slots; responses are averaged over that image's repeats
    before the RDM is computed, because a single-trial pattern is far too noisy.
    """
    from nsd_rsa.loaders import average_repeats

    labels, rows = [], []
    for roi in rois:
        patterns, kept = average_repeats(data, images=images, roi=roi, valid=valid)
        if len(kept) != len(images):
            raise ValueError(f"{data.subject}/{roi}: expected {len(images)} images, got {len(kept)}")
        labels.append(roi)
        rows.append(compute_rdm(patterns.astype(np.float64), metric=metric).astype(dtype))
    return RDMBank(labels, np.vstack(rows))


def best_layer_per_roi(scores: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """For each ROI (column), the highest-scoring readout and its name."""
    idx = np.nanargmax(scores, axis=0)
    return scores[idx, np.arange(scores.shape[1])], [labels[i] for i in idx]


def layer_depth_index(labels: list[str]) -> np.ndarray:
    """Normalised depth 0..1 for each readout, from its layer name.

    Used to ask the hierarchy question — whether the best-matching layer moves deeper as
    you move forward through cortex — on a scale comparable across models of different
    depth.
    """
    stems = [lab.split(".")[0] for lab in labels]
    order = sorted(set(stems))
    rank = {s: i for i, s in enumerate(order)}
    denom = max(len(order) - 1, 1)
    return np.array([rank[s] / denom for s in stems], dtype=float)
