"""RDM construction and comparison — the mathematical core of RSA.

An RDM (representational dissimilarity matrix) is the N x N matrix of pairwise distances
between the response patterns evoked by N stimuli. Its point is dimensionality invariance:
a model layer with 768 units and a cortical ROI with 4000 vertices produce RDMs of the
same shape, so they can be compared directly.

Conventions used throughout:
  * `patterns` is always (n_stimuli, n_features) — same layout as a batch of activations.
  * RDMs are returned as condensed upper-triangle vectors by default (scipy convention),
    because that is what every downstream comparison and permutation test consumes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr


def _validate(patterns: np.ndarray) -> np.ndarray:
    arr = np.asarray(patterns, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"patterns must be 2D (n_stimuli, n_features), got shape {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError(f"need at least 3 stimuli to form an RDM, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError("patterns contain NaN or inf")
    return arr


def compute_rdm(patterns: np.ndarray, metric: str = "correlation", condensed: bool = True):
    """Pairwise dissimilarity between stimulus response patterns.

    metric:
      'correlation' — 1 - Pearson r across features. The RSA default. It is invariant to
          per-stimulus additive and multiplicative scaling, which matters because overall
          fMRI response amplitude varies with attention and arousal in ways we do not
          want to model. Cost: it discards mean-response differences entirely.
      'euclidean'   — keeps amplitude information. Sensitive to feature scaling.
      'cosine'      — scale-invariant but not mean-centred; between the two above.

    Returns the condensed upper triangle (length n*(n-1)/2) unless condensed=False.
    """
    arr = _validate(patterns)

    if metric == "correlation":
        centred = arr - arr.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError(
                "at least one stimulus has zero variance across features; "
                "correlation distance is undefined for it"
            )
        unit = centred / norms
        sim = unit @ unit.T
        np.clip(sim, -1.0, 1.0, out=sim)
        full = 1.0 - sim
    elif metric == "cosine":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("zero-norm pattern; cosine distance undefined")
        unit = arr / norms
        sim = np.clip(unit @ unit.T, -1.0, 1.0)
        full = 1.0 - sim
    elif metric == "euclidean":
        sq = np.sum(arr**2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (arr @ arr.T)
        full = np.sqrt(np.maximum(d2, 0.0))
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    # Enforce exact symmetry and a zero diagonal; float error otherwise leaks into
    # squareform, which refuses matrices whose diagonal is not exactly zero.
    full = 0.5 * (full + full.T)
    np.fill_diagonal(full, 0.0)

    return squareform(full, checks=False) if condensed else full


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray, method: str = "spearman") -> float:
    """Similarity between two RDMs, both given as condensed upper triangles.

    Spearman is the RSA default: we expect the model-brain relationship to be monotonic
    but have no reason to believe it is linear, and rank correlation is robust to the
    very different distance scales of the two systems.
    """
    a = np.asarray(rdm_a, dtype=np.float64).ravel()
    b = np.asarray(rdm_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"RDM length mismatch: {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("RDMs contain NaN or inf")

    if method == "spearman":
        return float(spearmanr(a, b).statistic)
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "kendall":
        from scipy.stats import kendalltau

        return float(kendalltau(a, b).statistic)
    raise ValueError(f"unknown method: {method!r}")


def permutation_test(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_perm: int = 10_000,
    method: str = "spearman",
    seed: int = 0,
) -> tuple[float, float]:
    """Significance of an RDM correlation by permuting *stimulus labels*.

    Critical detail: we permute the stimulus ordering and rebuild the RDM, NOT the
    entries of the condensed vector. RDM entries are not exchangeable — each stimulus
    appears in n-1 of them, so shuffling entries destroys that dependency structure and
    produces null distributions that are far too narrow, i.e. false significance.

    Returns (observed_similarity, p_value), p one-sided (H1: positive similarity).
    """
    a_full = squareform(np.asarray(rdm_a, dtype=np.float64).ravel(), checks=False)
    b_cond = np.asarray(rdm_b, dtype=np.float64).ravel()
    n = a_full.shape[0]

    observed = compare_rdms(squareform(a_full, checks=False), b_cond, method=method)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        permuted = squareform(a_full[np.ix_(perm, perm)], checks=False)
        if compare_rdms(permuted, b_cond, method=method) >= observed:
            count += 1

    # +1 in numerator and denominator: the observed value is itself one draw from the
    # null under H0, and this keeps p strictly positive (never a meaningless p = 0).
    p_value = (count + 1) / (n_perm + 1)
    return observed, p_value


def bootstrap_ci(
    patterns_a: np.ndarray,
    patterns_b: np.ndarray,
    n_boot: int = 1000,
    metric: str = "correlation",
    method: str = "spearman",
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for an RDM correlation, resampling *stimuli* (not RDM entries).

    Resampling stimuli is what generalises the claim to "other images from this
    distribution", which is the inference we actually want to make.

    Returns (point_estimate, ci_low, ci_high).
    """
    a = _validate(patterns_a)
    b = _validate(patterns_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"stimulus count mismatch: {a.shape[0]} vs {b.shape[0]}")

    point = compare_rdms(compute_rdm(a, metric), compute_rdm(b, metric), method)

    rng = np.random.default_rng(seed)
    n = a.shape[0]
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 3:
            continue
        try:
            r = compare_rdms(
                compute_rdm(a[idx], metric), compute_rdm(b[idx], metric), method
            )
        except ValueError:
            # A resample can duplicate a stimulus enough to create a zero-variance
            # pattern; skip that draw rather than crashing the whole bootstrap.
            continue
        if np.isfinite(r):
            draws.append(r)

    if not draws:
        return point, float("nan"), float("nan")

    lo, hi = np.percentile(draws, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return point, float(lo), float(hi)


def rank_transform(rdm: np.ndarray) -> np.ndarray:
    """Rank-transform an RDM to [0, 1]. Useful for plotting RDMs on a common scale."""
    r = rankdata(np.asarray(rdm, dtype=np.float64).ravel())
    return (r - r.min()) / (r.max() - r.min())
