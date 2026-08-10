"""Noise ceilings — the upper bound on what any model could possibly explain.

The idea in ML terms: your data has irreducible label noise, so even a perfect model
cannot reach a correlation of 1.0. The noise ceiling estimates that limit from the
repeated measurements. Reporting a raw RSA correlation of 0.25 is meaningless on its own;
reporting that it is 0.25 against a ceiling of 0.30 says the model captures most of the
explainable structure. Without this normalisation the numbers cannot be compared across
ROIs (which differ in SNR), across subjects, or against published work.

Two different ceilings appear in this project, and they answer different questions:

  * `voxel_noise_ceiling` — per-vertex, from NSD's ncsnr. Used for the encoding model
    (S5), where the target is each vertex's response.
  * `rsa_noise_ceiling` — for RDM comparisons across subjects (S3). Used for RSA, where
    the target is the *structure* of the representation.
"""

from __future__ import annotations

import numpy as np

from nsd_rsa.rdm import compare_rdms


def voxel_noise_ceiling(ncsnr: np.ndarray, n_repeats: int = 3) -> np.ndarray:
    """Fraction of a vertex's variance that is stimulus-driven, as a percentage.

    NSD supplies `ncsnr`, the ratio of signal standard deviation to noise standard
    deviation, estimated from repeated presentations. When responses are averaged over
    n repeats, the noise variance shrinks by n while the signal stays put, so

        NC = ncsnr^2 / (ncsnr^2 + 1/n)

    Averaging more repeats therefore raises the ceiling — a real effect, not a trick:
    a cleaner measurement genuinely leaves more variance available to explain.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    ncsnr = np.asarray(ncsnr, dtype=np.float64)
    if (ncsnr < 0).any():
        raise ValueError("ncsnr must be non-negative")
    sq = ncsnr**2
    return 100.0 * sq / (sq + 1.0 / n_repeats)


def split_half_reliability(
    half_a: np.ndarray, half_b: np.ndarray, spearman_brown: bool = True
) -> np.ndarray:
    """Per-vertex correlation between two independent halves of the repeats.

    Each vertex gets one correlation across images: does this vertex respond to the same
    pictures the same way on independent measurements? Values near 0 mean the vertex
    carries no reproducible stimulus signal.

    The Spearman-Brown correction rescales the half-vs-half correlation to estimate the
    reliability of the *full* dataset, since each half uses only half the trials:
        r_full = 2r / (1 + r)
    """
    a = np.asarray(half_a, dtype=np.float64)
    b = np.asarray(half_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"halves must match: {a.shape} vs {b.shape}")
    if a.shape[0] < 3:
        raise ValueError("need at least 3 images to correlate halves")

    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, (a * b).sum(axis=0) / denom, np.nan)

    if spearman_brown:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(r > -1, 2 * r / (1 + r), np.nan)
        r = np.clip(r, -1.0, 1.0)
    return r


def rsa_noise_ceiling(
    rdms: np.ndarray, method: str = "spearman"
) -> tuple[float, float]:
    """Lower and upper bounds on the RSA correlation any model could achieve.

    Following the standard construction (Nili et al., 2014). Given one RDM per subject:

      upper — correlate each subject's RDM with the mean RDM of ALL subjects, including
              themselves. Because each subject contributes to the target, this
              overestimates: it is the score of a model that has partly seen the answer.
      lower — correlate each subject's RDM with the mean of the OTHERS. Leave-one-out, so
              it slightly underestimates.

    The true ceiling lies between them. A model landing inside the band is doing as well
    as the data permit; the band's width reflects how variable subjects are.

    `rdms` is (n_subjects, n_pairs) of condensed upper triangles.
    """
    rdms = np.asarray(rdms, dtype=np.float64)
    if rdms.ndim != 2:
        raise ValueError(f"expected (n_subjects, n_pairs), got shape {rdms.shape}")
    n = rdms.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 subjects for a leave-one-out ceiling, got {n}")

    group_mean = rdms.mean(axis=0)
    upper = np.mean([compare_rdms(r, group_mean, method) for r in rdms])

    lower_vals = []
    for i in range(n):
        others = np.delete(rdms, i, axis=0).mean(axis=0)
        lower_vals.append(compare_rdms(rdms[i], others, method))
    lower = float(np.mean(lower_vals))

    return lower, float(upper)


def normalise_to_ceiling(score: float, ceiling: float) -> float:
    """Express a raw RSA score as a fraction of the achievable ceiling.

    Kept as a named function because the failure mode matters: dividing by a ceiling near
    zero produces enormous meaningless numbers, which is exactly how ROIs with almost no
    signal end up looking like a model's best result.
    """
    if not np.isfinite(ceiling) or ceiling <= 0.01:
        return float("nan")
    return float(score / ceiling)
