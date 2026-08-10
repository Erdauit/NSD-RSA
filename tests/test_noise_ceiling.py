"""Tests for noise-ceiling estimation.

Every headline number in the paper is a ratio to one of these ceilings, so an error here
would rescale all results at once — and plausibly, which is why it needs testing rather
than eyeballing.
"""

import numpy as np
import pytest

from nsd_rsa.loaders import SubjectBetas, average_repeats, split_half
from nsd_rsa.noise_ceiling import (
    normalise_to_ceiling,
    rsa_noise_ceiling,
    split_half_reliability,
    voxel_noise_ceiling,
)
from nsd_rsa.rdm import compute_rdm

# --- voxel noise ceiling -------------------------------------------------------


def test_zero_ncsnr_gives_zero_ceiling():
    assert voxel_noise_ceiling(np.array([0.0]), 3)[0] == 0.0


def test_ceiling_increases_with_repeats():
    """Averaging more repeats genuinely leaves more explainable variance."""
    ncsnr = np.array([0.5])
    vals = [voxel_noise_ceiling(ncsnr, n)[0] for n in (1, 2, 3, 10)]
    assert vals == sorted(vals)
    assert vals[0] < vals[-1]


def test_ceiling_approaches_100_for_high_snr():
    assert voxel_noise_ceiling(np.array([50.0]), 3)[0] > 99.9


def test_known_value_unit_snr_single_trial():
    """ncsnr=1, n=1 -> 1/(1+1) = 50%."""
    assert voxel_noise_ceiling(np.array([1.0]), 1)[0] == pytest.approx(50.0)


def test_ceiling_is_bounded():
    v = voxel_noise_ceiling(np.linspace(0, 20, 200), 3)
    assert v.min() >= 0.0
    assert v.max() <= 100.0


def test_negative_ncsnr_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        voxel_noise_ceiling(np.array([-0.1]), 3)


def test_zero_repeats_rejected():
    with pytest.raises(ValueError, match="n_repeats"):
        voxel_noise_ceiling(np.array([1.0]), 0)


# --- split-half reliability ----------------------------------------------------


def test_identical_halves_are_perfectly_reliable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 10))
    assert np.allclose(split_half_reliability(x, x, spearman_brown=False), 1.0)


def test_pure_noise_halves_are_unreliable():
    rng = np.random.default_rng(0)
    r = split_half_reliability(
        rng.normal(size=(200, 300)), rng.normal(size=(200, 300)), spearman_brown=False
    )
    assert abs(np.nanmean(r)) < 0.05


def test_reliability_tracks_signal_strength():
    """More stimulus signal relative to noise must yield higher reliability."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(200, 100))
    means = []
    for noise_sd in (0.2, 1.0, 4.0):
        a = signal + noise_sd * rng.normal(size=signal.shape)
        b = signal + noise_sd * rng.normal(size=signal.shape)
        means.append(np.nanmean(split_half_reliability(a, b, spearman_brown=False)))
    assert means[0] > means[1] > means[2]


def test_spearman_brown_raises_the_estimate():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(150, 50))
    a, b = sig + rng.normal(size=sig.shape), sig + rng.normal(size=sig.shape)
    raw = np.nanmean(split_half_reliability(a, b, spearman_brown=False))
    corrected = np.nanmean(split_half_reliability(a, b, spearman_brown=True))
    assert corrected > raw


def test_constant_vertex_yields_nan_not_crash():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 4))
    b = rng.normal(size=(30, 4))
    a[:, 2] = 5.0
    r = split_half_reliability(a, b, spearman_brown=False)
    assert np.isnan(r[2])
    assert np.isfinite(r[[0, 1, 3]]).all()


def test_mismatched_halves_rejected():
    with pytest.raises(ValueError, match="halves must match"):
        split_half_reliability(np.zeros((10, 5)), np.zeros((10, 6)))


# --- RSA noise ceiling ---------------------------------------------------------


def make_subject_rdms(n_subjects=8, n_stim=30, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n_stim, 20))
    return np.vstack(
        [compute_rdm(shared + noise * rng.normal(size=shared.shape)) for _ in range(n_subjects)]
    )


def test_lower_bound_is_below_upper_bound():
    lo, hi = rsa_noise_ceiling(make_subject_rdms())
    assert lo < hi


def test_ceiling_is_high_when_subjects_agree():
    lo, hi = rsa_noise_ceiling(make_subject_rdms(noise=0.05))
    assert lo > 0.9 and hi > 0.9


def test_ceiling_is_low_when_subjects_disagree():
    rng = np.random.default_rng(1)
    rdms = np.vstack([compute_rdm(rng.normal(size=(30, 20))) for _ in range(8)])
    lo, hi = rsa_noise_ceiling(rdms)
    assert lo < 0.2


def test_ceiling_rises_as_noise_falls():
    los = [rsa_noise_ceiling(make_subject_rdms(noise=n))[0] for n in (2.0, 1.0, 0.2)]
    assert los == sorted(los)


def test_too_few_subjects_rejected():
    with pytest.raises(ValueError, match="at least 3 subjects"):
        rsa_noise_ceiling(make_subject_rdms(n_subjects=2))


# --- normalisation guard -------------------------------------------------------


def test_normalise_is_a_plain_ratio():
    assert normalise_to_ceiling(0.15, 0.30) == pytest.approx(0.5)


def test_normalise_refuses_near_zero_ceiling():
    """Dividing by a near-zero ceiling is how a dead ROI turns into a model's best score."""
    assert np.isnan(normalise_to_ceiling(0.05, 0.001))
    assert np.isnan(normalise_to_ceiling(0.05, 0.0))


# --- loaders -------------------------------------------------------------------


def make_betas(n_images=20, n_vertices=40, n_repeats=3, seed=0) -> SubjectBetas:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_images, n_vertices))
    rows, image, repeat = [], [], []
    for img in range(n_images):
        for rep in range(n_repeats):
            rows.append(signal[img] + rng.normal(size=n_vertices))
            image.append(img)
            repeat.append(rep)
    return SubjectBetas(
        subject="test",
        betas=np.asarray(rows, dtype=np.float32),
        image=np.asarray(image),
        repeat=np.asarray(repeat),
        session=np.ones(len(rows), dtype=int),
        vertex_index=np.arange(n_vertices),
        roi_label=np.where(np.arange(n_vertices) < n_vertices // 2, 1, 5),
        lh_size=n_vertices // 2,
    )


def test_average_repeats_collapses_to_one_row_per_image():
    d = make_betas()
    patterns, images = average_repeats(d)
    assert patterns.shape == (20, 40)
    assert list(images) == list(range(20))


def test_average_repeats_actually_averages():
    d = make_betas(n_images=3, n_vertices=5)
    patterns, _ = average_repeats(d)
    assert np.allclose(patterns[0], d.betas[d.image == 0].mean(axis=0), atol=1e-6)


def test_average_repeats_can_restrict_to_roi():
    d = make_betas()
    patterns, _ = average_repeats(d, roi="early")
    assert patterns.shape[1] == int((d.roi_label == 1).sum())


def test_split_half_returns_disjoint_averages():
    d = make_betas()
    a, b, images = split_half(d, seed=0)
    assert a.shape == b.shape == (20, 40)
    assert len(images) == 20
    assert not np.allclose(a, b)  # independent noise must differ


def test_split_half_of_noisy_data_is_positively_reliable():
    """End-to-end: the synthetic data has real shared signal, so reliability must be > 0."""
    d = make_betas(n_images=100, n_vertices=60)
    a, b, _ = split_half(d, seed=0)
    assert np.nanmean(split_half_reliability(a, b)) > 0.2


def test_split_half_drops_singleton_images():
    d = make_betas(n_images=5, n_repeats=1)
    with pytest.raises(ValueError, match="no image has 2"):
        split_half(d)
