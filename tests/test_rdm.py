"""Tests for the RSA core. These are the functions every result depends on, so they get
real tests: known-answer checks, invariances that must hold, and failure modes that must raise.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from nsd_rsa.rdm import (
    bootstrap_ci,
    compare_rdms,
    compute_rdm,
    permutation_test,
    rank_transform,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- shape and basic structure -------------------------------------------------


def test_condensed_length(rng):
    n = 20
    rdm = compute_rdm(rng.normal(size=(n, 50)))
    assert rdm.shape == (n * (n - 1) // 2,)


def test_square_form_has_zero_diagonal(rng):
    full = compute_rdm(rng.normal(size=(10, 30)), condensed=False)
    assert full.shape == (10, 10)
    assert np.allclose(np.diag(full), 0.0)
    assert np.allclose(full, full.T)


# --- correctness against scipy -------------------------------------------------


@pytest.mark.parametrize("metric", ["correlation", "euclidean", "cosine"])
def test_matches_scipy_pdist(rng, metric):
    patterns = rng.normal(size=(15, 40))
    assert np.allclose(compute_rdm(patterns, metric), pdist(patterns, metric=metric), atol=1e-9)


def test_correlation_distance_of_identical_patterns_is_zero():
    p = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    full = compute_rdm(p, "correlation", condensed=False)
    assert full[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_correlation_distance_of_anticorrelated_patterns_is_two():
    p = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 1.0, 5.0]])
    full = compute_rdm(p, "correlation", condensed=False)
    assert full[0, 1] == pytest.approx(2.0, abs=1e-12)


# --- invariances that justify the metric choice --------------------------------


def test_correlation_rdm_invariant_to_per_stimulus_affine_rescaling(rng):
    """The reason we use 1 - Pearson: per-stimulus gain and offset must not matter,
    because overall fMRI response amplitude fluctuates with arousal and attention."""
    patterns = rng.normal(size=(12, 60))
    gains = rng.uniform(0.5, 3.0, size=(12, 1))
    offsets = rng.normal(size=(12, 1)) * 5
    rescaled = patterns * gains + offsets
    assert np.allclose(compute_rdm(patterns), compute_rdm(rescaled), atol=1e-9)


def test_euclidean_rdm_is_not_invariant_to_rescaling(rng):
    """Contrast case — documents what we give up by not choosing euclidean."""
    patterns = rng.normal(size=(12, 60))
    rescaled = patterns * 3.0 + 1.0
    assert not np.allclose(compute_rdm(patterns, "euclidean"), compute_rdm(rescaled, "euclidean"))


def test_rdm_invariant_to_orthogonal_rotation_of_features(rng):
    """RDMs see only the geometry of the stimulus configuration, not the basis it is
    expressed in. This is exactly why a 768-d model and a 4000-vertex ROI are comparable."""
    patterns = rng.normal(size=(10, 25))
    q, _ = np.linalg.qr(rng.normal(size=(25, 25)))
    assert np.allclose(
        compute_rdm(patterns, "euclidean"), compute_rdm(patterns @ q, "euclidean"), atol=1e-9
    )


# --- comparison ----------------------------------------------------------------


def test_identical_rdms_correlate_perfectly(rng):
    rdm = compute_rdm(rng.normal(size=(20, 30)))
    assert compare_rdms(rdm, rdm) == pytest.approx(1.0)


def test_spearman_is_invariant_to_monotonic_transform(rng):
    """Justifies Spearman over Pearson: model and brain distances live on different
    scales, and we only claim the relationship is monotonic."""
    a = compute_rdm(rng.normal(size=(20, 30)))
    b = compute_rdm(rng.normal(size=(20, 30)))
    assert compare_rdms(a, b) == pytest.approx(compare_rdms(np.exp(a * 2), b), abs=1e-9)


def test_length_mismatch_raises(rng):
    with pytest.raises(ValueError, match="length mismatch"):
        compare_rdms(compute_rdm(rng.normal(size=(10, 5))), compute_rdm(rng.normal(size=(12, 5))))


def test_shared_structure_is_detected(rng):
    """Two noisy views of the same underlying representation must correlate positively."""
    latent = rng.normal(size=(40, 8))
    view_a = latent @ rng.normal(size=(8, 100)) + 0.1 * rng.normal(size=(40, 100))
    view_b = latent @ rng.normal(size=(8, 60)) + 0.1 * rng.normal(size=(40, 60))
    assert compare_rdms(compute_rdm(view_a), compute_rdm(view_b)) > 0.5


# --- inference -----------------------------------------------------------------


def test_permutation_test_rejects_for_shared_structure(rng):
    latent = rng.normal(size=(30, 5))
    a = latent @ rng.normal(size=(5, 50))
    b = latent @ rng.normal(size=(5, 40)) + 0.2 * rng.normal(size=(30, 40))
    observed, p = permutation_test(compute_rdm(a), compute_rdm(b), n_perm=500, seed=0)
    assert observed > 0.4
    assert p < 0.01


def test_permutation_test_does_not_reject_for_independent_data(rng):
    a = compute_rdm(rng.normal(size=(30, 50)))
    b = compute_rdm(rng.normal(size=(30, 50)))
    _, p = permutation_test(a, b, n_perm=500, seed=0)
    assert p > 0.05


def test_permutation_p_value_is_never_zero(rng):
    """(count + 1) / (n_perm + 1) guarantees a strictly positive p."""
    rdm = compute_rdm(rng.normal(size=(15, 20)))
    _, p = permutation_test(rdm, rdm, n_perm=50, seed=0)
    assert p > 0


def test_bootstrap_ci_brackets_point_estimate(rng):
    latent = rng.normal(size=(50, 6))
    a = latent @ rng.normal(size=(6, 40))
    b = latent @ rng.normal(size=(6, 30)) + 0.3 * rng.normal(size=(50, 30))
    point, lo, hi = bootstrap_ci(a, b, n_boot=200, seed=0)
    assert lo <= point <= hi
    assert lo > 0


def test_bootstrap_is_deterministic_under_seed(rng):
    a = rng.normal(size=(30, 20))
    b = rng.normal(size=(30, 20))
    assert bootstrap_ci(a, b, n_boot=100, seed=7) == bootstrap_ci(a, b, n_boot=100, seed=7)


# --- input validation ----------------------------------------------------------


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="must be 2D"):
        compute_rdm(np.arange(10.0))


def test_rejects_too_few_stimuli():
    with pytest.raises(ValueError, match="at least 3 stimuli"):
        compute_rdm(np.random.default_rng(0).normal(size=(2, 10)))


def test_rejects_nan():
    p = np.random.default_rng(0).normal(size=(5, 10))
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_rdm(p)


def test_rejects_zero_variance_pattern_for_correlation():
    p = np.random.default_rng(0).normal(size=(5, 10))
    p[2, :] = 1.0
    with pytest.raises(ValueError, match="zero variance"):
        compute_rdm(p, "correlation")


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        compute_rdm(np.random.default_rng(0).normal(size=(5, 10)), "mahalanobis")


# --- helpers -------------------------------------------------------------------


def test_rank_transform_spans_unit_interval(rng):
    out = rank_transform(compute_rdm(rng.normal(size=(15, 20))))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_squareform_roundtrip(rng):
    rdm = compute_rdm(rng.normal(size=(12, 30)))
    assert np.allclose(squareform(squareform(rdm)), rdm)
