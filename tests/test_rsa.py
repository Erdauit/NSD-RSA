"""Tests for the batched RSA layer.

The matmul shortcut (Spearman as Pearson on ranks) is a performance optimisation on the
project's central computation. If it were subtly wrong, every number in the paper would
be wrong together and still look plausible — so it is checked against scipy directly.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from nsd_rsa.rdm import compute_rdm
from nsd_rsa.rsa import (
    RDMBank,
    best_layer_per_roi,
    compare_banks,
    layer_depth_index,
    rank_zscore,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_bank(rng, n_items=4, n_stim=25, n_feat=30):
    rdms = np.vstack([compute_rdm(rng.normal(size=(n_stim, n_feat))) for _ in range(n_items)])
    return RDMBank([f"item{i}" for i in range(n_items)], rdms)


# --- the shortcut must equal scipy ---------------------------------------------


def test_compare_banks_matches_scipy_spearman(rng):
    a, b = make_bank(rng, 3), make_bank(rng, 4)
    got = compare_banks(a, b)
    for i in range(len(a)):
        for j in range(len(b)):
            expected = spearmanr(a.rdms[i], b.rdms[j]).statistic
            assert got[i, j] == pytest.approx(expected, abs=1e-9)


def test_self_comparison_has_unit_diagonal(rng):
    a = make_bank(rng, 5)
    assert np.allclose(np.diag(compare_banks(a, a)), 1.0, atol=1e-9)


def test_correlations_stay_in_range(rng):
    out = compare_banks(make_bank(rng, 6), make_bank(rng, 6))
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_shared_structure_is_detected(rng):
    """Two noisy views of one latent representation must correlate positively."""
    latent = rng.normal(size=(40, 6))
    a = RDMBank(["a"], compute_rdm(latent @ rng.normal(size=(6, 50)))[None])
    b = RDMBank(["b"], compute_rdm(latent @ rng.normal(size=(6, 30)))[None])
    assert compare_banks(a, b)[0, 0] > 0.5


# --- degenerate inputs must not poison a whole heatmap --------------------------


def test_constant_rdm_yields_zero_not_nan(rng):
    good = make_bank(rng, 2)
    dead = RDMBank(["dead"], np.ones((1, good.n_pairs), dtype=np.float32))
    out = compare_banks(dead, good)
    assert np.isfinite(out).all()
    assert np.allclose(out, 0.0)


def test_rank_zscore_is_unit_variance(rng):
    z = rank_zscore(make_bank(rng, 3).rdms)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-9)


def test_mismatched_stimulus_sets_raise(rng):
    a = make_bank(rng, 2, n_stim=20)
    b = make_bank(rng, 2, n_stim=25)
    with pytest.raises(ValueError, match="different stimulus sets"):
        compare_banks(a, b)


# --- helpers -------------------------------------------------------------------


def test_best_layer_per_roi_picks_the_maximum():
    scores = np.array([[0.1, 0.9], [0.5, 0.2], [0.3, 0.4]])
    vals, names = best_layer_per_roi(scores, ["l0", "l1", "l2"])
    assert list(vals) == [0.5, 0.9]
    assert names == ["l1", "l0"]


def test_layer_depth_index_spans_unit_interval():
    labels = ["block00.cls", "block00.patchmean", "block01.cls", "block02.cls"]
    depth = layer_depth_index(labels)
    assert depth.min() == 0.0
    assert depth.max() == 1.0
    # Both readouts of the same block sit at the same depth.
    assert depth[0] == depth[1]


def test_layer_depth_index_handles_single_layer():
    assert list(layer_depth_index(["stage1.gap"])) == [0.0]


def test_bank_subset_preserves_order():
    bank = RDMBank(["a", "b", "c"], np.arange(12).reshape(3, 4).astype(np.float32))
    sub = bank.subset(["c", "a"])
    assert sub.labels == ["c", "a"]
    assert np.array_equal(sub.rdms[0], bank.rdms[2])
