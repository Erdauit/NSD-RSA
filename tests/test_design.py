"""Tests for experimental-design bookkeeping.

A silent error here mislabels which image a brain response belongs to, which would
corrupt every downstream result while leaving all the numbers looking plausible. So
these tests are about label correctness, not numerics.

Synthetic designs are used so the tests run without the 260 KB real file; a separate
integration test checks the real one when present.
"""

from pathlib import Path

import numpy as np
import pytest

from nsd_rsa.design import (
    N_SHARED,
    SESSIONS_PER_SUBJECT,
    TRIALS_PER_SESSION,
    load_expdesign,
    shared_trials,
    usable_images,
    verify_expdesign,
)

REAL_DESIGN = Path(__file__).resolve().parents[1] / "data/meta/nsd_expdesign.mat"
needs_real = pytest.mark.skipif(not REAL_DESIGN.exists(), reason="NSD metadata not downloaded")


def make_design(seed: int = 0) -> dict:
    """A design with NSD's structure: shared images occupy slots 1..1000 and repeat 3x."""
    rng = np.random.default_rng(seed)
    order = np.repeat(np.arange(1, 10_001), 3)
    rng.shuffle(order)
    sharedix = np.arange(1, N_SHARED + 1) * 7  # arbitrary distinct image ids
    subjectim = np.tile(np.arange(1, 10_001) * 7, (8, 1))
    return {
        "subjectim": subjectim,
        "masterordering": order.astype(np.uint16),
        "sharedix": sharedix.astype(np.int32),
    }


# --- structural verification ---------------------------------------------------


def test_verify_accepts_wellformed_design():
    verify_expdesign(make_design())


def test_verify_rejects_broken_shared_prefix():
    d = make_design()
    d["subjectim"] = d["subjectim"].copy()
    d["subjectim"][3, 0] = 999_999
    with pytest.raises(ValueError, match="subjectim"):
        verify_expdesign(d)


def test_verify_rejects_wrong_repeat_count():
    d = make_design()
    mo = d["masterordering"].copy()
    mo[mo == 1] = 2  # image 1 now has 0 repeats, image 2 has 6
    d["masterordering"] = mo
    with pytest.raises(ValueError, match="3 repeats"):
        verify_expdesign(d)


# --- trial location ------------------------------------------------------------


def test_complete_subject_gets_all_shared_presentations():
    t = shared_trials("subj01", make_design())
    assert len(t) == N_SHARED * 3
    assert set(np.unique(t.image)) == set(range(N_SHARED))
    assert (t.repeats_per_image() == 3).all()


def test_frames_and_sessions_are_in_range():
    t = shared_trials("subj01", make_design())
    assert t.frame.min() >= 0
    assert t.frame.max() < TRIALS_PER_SESSION
    assert t.session.min() >= 1
    assert t.session.max() <= SESSIONS_PER_SUBJECT["subj01"]


def test_repeat_counter_is_sequential_per_image():
    """Repeat indices must be 0,1,2 in presentation order — not arbitrary labels,
    because split-half reliability splits on them."""
    t = shared_trials("subj01", make_design())
    for img in (0, 17, 999):
        sel = t.image == img
        order = np.argsort(t.session[sel] * TRIALS_PER_SESSION + t.frame[sel])
        assert list(t.repeat[sel][order]) == [0, 1, 2]


def test_short_subject_is_truncated_not_padded():
    """subj04 ran 30 of 40 sessions, so it must simply lack the design's tail."""
    d = make_design()
    t = shared_trials("subj04", d)
    assert t.session.max() <= 30
    assert len(t) < N_SHARED * 3
    full = shared_trials("subj01", d)
    # Its trials are a prefix of the complete subject's, since the design is fixed.
    assert len(t) == int(((full.session <= 30)).sum())


def test_unknown_subject_raises():
    with pytest.raises(KeyError):
        shared_trials("subj99", make_design())


# --- stimulus-set selection ----------------------------------------------------


def test_usable_images_is_monotonic_in_min_repeats():
    d = make_design()
    subs = list(SESSIONS_PER_SUBJECT)
    sizes = [len(usable_images(subs, d, m)) for m in (1, 2, 3)]
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_complete_subjects_have_all_images_at_three_repeats():
    d = make_design()
    assert len(usable_images(["subj01", "subj02", "subj05", "subj07"], d, 3)) == N_SHARED


# --- the real file -------------------------------------------------------------


@needs_real
def test_real_design_passes_verification():
    verify_expdesign(load_expdesign(REAL_DESIGN))


@needs_real
def test_real_design_reproduces_measured_subset_sizes():
    """These three numbers drive the stimulus-set decision in configs/data.yaml.
    If NSD ever revises the design, this test should fail loudly rather than let the
    analysis silently change size."""
    d = load_expdesign(REAL_DESIGN)
    subs = list(SESSIONS_PER_SUBJECT)
    assert len(usable_images(subs, d, 1)) == 907
    assert len(usable_images(subs, d, 2)) == 766
    assert len(usable_images(subs, d, 3)) == 515


@needs_real
def test_real_subj01_has_3000_shared_trials():
    t = shared_trials("subj01", load_expdesign(REAL_DESIGN))
    assert len(t) == 3000
    assert (t.repeats_per_image() == 3).all()
