"""Tests for S3 access and the MGH byte-layout assumptions behind `range_read`.

`range_read` is only correct because NSD's fsaverage betas are uncompressed and stored
frame-contiguously. That is an assumption about someone else's file format, so it is
asserted in code and tested here. If it ever stops holding, range reads would return
plausible-looking wrong numbers rather than failing — the worst kind of bug.

Network tests are marked and skipped when the bucket is unreachable.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from nsd_rsa import nsd


def _bucket_reachable() -> bool:
    try:
        nsd.get_client().list_objects_v2(
            Bucket=nsd.BUCKET, Prefix="nsddata/experiments/nsd/", MaxKeys=1
        )
        return True
    except Exception:  # noqa: BLE001
        return False


needs_net = pytest.mark.skipif(not _bucket_reachable(), reason="NSD S3 unreachable")


# --- key construction (pure, no network) ---------------------------------------


def test_betas_key_zero_pads_session():
    key = nsd.betas_key("subj01", 7, "lh", "fsaverage", "betas_fithrf_GLMdenoise_RR")
    assert key.endswith("lh.betas_session07.mgh")


def test_streams_key_defaults_to_fsaverage():
    assert "fsaverage/label/lh.streams.mgz" in nsd.streams_key("lh")


def test_ncsnr_key_handles_splits():
    base = nsd.ncsnr_key("subj01", "lh", "fsaverage", "v")
    split = nsd.ncsnr_key("subj01", "lh", "fsaverage", "v", "split1")
    assert base.endswith("lh.ncsnr.mgh")
    assert split.endswith("lh.ncsnr_split1.mgh")


# --- MGH header parsing (pure) -------------------------------------------------


def make_header(width=163842, nframes=750, mgh_type=3) -> bytes:
    return struct.pack(">6i", 1, width, 1, 1, nframes, mgh_type) + b"\0" * 260


def test_parse_header_reads_float32_surface_layout():
    h = nsd.parse_mgh_header(make_header())
    assert h["dtype"] is np.float32
    assert h["nframes"] == 750
    assert h["frame_bytes"] == 163842 * 4


def test_parse_header_rejects_unknown_version():
    bad = struct.pack(">6i", 42, 10, 1, 1, 5, 3) + b"\0" * 260
    with pytest.raises(ValueError, match="version"):
        nsd.parse_mgh_header(bad)


def test_parse_header_rejects_unknown_datatype():
    with pytest.raises(ValueError, match="datatype"):
        nsd.parse_mgh_header(make_header(mgh_type=99))


# --- layout validation guards --------------------------------------------------


def test_validate_layout_rejects_size_mismatch(monkeypatch):
    """A compressed file would be far smaller than the contiguous model predicts.
    We must refuse rather than range-read garbage."""
    header = nsd.parse_mgh_header(make_header())
    monkeypatch.setitem(nsd._SIZE_CACHE, "fake-key", 1234)
    with pytest.raises(ValueError, match="layout assumption violated"):
        nsd.validate_mgh_layout("fake-key", header)


def test_validate_layout_accepts_expected_size(monkeypatch):
    header = nsd.parse_mgh_header(make_header())
    exact = nsd.MGH_HEADER_BYTES + 750 * 163842 * 4
    monkeypatch.setitem(nsd._SIZE_CACHE, "ok-key", exact + 16)  # + footer
    nsd.validate_mgh_layout("ok-key", header)


# --- network -------------------------------------------------------------------


@needs_net
def test_real_betas_file_is_float32_and_contiguous():
    """The measurement the whole download strategy rests on."""
    key = nsd.betas_key("subj01", 1, "lh", "fsaverage", "betas_fithrf_GLMdenoise_RR")
    header = nsd.read_mgh_header_s3(key)
    assert header["dtype"] is np.float32
    assert header["nframes"] == 750
    assert header["width"] == 163842
    nsd.validate_mgh_layout(key, header)  # raises if not contiguous


@needs_net
def test_range_read_matches_local_read_bitwise():
    """The correctness proof for range_read, run against a full local copy if one exists.

    This is the check that licensed the 10x transfer saving; it is kept as a test so the
    claim stays falsifiable rather than living only in a commit message.
    """
    local = Path(__file__).resolve().parents[1] / "data/raw/lh.betas_session01.mgh"
    if not local.exists():
        pytest.skip("no full session file on disk to compare against")

    key = nsd.betas_key("subj01", 1, "lh", "fsaverage", "betas_fithrf_GLMdenoise_RR")
    frames = np.array([0, 28, 137, 499, 749])
    truth = nsd.read_mgh_local(local)[frames]
    ranged = nsd.read_mgh_frames_ranged(key, frames)
    assert np.array_equal(truth, ranged)


@needs_net
def test_range_read_rejects_out_of_range_frame():
    key = nsd.betas_key("subj01", 1, "lh", "fsaverage", "betas_fithrf_GLMdenoise_RR")
    with pytest.raises(IndexError):
        nsd.read_mgh_frames_ranged(key, np.array([750]))


# --- retry behaviour -----------------------------------------------------------


def test_retry_returns_on_first_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert nsd.with_retry(fn) == "ok"
    assert len(calls) == 1


def test_retry_recovers_from_transient_failure(monkeypatch):
    """A read timeout mid-download killed a 40-session run once; it must not again."""
    from botocore.exceptions import ReadTimeoutError

    monkeypatch.setattr(nsd.time, "sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ReadTimeoutError(endpoint_url="s3://x")
        return "recovered"

    assert nsd.with_retry(flaky) == "recovered"
    assert len(calls) == 3


def test_retry_gives_up_and_reports(monkeypatch):
    from botocore.exceptions import ReadTimeoutError

    monkeypatch.setattr(nsd.time, "sleep", lambda _s: None)

    def always_fails():
        raise ReadTimeoutError(endpoint_url="s3://x")

    with pytest.raises(RuntimeError, match="giving up after 3 attempts"):
        nsd.with_retry(always_fails, attempts=3)


def test_retry_does_not_mask_programming_errors(monkeypatch):
    """Retrying a KeyError would waste minutes hiding a real bug."""
    monkeypatch.setattr(nsd.time, "sleep", lambda _s: None)
    calls = []

    def bad():
        calls.append(1)
        raise KeyError("typo in a key name")

    with pytest.raises(KeyError):
        nsd.with_retry(bad)
    assert len(calls) == 1


def test_short_read_is_treated_as_transient():
    """A truncated body must be retried, never written into the array as partial data."""
    assert nsd._is_transient(OSError("short read for frame 3"))
