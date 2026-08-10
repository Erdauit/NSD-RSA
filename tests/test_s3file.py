"""Tests for the S3-backed file object.

This object exists so h5py can read 1000 images out of a 39.6 GB remote file without
transferring the rest. Its correctness is byte-level: a seek/read bug would hand HDF5
wrong offsets, and HDF5 would decode whatever it found into plausible-looking image data.
"""

import io

import numpy as np
import pytest

from nsd_rsa.s3file import S3File


class FakeClient:
    """In-memory stand-in for boto3's S3 client, recording every range served."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.ranges: list[tuple[int, int]] = []

    def head_object(self, Bucket, Key):  # noqa: N803 — boto3 API casing
        return {"ContentLength": len(self.payload)}

    def get_object(self, Bucket, Key, Range):  # noqa: N803
        start, end = Range.removeprefix("bytes=").split("-")
        start, end = int(start), int(end)
        self.ranges.append((start, end))
        return {"Body": io.BytesIO(self.payload[start : end + 1])}

    @property
    def bytes_served(self) -> int:
        return sum(e - s + 1 for s, e in self.ranges)


@pytest.fixture
def payload():
    return np.random.default_rng(0).integers(0, 256, size=300_000, dtype=np.uint8).tobytes()


@pytest.fixture
def f(payload):
    return S3File(FakeClient(payload), "bucket", "key", block_size=4096, passthrough=8192)


# --- basic reading -------------------------------------------------------------


def test_reports_object_size(f, payload):
    assert f.size == len(payload)


def test_sequential_read_matches_source(f, payload):
    assert f.read(1000) == payload[:1000]
    assert f.read(1000) == payload[1000:2000]


def test_read_all_matches_source(f, payload):
    assert f.read() == payload


def test_seek_absolute_then_read(f, payload):
    f.seek(50_000)
    assert f.read(256) == payload[50_000:50_256]


def test_seek_relative_and_from_end(f, payload):
    f.seek(100)
    f.seek(50, io.SEEK_CUR)
    assert f.tell() == 150
    f.seek(-10, io.SEEK_END)
    assert f.read() == payload[-10:]


def test_read_past_end_is_truncated_not_an_error(f, payload):
    f.seek(len(payload) - 5)
    assert f.read(1000) == payload[-5:]
    assert f.read(10) == b""


def test_invalid_whence_rejected(f):
    with pytest.raises(ValueError, match="whence"):
        f.seek(0, 99)


def test_readinto_fills_buffer(f, payload):
    buf = bytearray(64)
    assert f.readinto(buf) == 64
    assert bytes(buf) == payload[:64]


def test_declares_capabilities(f):
    assert f.readable() and f.seekable() and not f.writable()


# --- transfer efficiency -------------------------------------------------------


def test_small_reads_are_coalesced_into_one_request(payload):
    """HDF5 metadata lookups are tiny and numerous; without read-ahead each would be a
    separate round trip on a high-latency link."""
    f = S3File(FakeClient(payload), "b", "k", block_size=4096, passthrough=8192)
    for _ in range(8):
        f.read(64)
    assert f.n_requests == 1


def test_large_reads_bypass_the_cache(payload):
    """A payload-sized read must transfer exactly what was asked for. When the
    passthrough threshold was above image size, every image dragged a full block along
    and roughly doubled the transfer."""
    f = S3File(FakeClient(payload), "b", "k", block_size=65536, passthrough=8192)
    f.read(20_000)
    assert f.n_requests == 1
    assert f.n_bytes == 20_000


def test_no_wasted_bytes_on_repeated_large_reads(payload):
    f = S3File(FakeClient(payload), "b", "k", block_size=65536, passthrough=8192)
    for _ in range(10):
        f.read(10_000)
    assert f.n_bytes == 100_000


def test_cache_is_invalidated_on_seek_away(f, payload):
    f.read(64)
    f.seek(200_000)
    assert f.read(64) == payload[200_000:200_064]


# --- integration with h5py -----------------------------------------------------


def test_h5py_can_read_through_it(tmp_path):
    """The real requirement: HDF5 must be able to open and slice a file it only sees
    through range requests."""
    h5py = pytest.importorskip("h5py")

    local = tmp_path / "sample.h5"
    # Must be well above the read-ahead block, or "we transferred less than the whole
    # file" is trivially false and the test proves nothing. The real file is 39.6 GB.
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(2000, 512), dtype=np.uint8)
    with h5py.File(local, "w") as h:
        h.create_dataset("imgBrick", data=data)

    client = FakeClient(local.read_bytes())
    assert len(client.payload) > 500_000, "sample file too small to test partial reads"

    f = S3File(client, "b", "k", block_size=4096, passthrough=8192)
    with h5py.File(f, "r") as h:
        assert h["imgBrick"].shape == (2000, 512)
        assert np.array_equal(h["imgBrick"][7], data[7])
        assert np.array_equal(h["imgBrick"][1900], data[1900])

    # The whole point: two rows cost far less than the whole file.
    assert client.bytes_served < len(client.payload) / 4
