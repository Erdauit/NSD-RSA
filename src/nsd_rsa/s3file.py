"""A seekable read-only file object backed by S3 range requests.

Why this exists: `nsd_stimuli.hdf5` is 39.6 GB and holds all 73,000 NSD images, but we
need 1,000 of them. Handing this object to `h5py` lets HDF5 do its own seeking, so only
the bytes belonging to the requested images are ever transferred. The dataset is stored
contiguous and uncompressed (verified), so one image is one contiguous byte range.

Keep the block cache in mind: HDF5 issues many small reads for its own metadata, and
without coalescing each one would be a separate HTTPS round trip on a high-latency link.
"""

from __future__ import annotations

import io


class S3File(io.RawIOBase):
    """Minimal seekable reader over an S3 object, with a simple read-ahead cache."""

    def __init__(self, client, bucket: str, key: str, block_size: int = 1 << 20):
        self._client = client
        self._bucket = bucket
        self._key = key
        self._block = block_size
        self.size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        self._pos = 0
        self._cache_start = -1
        self._cache = b""
        self.n_requests = 0
        self.n_bytes = 0

    # --- io plumbing ---
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        return self._pos

    # --- transfer ---
    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        body = self._client.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={start}-{end}"
        )["Body"].read()
        self.n_requests += 1
        self.n_bytes += len(body)
        return body

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        size = min(size, self.size - self._pos)
        if size <= 0:
            return b""

        # Large reads (an actual image) go straight through; caching them would just
        # double memory for no reuse.
        if size >= self._block:
            data = self._fetch(self._pos, size)
            self._pos += len(data)
            return data

        # Small reads are HDF5 metadata lookups: fetch a block and serve from it.
        if not (self._cache_start <= self._pos and self._pos + size <= self._cache_start + len(self._cache)):
            self._cache_start = self._pos
            self._cache = self._fetch(self._pos, self._block)

        off = self._pos - self._cache_start
        data = self._cache[off : off + size]
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:  # noqa: ANN001
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
