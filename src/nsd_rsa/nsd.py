"""S3 access to the Natural Scenes Dataset.

NSD lives in the AWS Open Data registry, so the bucket reads anonymously. That is a
technical fact, not a licence: using the data requires the signed NSD Data Access
Agreement. See docs/NSD_ACCESS.md.

Everything here is about *getting bytes*. Interpreting them (trial ordering, ROI
masking, averaging over repeats) lives in `loaders.py`.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

BUCKET = "natural-scenes-dataset"
REGION = "us-east-2"

# MGH is a FreeSurfer volume/surface format: a fixed 284-byte header, then the data
# block, then a short footer. Verified against lh.betas_session01.mgh on 2026-08-10.
MGH_HEADER_BYTES = 284
MGH_DTYPES = {0: np.uint8, 1: np.int32, 3: np.float32, 4: np.int16}


def get_client():
    """Anonymous S3 client. Anonymous because NSD is public-read, not because the
    licence is optional."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED, max_pool_connections=32),
    )


# --- key construction ---------------------------------------------------------


def betas_key(subject: str, session: int, hemi: str, space: str, version: str) -> str:
    return (
        f"nsddata_betas/ppdata/{subject}/{space}/{version}/"
        f"{hemi}.betas_session{session:02d}.mgh"
    )


def ncsnr_key(subject: str, hemi: str, space: str, version: str, split: str | None = None) -> str:
    name = f"ncsnr_{split}" if split else "ncsnr"
    return f"nsddata_betas/ppdata/{subject}/{space}/{version}/{hemi}.{name}.mgh"


def streams_key(hemi: str, subject: str = "fsaverage") -> str:
    """ROI atlas. In fsaverage space the atlas is shared, so `subject` stays 'fsaverage'."""
    return f"nsddata/freesurfer/{subject}/label/{hemi}.streams.mgz"


EXPDESIGN_KEY = "nsddata/experiments/nsd/nsd_expdesign.mat"
STIM_INFO_KEY = "nsddata/experiments/nsd/nsd_stim_info_merged.csv"


# --- download -----------------------------------------------------------------


def download(key: str, dest: Path, client=None, force: bool = False) -> Path:
    """Download an object unless a same-sized copy is already on disk."""
    client = client or get_client()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote_size = client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    if dest.exists() and not force and dest.stat().st_size == remote_size:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    client.download_file(BUCKET, key, str(tmp))
    tmp.replace(dest)
    return dest


_SIZE_CACHE: dict[str, int] = {}


def head_size(key: str, client=None) -> int:
    """Object size, memoised. Sizes are immutable here, and without the cache every
    single-frame range read would pay an extra HEAD round-trip."""
    if key not in _SIZE_CACHE:
        client = client or get_client()
        _SIZE_CACHE[key] = client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    return _SIZE_CACHE[key]


def validate_mgh_layout(key: str, header: dict[str, Any], client=None) -> None:
    """Assert the file really is uncompressed and frame-contiguous before we range-read it.

    A compressed or padded file would still return bytes for any range we ask for — they
    would just be the wrong bytes, silently. So this is checked, once per file, rather
    than assumed.
    """
    expected = MGH_HEADER_BYTES + header["nframes"] * header["frame_bytes"]
    actual = head_size(key, client)
    if not 0 <= actual - expected <= 64:
        raise ValueError(
            f"MGH layout assumption violated for {key}: file is {actual} bytes, "
            f"contiguous model predicts {expected} (+small footer). "
            "Refusing to range-read — the file may be compressed or padded."
        )


# --- MGH parsing --------------------------------------------------------------


def parse_mgh_header(raw: bytes) -> dict[str, Any]:
    """Parse the fixed part of an MGH header. Big-endian, per FreeSurfer spec."""
    version, width, height, depth, nframes, mgh_type = struct.unpack(">6i", raw[:24])
    if version != 1:
        raise ValueError(f"unexpected MGH version {version}")
    if mgh_type not in MGH_DTYPES:
        raise ValueError(f"unsupported MGH datatype code {mgh_type}")
    dtype = MGH_DTYPES[mgh_type]
    return {
        "version": version,
        "width": width,
        "height": height,
        "depth": depth,
        "nframes": nframes,
        "dtype": dtype,
        "itemsize": np.dtype(dtype).itemsize,
        "frame_bytes": width * height * depth * np.dtype(dtype).itemsize,
    }


def read_mgh_header_s3(key: str, client=None) -> dict[str, Any]:
    """Read just the header of a remote MGH via a 284-byte range request."""
    client = client or get_client()
    raw = client.get_object(Bucket=BUCKET, Key=key, Range=f"bytes=0-{MGH_HEADER_BYTES - 1}")[
        "Body"
    ].read()
    return parse_mgh_header(raw)


def read_mgh_frames_ranged(
    key: str, frames: np.ndarray, client=None, header: dict | None = None
) -> np.ndarray:
    """Fetch specific frames (trials) from a remote MGH using HTTP range requests.

    This is the `range_read` strategy. It is valid only because NSD's fsaverage betas
    are stored uncompressed and frame-contiguous, so frame f is exactly the byte range
    [284 + f*frame_bytes, 284 + (f+1)*frame_bytes). Both facts are asserted here rather
    than assumed: a compressed or non-contiguous file would silently produce garbage.

    Returns (n_frames, n_vertices) float32, in the order given by `frames`.
    """
    client = client or get_client()
    header = header or read_mgh_header_s3(key, client)
    validate_mgh_layout(key, header, client)

    n_vertices = header["width"] * header["height"] * header["depth"]
    frame_bytes = header["frame_bytes"]

    frames = np.asarray(frames, dtype=np.int64)
    if frames.min() < 0 or frames.max() >= header["nframes"]:
        raise IndexError(
            f"frame index out of range: requested [{frames.min()}, {frames.max()}], "
            f"file has {header['nframes']} frames"
        )

    out = np.empty((len(frames), n_vertices), dtype=np.float32)
    for i, f in enumerate(frames):
        start = MGH_HEADER_BYTES + int(f) * frame_bytes
        end = start + frame_bytes - 1
        body = client.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={start}-{end}")[
            "Body"
        ].read()
        if len(body) != frame_bytes:
            raise OSError(f"short read for frame {f}: got {len(body)} of {frame_bytes} bytes")
        out[i] = np.frombuffer(body, dtype=">f4" if header["dtype"] is np.float32 else header["dtype"])
    return out


def read_mgh_local(path: Path) -> np.ndarray:
    """Read a local .mgh/.mgz fully via nibabel. Returns (n_frames, n_vertices)."""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    # MGH surface files are (n_vertices, 1, 1, n_frames); squeeze to (frames, vertices).
    data = np.squeeze(data)
    if data.ndim == 1:
        return data[None, :]
    return data.T if data.shape[0] != 1 else data
