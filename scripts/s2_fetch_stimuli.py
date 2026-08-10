#!/usr/bin/env python
"""S2 (prep) — fetch the 1000 shared NSD images.

The full stimulus file is 39.6 GB for all 73,000 images. We need 1,000 of them. The
dataset inside is stored contiguous and uncompressed, so each image is one contiguous
byte range and h5py — handed an S3-backed file object — transfers only what we ask for.

    39.6 GB   whole nsd_stimuli.hdf5
     542 MB   the 1000 shared images (425 x 425 x 3 uint8 each)

Images are written once to a single .npy so model activation extraction never touches
the network again.

Usage: make s2-stimuli
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa import nsd  # noqa: E402
from nsd_rsa.config import load_config, resolve_paths  # noqa: E402
from nsd_rsa.design import load_expdesign  # noqa: E402
from nsd_rsa.s3file import S3File  # noqa: E402

STIMULI_KEY = "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    out = paths["stimuli"] / "shared1000_images.npy"
    meta_out = paths["stimuli"] / "shared1000_nsd_ids.npy"

    if out.exists() and not args.force:
        arr = np.load(out, mmap_mode="r")
        print(f"already present: {out} shape={arr.shape}")
        return 0

    import h5py

    client = nsd.get_client()

    design_path = paths["meta"] / "nsd_expdesign.mat"
    if not design_path.exists():
        nsd.download(nsd.EXPDESIGN_KEY, design_path, client)
    sharedix = load_expdesign(design_path)["sharedix"]  # 1-based NSD image ids

    # Also grab the stimulus metadata table: COCO ids, categories, crop boxes. Small,
    # and needed later if we want to split results by image content.
    nsd.download(nsd.STIM_INFO_KEY, paths["meta"] / "nsd_stim_info_merged.csv", client)

    f = S3File(client, nsd.BUCKET, STIMULI_KEY)
    print(f"remote stimulus file: {f.size/1e9:.2f} GB")

    with h5py.File(f, "r") as h:
        brick = h["imgBrick"]
        if brick.chunks is not None or brick.compression is not None:
            raise RuntimeError(
                "imgBrick is chunked or compressed — the contiguous-read assumption "
                "behind this script no longer holds"
            )
        print(f"imgBrick: {brick.shape} {brick.dtype}")

        n = len(sharedix)
        images = np.empty((n,) + brick.shape[1:], dtype=np.uint8)
        per_image = int(np.prod(brick.shape[1:]))
        t0 = time.time()
        for i, nsd_id in enumerate(sharedix):
            images[i] = brick[int(nsd_id) - 1]  # ids are 1-based
            if (i + 1) % 50 == 0 or i == n - 1:
                done = (i + 1) * per_image
                el = time.time() - t0
                print(f"  {i+1:4d}/{n}  {done/1e6:6.0f} MB  {done/1e6/el:.2f} MB/s  "
                      f"~{(n-i-1)*el/(i+1)/60:.0f} min left")

    np.save(out, images)
    np.save(meta_out, np.asarray(sharedix))
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.0f} MB, shape={images.shape})")
    print(f"transferred {f.n_bytes/1e6:.0f} MB in {f.n_requests} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
