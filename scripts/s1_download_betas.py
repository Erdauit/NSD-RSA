#!/usr/bin/env python
"""S1 — pull the shared1000 betas for one or more subjects.

Strategy `range_read`: NSD's fsaverage betas are uncompressed and frame-contiguous, so
a single trial is one byte range. We fetch only the ~3000 shared1000 trials instead of
all 30000, then keep only vertices inside the streams ROIs.

    39.3 GB  full download of every session file (subj01)
     3.9 GB  range_read of just the shared trials
     0.8 GB  what actually lands on disk after ROI masking

Correctness of range_read was established by downloading one full session file and
checking the range-read frames are bitwise identical (scripts/../tests). Do not switch
this on for a new space or beta version without repeating that check.

Resumable: each session is cached separately, so an interrupted run continues.

Usage:
    python scripts/s1_download_betas.py --config configs/data.yaml
    python scripts/s1_download_betas.py --config configs/data.yaml --subjects subj01 subj02
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsd_rsa import nsd  # noqa: E402
from nsd_rsa.config import load_config, resolve_paths, set_seed  # noqa: E402
from nsd_rsa.design import (  # noqa: E402
    SESSIONS_PER_SUBJECT,
    load_expdesign,
    shared_trials,
    verify_expdesign,
)
from nsd_rsa.rois import load_streams  # noqa: E402


def fetch_metadata(paths: dict[str, Path], client) -> None:
    """Small fixed files: experiment design and the ROI atlas. ~0.3 MB total."""
    meta = paths["meta"]
    wanted = [
        (nsd.EXPDESIGN_KEY, meta / "nsd_expdesign.mat"),
        (nsd.streams_key("lh"), meta / "lh.streams.mgz"),
        (nsd.streams_key("rh"), meta / "rh.streams.mgz"),
        ("nsddata/freesurfer/fsaverage/label/streams.mgz.ctab", meta / "streams.ctab"),
    ]
    for key, dest in wanted:
        nsd.download(key, dest, client)


def fetch_ncsnr(subject: str, paths: dict[str, Path], cfg: dict, client) -> None:
    """Noise-ceiling SNR maps, including the two split-half versions. ~4 MB/subject."""
    n = cfg["nsd"]
    for hemi in n["hemis"]:
        for split in (None, "split1", "split2"):
            key = nsd.ncsnr_key(subject, hemi, n["space"], n["betas_version"], split)
            name = f"{hemi}.ncsnr{'_' + split if split else ''}.mgh"
            nsd.download(key, paths["meta"] / subject / name, client)


def download_session(
    subject: str,
    session: int,
    frames: np.ndarray,
    roi_idx_by_hemi: dict[str, np.ndarray],
    cfg: dict,
    client,
    workers: int,
) -> np.ndarray:
    """Fetch one session's shared trials, both hemispheres, ROI-masked.

    Returns (n_frames, n_roi_vertices) float32 with hemispheres concatenated [lh, rh].
    """
    n = cfg["nsd"]
    per_hemi = []
    for hemi in n["hemis"]:
        key = nsd.betas_key(subject, session, hemi, n["space"], n["betas_version"])
        header = nsd.read_mgh_header_s3(key, client)
        keep = roi_idx_by_hemi[hemi]

        def one(f: int, _key=key, _hdr=header, _keep=keep) -> np.ndarray:
            arr = nsd.read_mgh_frames_ranged(_key, np.array([f]), client, _hdr)
            return arr[0, _keep]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(one, frames.tolist()))
        per_hemi.append(np.asarray(rows, dtype=np.float32))

    return np.concatenate(per_hemi, axis=1)


def process_subject(subject: str, cfg: dict, paths: dict[str, Path], client) -> Path:
    n = cfg["nsd"]
    workers = cfg["download"].get("max_parallel", 32)

    design = load_expdesign(paths["meta"] / "nsd_expdesign.mat")
    verify_expdesign(design)
    trials = shared_trials(subject, design)

    atlas = load_streams(paths["meta"])
    keep_mask = atlas.any_roi_mask(n["rois"])
    keep_idx = np.where(keep_mask)[0]
    # Split the concatenated ROI index back into per-hemisphere indices, because each
    # hemisphere is a separate file on S3.
    roi_idx_by_hemi = {
        "lh": keep_idx[keep_idx < atlas.lh_size],
        "rh": keep_idx[keep_idx >= atlas.lh_size] - atlas.lh_size,
    }

    print(f"\n=== {subject} ===")
    print(f"  sessions collected      : {SESSIONS_PER_SUBJECT[subject]}")
    print(f"  shared1000 trials found : {len(trials)}")
    reps = trials.repeats_per_image()
    print(f"  images with 3/2/1/0 reps: {(reps==3).sum()}/{(reps==2).sum()}/"
          f"{(reps==1).sum()}/{(reps==0).sum()}")
    print(f"  ROI vertices kept       : {len(keep_idx):,} of {atlas.n_vertices:,} "
          f"({len(keep_idx)/atlas.n_vertices*100:.1f}%)")
    est_gb = len(trials) * atlas.n_vertices * 4 / 1e9
    print(f"  estimated transfer      : {est_gb:.2f} GB")

    fetch_ncsnr(subject, paths, cfg, client)

    session_cache = paths["cache"] / "betas_sessions" / subject
    session_cache.mkdir(parents=True, exist_ok=True)

    sessions = np.unique(trials.session)
    t_start = time.time()
    bytes_done = 0
    # Count only sessions we actually transfer: cached ones return instantly, and
    # including them in the rate would make the ETA wildly optimistic on a resumed run.
    downloaded = 0
    for i, sess in enumerate(sessions, 1):
        out = session_cache / f"session{sess:02d}.npy"
        frames = trials.frames_in_session(sess)
        if out.exists():
            print(f"  [{i:2d}/{len(sessions)}] session {sess:02d}: cached ({len(frames)} trials)")
            continue
        remaining_sessions = sum(
            1 for s in sessions[i - 1 :] if not (session_cache / f"session{s:02d}.npy").exists()
        )

        t0 = time.time()
        # Session-level retry on top of the per-request retry inside nsd.with_retry:
        # a whole session can still fail if the link drops for longer than the inner
        # backoff covers, and losing an hour of progress to one outage is unacceptable.
        data = nsd.with_retry(
            lambda s=sess, fr=frames: download_session(
                subject, s, fr, roi_idx_by_hemi, cfg, client, workers
            ),
            attempts=4,
            base_delay=10.0,
        )
        np.save(out, data)

        moved = len(frames) * atlas.n_vertices * 4
        bytes_done += moved
        downloaded += 1
        dt = time.time() - t0
        elapsed = time.time() - t_start
        rate = bytes_done / 1e6 / max(elapsed, 1e-9)
        remaining = (remaining_sessions - 1) * (elapsed / downloaded) / 60
        print(f"  [{i:2d}/{len(sessions)}] session {sess:02d}: {len(frames):3d} trials, "
              f"{moved/1e6:6.1f} MB in {dt:5.0f}s | {rate:.2f} MB/s | ~{remaining:.0f} min left")

    return assemble(subject, trials, keep_idx, atlas, paths, session_cache)


def assemble(subject, trials, keep_idx, atlas, paths, session_cache) -> Path:
    """Stitch per-session caches into one file, ordered to match `trials`."""
    import h5py

    chunks, order = [], []
    for sess in np.unique(trials.session):
        p = session_cache / f"session{sess:02d}.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing session cache {p} — rerun to finish the download")
        chunks.append(np.load(p))
        order.append(np.where(trials.session == sess)[0])

    betas = np.concatenate(chunks, axis=0)
    order = np.concatenate(order)
    # Restore the canonical trial order defined by `trials`.
    inverse = np.argsort(order)
    betas = betas[inverse]

    out = paths["betas"] / f"{subject}_shared_betas.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("betas", data=betas, compression="lzf")
        f.create_dataset("image", data=trials.image)
        f.create_dataset("repeat", data=trials.repeat)
        f.create_dataset("session", data=trials.session)
        f.create_dataset("frame", data=trials.frame)
        f.create_dataset("vertex_index", data=keep_idx)
        f.create_dataset("roi_label", data=atlas.labels[keep_idx])
        f.attrs["subject"] = subject
        f.attrs["lh_size"] = atlas.lh_size
        f.attrs["space"] = "fsaverage"
        f.attrs["note"] = "betas are per-trial, NOT averaged over repeats"

    print(f"  -> {out}  {out.stat().st_size/1e6:.0f} MB  shape={betas.shape}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    paths = resolve_paths(cfg)
    client = nsd.get_client()

    fetch_metadata(paths, client)
    subjects = args.subjects or cfg["nsd"]["subjects"]

    for subject in subjects:
        process_subject(subject, cfg, paths, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
