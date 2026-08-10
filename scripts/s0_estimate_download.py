#!/usr/bin/env python
"""S0 — estimate NSD download volume BEFORE downloading anything.

Two modes:
  * analytic (default, works with no S3 access): compute sizes from the known NSD
    geometry in configs/data.yaml. Gives you the number today.
  * --probe: additionally query real object sizes from the NSD S3 bucket. Requires
    the signed Data Access Agreement to be in effect. Use this to confirm the
    analytic numbers before committing to a multi-hour download.

Run: make s0-estimate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

GB = 1e9


def human(n_bytes: float) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n_bytes >= div:
            return f"{n_bytes / div:.1f} {unit}"
    return f"{n_bytes:.0f} B"


def analytic_estimate(cfg: dict) -> dict:
    nsd = cfg["nsd"]
    n_vert = nsd["fsaverage_vertices_per_hemi"]
    n_hemi = len(nsd["hemis"])
    n_trials = nsd["trials_per_session"]
    itemsize = nsd["betas_itemsize"]  # 4 — verified from the MGH header, see configs/data.yaml
    sessions = nsd["sessions_per_subject"]

    subjects = nsd["subjects"]
    all_subjects = nsd["all_subjects"]

    # --- one session file, one hemisphere ---
    per_session_hemi = n_vert * n_trials * itemsize
    per_session = per_session_hemi * n_hemi

    def subject_total(subj: str) -> float:
        return per_session * sessions[subj]

    pilot_transfer = sum(subject_total(s) for s in subjects)
    full_transfer = sum(subject_total(s) for s in all_subjects)

    # --- what we actually KEEP: shared1000 x vertices, averaged over repeats ---
    # Conservative vertex count for the union of streams ROIs (early..ventral) on
    # fsaverage, both hemispheres. Refined for real in S1 once ROI files are read.
    roi_vertices_est = 30_000
    store_itemsize = 4 if cfg["download"]["store_dtype"] == "float32" else 2
    kept_per_subject = nsd["n_stimuli"] * roi_vertices_est * store_itemsize

    # --- small stuff ---
    ncsnr_per_subject = n_vert * n_hemi * 4  # float32 noise-ceiling SNR map
    roi_labels_per_subject = n_vert * n_hemi * 4
    expdesign = 5e6
    stim_info_csv = 150e6
    stimuli_1000 = 1000 * 425 * 425 * 3  # uint8 RGB, ~542 KB each

    # --- range_read strategy: fetch only the trials we need ---
    # The .mgh is uncompressed and frame-contiguous, so one trial = one contiguous
    # block of n_vert * itemsize bytes. We need n_stimuli * n_repeats trials per hemi.
    per_trial_block = n_vert * itemsize
    trials_needed = nsd["n_stimuli"] * nsd["n_repeats"]
    range_read_per_subject = per_trial_block * trials_needed * n_hemi
    range_pilot = range_read_per_subject * len(subjects)
    range_full = range_read_per_subject * len(all_subjects)

    return {
        "per_trial_block": per_trial_block,
        "trials_needed": trials_needed,
        "range_pilot": range_pilot,
        "range_full": range_full,
        "per_session_file": per_session,
        "pilot_subjects": subjects,
        "pilot_transfer": pilot_transfer,
        "pilot_peak_disk": per_session * 2,  # stream: 1 in flight + 1 being extracted
        "full_transfer": full_transfer,
        "kept_per_subject": kept_per_subject,
        "kept_pilot": kept_per_subject * len(subjects),
        "kept_full": kept_per_subject * len(all_subjects),
        "meta_per_subject": ncsnr_per_subject + roi_labels_per_subject,
        "meta_fixed": expdesign + stim_info_csv,
        "stimuli": stimuli_1000,
        "roi_vertices_est": roi_vertices_est,
    }


def probe_s3(cfg: dict) -> dict | None:
    """Ask S3 for the real size of one betas session file, to validate the analytic model."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("  (boto3 not installed — skipping probe)")
        return None

    nsd = cfg["nsd"]
    subj = nsd["subjects"][0]
    prefix = f"nsddata_betas/ppdata/{subj}/{nsd['space']}/{nsd['betas_version']}/"
    s3 = boto3.client(
        "s3", region_name=nsd["region"], config=Config(signature_version=UNSIGNED)
    )
    try:
        resp = s3.list_objects_v2(Bucket=nsd["bucket"], Prefix=prefix, MaxKeys=100)
    except Exception as e:  # noqa: BLE001
        print(f"  probe failed: {type(e).__name__}: {e}")
        print("  -> Have you signed the NSD Data Access Agreement? See docs/NSD_ACCESS.md")
        return None

    objs = resp.get("Contents", [])
    if not objs:
        print(f"  probe returned no objects under s3://{nsd['bucket']}/{prefix}")
        return None

    betas = [o for o in objs if "betas_session" in o["Key"]]
    print(f"  listed {len(objs)} objects under {prefix}")
    for o in betas[:4]:
        print(f"    {Path(o['Key']).name:<40} {human(o['Size'])}")
    if betas:
        return {"real_session_hemi_bytes": betas[0]["Size"]}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--probe", action="store_true", help="query real sizes from S3")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    est = analytic_estimate(cfg)
    nsd = cfg["nsd"]

    print("=" * 68)
    print("NSD DOWNLOAD ESTIMATE — analytic, from configs/data.yaml")
    print("=" * 68)
    print(f"space               {nsd['space']}  ({nsd['fsaverage_vertices_per_hemi']:,} vertices/hemi)")
    print(f"betas version       {nsd['betas_version']}")
    print(f"stimulus subset     {nsd['stimulus_subset']} ({nsd['n_stimuli']} images x {nsd['n_repeats']} repeats)")
    print(f"strategy            {cfg['download']['strategy']}")
    print()
    print("--- TRANSFER (bytes pulled over the network) ---")
    print(f"one session file (both hemis)   {human(est['per_session_file'])}")
    print()
    print(f"{'':<24}{'stream_sessions':>16}{'range_read':>16}")
    print(f"{'pilot ' + str(est['pilot_subjects']):<24}{human(est['pilot_transfer']):>16}{human(est['range_pilot']):>16}")
    print(f"{'all 8 subjects':<24}{human(est['full_transfer']):>16}{human(est['range_full']):>16}")
    print(f"  -> range_read saves {est['full_transfer'] / est['range_full']:.0f}x "
          f"({est['trials_needed']:,} trials/hemi x {human(est['per_trial_block'])} per trial)")
    print()
    print("--- PEAK DISK (stream_sessions: extract then delete) ---")
    print(f"transient raw            {human(est['pilot_peak_disk'])}")
    print(f"kept betas / subject     {human(est['kept_per_subject'])}  (~{est['roi_vertices_est']:,} ROI vertices)")
    print(f"kept betas / all 8       {human(est['kept_full'])}")
    print(f"stimuli (1000 images)    {human(est['stimuli'])}")
    print(f"metadata (fixed)         {human(est['meta_fixed'])}")
    print()
    total_keep_pilot = est["kept_pilot"] + est["stimuli"] + est["meta_fixed"] + est["meta_per_subject"] * len(nsd["subjects"])
    total_keep_full = est["kept_full"] + est["stimuli"] + est["meta_fixed"] + est["meta_per_subject"] * len(nsd["all_subjects"])
    print("--- BOTTOM LINE ---")
    print(f"pilot   : transfer {human(est['pilot_transfer']):>9}   ->  keep {human(total_keep_pilot):>9} on disk")
    print(f"all 8   : transfer {human(est['full_transfer']):>9}   ->  keep {human(total_keep_full):>9} on disk")
    print()
    print("Why transfer >> kept: shared1000 trials are scattered across ALL sessions,")
    print("so every session file must be read even though we keep 1000 images from it.")
    print("`range_read` avoids that by fetching only the needed trial blocks; switch to it")
    print("in configs/data.yaml once the pilot has validated the extraction logic.")

    if args.probe:
        print()
        print("--- S3 PROBE (real object sizes) ---")
        probe = probe_s3(cfg)
        if probe:
            real = probe["real_session_hemi_bytes"]
            model = est["per_session_file"] / 2
            print(f"  real  one-hemi session file: {human(real)}")
            print(f"  model one-hemi session file: {human(model)}")
            print(f"  ratio real/model: {real / model:.2f}  (≈1.0 means the estimate holds)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
