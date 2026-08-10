#!/usr/bin/env python
"""Verify the environment is ready. Run via `make check`.

Checks Python version, required packages, compute device, and (optionally) whether
AWS credentials can see the NSD bucket. Exits non-zero if anything essential is broken.
"""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "nibabel",
    "h5py",
    "torch",
    "torchvision",
    "timm",
    "PIL",
    "boto3",
    "yaml",
    "matplotlib",
]

OK = "\033[32m  ok\033[0m"
BAD = "\033[31mFAIL\033[0m"
WARN = "\033[33mwarn\033[0m"


def check_python() -> bool:
    v = sys.version_info
    good = (v.major, v.minor) >= (3, 11)
    print(f"[{OK if good else BAD}] python {platform.python_version()} (need >= 3.11)")
    return good


def check_packages() -> bool:
    all_good = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"[{OK}] {name:<12} {ver}")
        except ImportError:
            print(f"[{BAD}] {name:<12} not installed")
            all_good = False
    return all_good


def check_device() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        print(f"[{OK}] compute      cuda ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        print(f"[{OK}] compute      mps (Apple Silicon GPU)")
    else:
        print(f"[{WARN}] compute      cpu only — S2 activation extraction will be slow")


def check_nsd_access() -> None:
    """NSD is on AWS Open Data: public read, but you must have signed the agreement."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print(f"[{WARN}] nsd s3      boto3 missing, skipped")
        return
    try:
        s3 = boto3.client("s3", region_name="us-east-2", config=Config(signature_version=UNSIGNED))
        resp = s3.list_objects_v2(
            Bucket="natural-scenes-dataset", Prefix="nsddata/experiments/nsd/", MaxKeys=1
        )
        if resp.get("KeyCount", 0) > 0:
            print(f"[{OK}] nsd s3       bucket reachable")
        else:
            print(f"[{WARN}] nsd s3       reachable but empty listing")
    except Exception as e:  # noqa: BLE001 — diagnostic script, any failure is informative
        print(f"[{WARN}] nsd s3       not reachable ({type(e).__name__}) — fine before access")


def check_disk() -> None:
    import shutil

    free_gb = shutil.disk_usage(".").free / 1e9
    flag = OK if free_gb > 60 else WARN
    print(f"[{flag}] disk free    {free_gb:.0f} GB (want >= 60 GB for one-subject pilot)")


def main() -> int:
    print("=== NSD-RSA environment check ===")
    py = check_python()
    pkgs = check_packages()
    check_device()
    check_disk()
    check_nsd_access()
    print()
    if py and pkgs:
        print("Environment OK.")
        return 0
    print("Environment INCOMPLETE — run `make setup`.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
