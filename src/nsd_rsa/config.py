"""Config loading and deterministic seeding.

Every knob in this project lives in configs/*.yaml. Code reads config; code never
hardcodes a path, a subject list, or a hyperparameter.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repo root, resolved from this file's location (works regardless of cwd)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a yaml config. Relative paths resolve against the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    cfg = yaml.safe_load(p.read_text())
    cfg["_config_path"] = str(p)
    return cfg


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Turn the `paths:` block into absolute Paths, creating directories as needed."""
    root = project_root()
    out: dict[str, Path] = {}
    for key, rel in cfg.get("paths", {}).items():
        p = Path(rel)
        out[key] = p if p.is_absolute() else root / p
        out[key].mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Called at the top of each stage script."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
