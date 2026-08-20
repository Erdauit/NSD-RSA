"""Hand-crafted low-level image features: the trivial competitors and probe targets.

Used twice: as RDM baselines (any alignment claim must clear what raw brightness
achieves) and as decoding targets for the degradation-vs-retargeting control (what does
the language stack *lose*, low-level structure or everything?).
"""

from __future__ import annotations

import numpy as np


def block_pool(arr: np.ndarray, grid: int) -> np.ndarray:
    """Mean-pool a 2D map onto a (grid x grid) summary. Pixel-precise maps do not align
    across images; pooled density is what carries the comparable signal."""
    h, w = arr.shape
    bh, bw = h // grid, w // grid
    return arr[: bh * grid, : bw * grid].reshape(grid, bh, grid, bw).mean(axis=(1, 3))


def edge_features(gray: np.ndarray, grid: int = 16) -> np.ndarray:
    """Canny edge density per image, pooled on a grid. `gray` is (n, H, W) in [0, 255]."""
    from skimage.feature import canny

    out = np.empty((len(gray), grid * grid))
    for i, g in enumerate(gray):
        out[i] = block_pool(canny(g / 255.0, sigma=2.0).astype(np.float64), grid).ravel()
    return out


def gabor_features(gray: np.ndarray, grid: int = 8, side: int = 128,
                   progress_every: int = 100) -> np.ndarray:
    """V1-style Gabor energy: 4 spatial frequencies x 6 orientations, pooled on a grid.

    Energy = magnitude of the quadrature pair (complex Gabor response), i.e. the
    complex-cell model: sensitive to oriented structure, invariant to phase.
    """
    from skimage.filters import gabor
    from skimage.transform import resize

    freqs = (0.05, 0.1, 0.2, 0.33)
    thetas = tuple(np.pi * k / 6 for k in range(6))
    out = np.empty((len(gray), len(freqs) * len(thetas) * grid * grid))
    for i, g in enumerate(gray):
        small = resize(g / 255.0, (side, side), anti_aliasing=True)
        feats = []
        for f in freqs:
            for th in thetas:
                re, im = gabor(small, frequency=f, theta=th)
                feats.append(block_pool(np.hypot(re, im), grid).ravel())
        out[i] = np.concatenate(feats)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    gabor {i+1}/{len(gray)}", flush=True)
    return out
