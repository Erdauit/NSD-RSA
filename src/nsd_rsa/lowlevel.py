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


# ---- literature battery additions (docs/FEATURE_BATTERY.md) ---------------------------


def fourier_stats(gray01: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the rotationally averaged log power spectrum.

    Early visual cortex is tuned to near-natural 1/f spectra (Isherwood 2017); slope and
    intercept are the standard per-image summary (Groen 2013).
    """
    f = np.abs(np.fft.rfft2(gray01 - gray01.mean())) ** 2
    fy = np.fft.fftfreq(gray01.shape[0])[:, None]
    fx = np.fft.rfftfreq(gray01.shape[1])[None, :]
    rad = np.sqrt(fy**2 + fx**2).ravel()
    pw = f.ravel()
    keep = (rad > 0.01) & (rad < 0.5)
    slope, intercept = np.polyfit(np.log(rad[keep]), np.log(pw[keep] + 1e-12), 1)
    return float(slope), float(intercept)


def weibull_contrast(gray01: np.ndarray) -> tuple[float, float]:
    """Contrast energy (CE) and spatial coherence (SC): Weibull scale and shape of the
    gradient-magnitude distribution (Groen et al. 2013)."""
    from scipy.ndimage import sobel
    from scipy.stats import weibull_min

    gx, gy = sobel(gray01, 1), sobel(gray01, 0)
    mag = np.hypot(gx, gy).ravel()
    mag = mag[mag > 1e-6]
    shape, _, scale = weibull_min.fit(mag, floc=0)
    return float(scale), float(shape)


def subband_entropy(gray01: np.ndarray, levels: int = 4) -> float:
    """Rosenholtz-style clutter: mean Shannon entropy of Laplacian-pyramid coefficient
    histograms across levels."""
    from skimage.transform import pyramid_laplacian

    ents = []
    for lev in pyramid_laplacian(gray01, max_layer=levels, channel_axis=None):
        hist, _ = np.histogram(lev.ravel(), bins=64, density=True)
        p = hist[hist > 0]
        p = p / p.sum()
        ents.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(ents))


def feature_congestion_proxy(im: np.ndarray, grid: int = 8) -> float:
    """Clutter proxy in the spirit of Feature Congestion (Rosenholtz 2007): mean local
    variance of color (Lab a,b) and of orientation energy. A proxy, not the full model."""
    from skimage.color import rgb2lab

    lab = rgb2lab(im / 255.0)
    parts = []
    for ch in (lab[..., 1], lab[..., 2]):
        parts.append(block_pool((ch - ch.mean()) ** 2, grid).mean())
    gy, gx = np.gradient(lab[..., 0])
    parts.append(block_pool(np.hypot(gx, gy) ** 2, grid).mean())
    return float(np.log1p(np.mean(parts)))


def orientation_stats(gray01: np.ndarray, n_bins: int = 18) -> tuple[float, float, float]:
    """(rectilinearity, orientation entropy, curvature proxy) from gradient orientations.

    Rectilinearity: share of gradient energy within +-10 deg of cardinal orientations
    (Long et al. 2018 line of work). Curvature proxy: mean local circular dispersion of
    orientation among strong-gradient pixels.
    """
    from scipy.ndimage import sobel, uniform_filter

    gx, gy = sobel(gray01, 1), sobel(gray01, 0)
    mag = np.hypot(gx, gy)
    theta = np.mod(np.arctan2(gy, gx), np.pi)  # orientation, [0, pi)
    w = mag.ravel()
    if w.sum() < 1e-9:
        return 0.0, 0.0, 0.0
    hist, edges = np.histogram(theta.ravel(), bins=n_bins, range=(0, np.pi), weights=w)
    p = hist / hist.sum()
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / np.log2(n_bins))
    centers = (edges[:-1] + edges[1:]) / 2
    cardinal = (np.minimum(np.abs(centers - 0), np.abs(centers - np.pi)) < np.deg2rad(10)) | (
        np.abs(centers - np.pi / 2) < np.deg2rad(10)
    )
    rect = float(p[cardinal].sum())
    # curvature proxy: local circular variance of doubled orientation, magnitude-weighted
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    strong = mag > np.percentile(mag, 75)
    r = np.hypot(uniform_filter(c2, 5), uniform_filter(s2, 5))
    curv = float(1 - r[strong].mean()) if strong.any() else 0.0
    return rect, ent, curv


def gist_features(gray: np.ndarray, side: int = 128, grid: int = 4,
                  progress_every: int = 100) -> np.ndarray:
    """GIST-style descriptor (Oliva & Torralba): Gabor energy at 4 scales x 8
    orientations pooled on a (grid x grid) lattice -> 512 dims per image."""
    from skimage.filters import gabor
    from skimage.transform import resize

    freqs = (0.05, 0.1, 0.2, 0.33)
    thetas = tuple(np.pi * k / 8 for k in range(8))
    out = np.empty((len(gray), len(freqs) * len(thetas) * grid * grid))
    for i, g in enumerate(gray):
        small = resize(g / 255.0, (side, side), anti_aliasing=True)
        feats = []
        for f in freqs:
            for th in thetas:
                re, im_ = gabor(small, frequency=f, theta=th)
                feats.append(block_pool(np.hypot(re, im_), grid).ravel())
        out[i] = np.concatenate(feats)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    gist {i+1}/{len(gray)}", flush=True)
    return out
