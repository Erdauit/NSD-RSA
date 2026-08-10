"""ROI masks from NSD's `streams` atlas, in fsaverage space.

`streams` is NSD's coarse parcellation of visual cortex into processing stages. Label
values are fixed by streams.mgz.ctab (verified against the real file):

    0 Unknown   1 early        2 midventral   3 midlateral
    4 midparietal   5 ventral   6 lateral      7 parietal

`early` (V1-V3) is the low-level stage; `ventral` is the high-level object/face/place
stage. The hierarchy early -> midventral -> ventral is the axis we test model layers against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

STREAM_LABELS = {
    1: "early",
    2: "midventral",
    3: "midlateral",
    4: "midparietal",
    5: "ventral",
    6: "lateral",
    7: "parietal",
}
LABEL_BY_NAME = {v: k for k, v in STREAM_LABELS.items()}

# The ventral hierarchy, in anatomical order. This ordering is what makes the
# "early layers -> early cortex, late layers -> ventral cortex" claim testable.
VENTRAL_HIERARCHY = ["early", "midventral", "ventral"]

HEMIS = ("lh", "rh")


@dataclass
class StreamsAtlas:
    """Concatenated lh+rh stream labels, plus the offset where rh begins."""

    labels: np.ndarray  # (n_vertices_both_hemis,) int
    lh_size: int

    @property
    def n_vertices(self) -> int:
        return len(self.labels)

    def mask(self, roi: str) -> np.ndarray:
        """Boolean mask over concatenated vertices for one ROI name."""
        if roi not in LABEL_BY_NAME:
            raise KeyError(f"unknown ROI {roi!r}; known: {sorted(LABEL_BY_NAME)}")
        return self.labels == LABEL_BY_NAME[roi]

    def indices(self, roi: str) -> np.ndarray:
        return np.where(self.mask(roi))[0]

    def any_roi_mask(self, rois: list[str] | None = None) -> np.ndarray:
        """Mask for the union of the given ROIs (default: all seven)."""
        rois = rois or list(LABEL_BY_NAME)
        out = np.zeros(self.n_vertices, dtype=bool)
        for r in rois:
            out |= self.mask(r)
        return out

    def counts(self) -> dict[str, int]:
        return {name: int((self.labels == val).sum()) for val, name in STREAM_LABELS.items()}

    def hemi_of(self, idx: np.ndarray) -> np.ndarray:
        """'lh'/'rh' for concatenated vertex indices — needed to report laterality."""
        return np.where(np.asarray(idx) < self.lh_size, "lh", "rh")


def load_streams(meta_dir: str | Path) -> StreamsAtlas:
    """Load lh/rh streams atlas and concatenate. Vertex order is [lh..., rh...],
    matching how we concatenate betas."""
    import nibabel as nib

    meta_dir = Path(meta_dir)
    parts = []
    for hemi in HEMIS:
        p = meta_dir / f"{hemi}.streams.mgz"
        if not p.exists():
            raise FileNotFoundError(f"missing ROI atlas {p}; run scripts/s0_fetch_metadata.py")
        parts.append(np.squeeze(np.asarray(nib.load(str(p)).dataobj)).astype(np.int16))
    return StreamsAtlas(labels=np.concatenate(parts), lh_size=len(parts[0]))
