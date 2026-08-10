"""NSD experimental design: which trial showed which image.

The betas for a session are a (750, n_vertices) array with no labels attached. This
module answers "what was trial 437 of session 12?" from nsd_expdesign.mat.

Verified against the real file on 2026-08-10:
  subjectim      (8, 10000) int32  — image ids (1..73000) each subject saw
  masterordering (30000,)   uint16 — for each of 40*750 trials, which of that
                                     subject's 10000 images was shown (1-indexed)
  sharedix       (1000,)    int32  — the 1000 image ids shown to everyone

Two facts we verified rather than assumed, because everything downstream rests on them:
  1. subjectim[:, :1000] equals sharedix in identical order for ALL 8 subjects.
     So shared-image slot j means the same picture for every subject, no remapping.
  2. masterordering <= 1000 selects exactly 3000 trials = 1000 images x 3 repeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRIALS_PER_SESSION = 750
N_SHARED = 1000

# Sessions actually collected. NSD is unbalanced; ignoring this silently reads
# trials that were never scanned.
SESSIONS_PER_SUBJECT = {
    "subj01": 40, "subj02": 40, "subj03": 32, "subj04": 30,
    "subj05": 40, "subj06": 32, "subj07": 40, "subj08": 30,
}


@dataclass
class SharedTrials:
    """Where every shared1000 presentation lives, for one subject.

    Arrays are parallel and sorted by (session, frame):
      session : 1-based session number
      frame   : 0-based trial index within that session's beta file
      image   : 0-based index into the shared1000 set (0..999)
      repeat  : 0-based repetition counter for that image (0, 1, or 2)
    """

    subject: str
    session: np.ndarray
    frame: np.ndarray
    image: np.ndarray
    repeat: np.ndarray

    def __len__(self) -> int:
        return len(self.session)

    @property
    def n_sessions(self) -> int:
        return SESSIONS_PER_SUBJECT[self.subject]

    def repeats_per_image(self) -> np.ndarray:
        """(1000,) count of available repeats per shared image."""
        return np.bincount(self.image, minlength=N_SHARED)

    def frames_in_session(self, session: int) -> np.ndarray:
        return self.frame[self.session == session]


def load_expdesign(path: str | Path) -> dict:
    import scipy.io as sio

    m = sio.loadmat(str(path))
    return {
        "subjectim": m["subjectim"],
        "masterordering": m["masterordering"].ravel(),
        "sharedix": m["sharedix"].ravel(),
    }


def verify_expdesign(design: dict) -> None:
    """Assert the two structural facts this module relies on. Cheap, and turns a silent
    mislabelling bug into a loud failure."""
    subjectim, sharedix = design["subjectim"], design["sharedix"]
    for s in range(subjectim.shape[0]):
        if not np.array_equal(subjectim[s, :N_SHARED], sharedix):
            raise ValueError(
                f"subjectim[{s}, :1000] != sharedix — the assumption that the first 1000 "
                "columns are the shared images in a fixed order does not hold"
            )
    mo = design["masterordering"]
    shared = mo[mo <= N_SHARED]
    counts = np.bincount(shared, minlength=N_SHARED + 1)[1:]
    if not (counts == 3).all():
        raise ValueError("masterordering does not give exactly 3 repeats per shared image")


def shared_trials(subject: str, design: dict) -> SharedTrials:
    """Locate every shared1000 presentation this subject actually received.

    Subjects who did fewer than 40 sessions simply never saw the tail of the design,
    so some shared images have fewer than 3 repeats — or none at all. We truncate the
    ordering to the sessions that exist rather than pretending otherwise.
    """
    if subject not in SESSIONS_PER_SUBJECT:
        raise KeyError(f"unknown subject {subject!r}")
    n_sess = SESSIONS_PER_SUBJECT[subject]

    mo = design["masterordering"][: n_sess * TRIALS_PER_SESSION]
    trial_idx = np.where(mo <= N_SHARED)[0]

    session = trial_idx // TRIALS_PER_SESSION + 1
    frame = trial_idx % TRIALS_PER_SESSION
    image = mo[trial_idx].astype(np.int64) - 1  # to 0-based shared slot

    # Repeat counter, assigned in presentation order.
    repeat = np.zeros(len(image), dtype=np.int64)
    seen: dict[int, int] = {}
    for i, img in enumerate(image):
        repeat[i] = seen.get(img, 0)
        seen[img] = repeat[i] + 1

    return SharedTrials(subject, session, frame, image, repeat)


def usable_images(
    subjects: list[str], design: dict, min_repeats: int = 3
) -> np.ndarray:
    """Shared image slots with at least `min_repeats` repeats in EVERY listed subject.

    This is the stimulus-set decision that group RSA hinges on. Measured on the real
    design for all 8 subjects:
        min_repeats=1 -> 907 images
        min_repeats=2 -> 766 images
        min_repeats=3 -> 515 images
    Only subj01/02/05/07 (40 sessions) have all 1000 at 3 repeats.

    We default to 3 because unequal repeat counts mean unequal measurement noise per
    image, and an image measured once has a systematically noisier response pattern than
    one measured three times. In an RDM that inflates its distance to everything else,
    which is a stimulus-level confound masquerading as representational structure.
    """
    counts = np.vstack([shared_trials(s, design).repeats_per_image() for s in subjects])
    return np.where(counts.min(axis=0) >= min_repeats)[0]
