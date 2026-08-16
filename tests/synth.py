"""Synthetic broadcast frames for testing segmentation without real footage.

Each generator mimics the property the real classifier keys on, not the
appearance of the real thing: pitch frames are green-dominant with a static
overlay and moving blobs, crowd frames are busy and green-poor, and so on.
"""

from __future__ import annotations

import cv2
import numpy as np

W, H = 320, 180
BUG_RECT = (10, 10, 90, 22)      # x, y, w, h -- fixed position, as a real bug is


def _noise(rng, shape, scale=12):
    return rng.normal(0, scale, shape)


def pitch_frame(rng, with_bug: bool = True) -> np.ndarray:
    """Wide tactical view: green-dominant, moving players, optional score bug."""
    frame = np.zeros((H, W, 3), np.float32)
    frame[:] = (60, 140, 60)                       # BGR grass
    frame += _noise(rng, frame.shape, 10)

    # Pitch lines -- static geometry, but the camera pans so they move.
    offset = rng.integers(-40, 40)
    cv2.line(frame, (W // 2 + offset, 0), (W // 2 + offset, H), (200, 200, 200), 1)
    cv2.circle(frame, (W // 2 + offset, H // 2), 34, (200, 200, 200), 1)

    # Players: small, numerous, and in different places every frame. This is
    # what gives the pitch region high temporal variance.
    for _ in range(14):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        colour = (240, 240, 240) if rng.random() < 0.5 else (40, 40, 200)
        cv2.rectangle(frame, (x, y), (x + 5, y + 11), colour, -1)

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    if with_bug:
        x, y, w, h = BUG_RECT
        cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 30, 28), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (210, 210, 210), 1)
        cv2.rectangle(frame, (x + 4, y + 5), (x + 22, y + 17), (200, 120, 40), -1)
        cv2.putText(frame, "2-1", (x + 30, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1)
        # Only the clock ticks -- a small fraction of the bug's area, which is
        # why template correlation stays high while it is drawn.
        mins = int(rng.integers(0, 90))
        cv2.putText(frame, f"{mins:02d}", (x + 64, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1)

    return frame


def replay_frame(rng) -> np.ndarray:
    """Pitch footage with the bug removed -- the case green ratio cannot catch."""
    return pitch_frame(rng, with_bug=False)


def crowd_frame(rng) -> np.ndarray:
    """Busy, green-poor: many small warm-toned blocks."""
    frame = np.zeros((H, W, 3), np.float32)
    frame[:] = (40, 35, 60)
    for _ in range(500):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        b = int(rng.integers(0, 255))
        g = int(rng.integers(0, 90))       # keep green from dominating hue
        r = int(rng.integers(0, 255))
        cv2.rectangle(frame, (x, y), (x + 6, y + 6), (b, g, r), -1)
    frame += _noise(rng, frame.shape, 8)
    return np.clip(frame, 0, 255).astype(np.uint8)


def closeup_frame(rng) -> np.ndarray:
    """Player close-up: pitch reduced to a strip behind a large jersey mass.

    Green ratio needs to land between GREEN_SOME and GREEN_WIDE -- that band is
    exactly what "pitch visible but not dominant" means.
    """
    frame = np.zeros((H, W, 3), np.float32)
    frame[:] = (60, 140, 60)
    frame += _noise(rng, frame.shape, 10)
    # Out-of-focus stand fills the upper half; the subject fills the lower.
    cv2.rectangle(frame, (0, 0), (W, 52), (70, 60, 95), -1)
    cx = int(rng.integers(130, 190))
    cv2.ellipse(frame, (cx, 155), (112, 88), 0, 0, 360, (30, 30, 190), -1)
    cv2.circle(frame, (cx, 72), 38, (120, 150, 190), -1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def graphic_frame(rng) -> np.ndarray:
    """Full-screen graphic: flat, low-texture, green-poor."""
    frame = np.zeros((H, W, 3), np.float32)
    frame[:] = (110, 55, 25)
    cv2.rectangle(frame, (30, 50), (290, 90), (200, 130, 60), -1)
    cv2.rectangle(frame, (30, 105), (220, 135), (180, 110, 50), -1)
    frame += _noise(rng, frame.shape, 3)
    return np.clip(frame, 0, 255).astype(np.uint8)


GENERATORS = {
    "live": lambda rng: pitch_frame(rng, with_bug=True),
    "replay": replay_frame,
    "crowd": crowd_frame,
    "closeup": closeup_frame,
    "graphic": graphic_frame,
}


def build_sequence(plan, seed: int = 0):
    """Build frames from a plan of (kind, n_frames) pairs.

    Returns (frames, truth_boundaries) where truth_boundaries are the frame
    indices at which a new shot starts (excluding 0).
    """
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    boundaries: list[int] = []
    for kind, count in plan:
        if frames:
            boundaries.append(len(frames))
        gen = GENERATORS[kind]
        frames.extend(gen(rng) for _ in range(count))
    return frames, boundaries
