"""Pitch keypoints, homography, and canonical coordinates.

Broadcast cameras pan and zoom, so the image-to-pitch transform must be
re-solved per frame. Once solved, every position becomes metres on a canonical
105x68 m pitch -- which is what makes distances, speeds, zones, and formation
shape mean anything. Zones defined in pitch coordinates do not move with the
camera. That is the entire point.

**A bad homography is worse than no homography.** It does not raise; it silently
maps every player to a plausible-looking but wrong location, and the event layer
happily reports shots and turnovers derived from nonsense. So solving is only
half the job -- everything here is built around rejecting bad solutions and
returning ``None``, which downstream already treats as "no usable state for this
frame".

The keypoint detector is injected for the same reason as in
:mod:`caster.perception`: the weights are environment-specific, the geometry
around them is ordinary code that should be testable without a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

#: Canonical landmark positions in metres, origin at the bottom-left corner.
#: A keypoint model is expected to report some subset of these by name.
LANDMARKS: dict[str, tuple[float, float]] = {
    "corner_bl": (0.0, 0.0),
    "corner_tl": (0.0, 68.0),
    "corner_br": (105.0, 0.0),
    "corner_tr": (105.0, 68.0),
    "halfway_bottom": (52.5, 0.0),
    "halfway_top": (52.5, 68.0),
    "centre_spot": (52.5, 34.0),
    # Left penalty area (16.5 m deep, 40.32 m wide)
    "pen_l_bottom_outer": (16.5, 13.84),
    "pen_l_top_outer": (16.5, 54.16),
    "pen_l_bottom_goal_line": (0.0, 13.84),
    "pen_l_top_goal_line": (0.0, 54.16),
    "pen_spot_l": (11.0, 34.0),
    # Right penalty area
    "pen_r_bottom_outer": (88.5, 13.84),
    "pen_r_top_outer": (88.5, 54.16),
    "pen_r_bottom_goal_line": (105.0, 13.84),
    "pen_r_top_goal_line": (105.0, 54.16),
    "pen_spot_r": (94.0, 34.0),
    # Goal areas (5.5 m deep, 18.32 m wide)
    "goal_l_bottom_outer": (5.5, 24.84),
    "goal_l_top_outer": (5.5, 43.16),
    "goal_r_bottom_outer": (99.5, 24.84),
    "goal_r_top_outer": (99.5, 43.16),
}


class KeypointModel(Protocol):
    """Maps a frame to detected pitch landmarks in image coordinates.

    Returns ``{landmark_name: (x, y)}`` for whatever it can see. Names must come
    from :data:`LANDMARKS`; anything unrecognised is ignored.
    """

    def __call__(self, frame: np.ndarray) -> dict[str, tuple[float, float]]: ...


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------

MIN_CORRESPONDENCES = 4          # cv2.findHomography's hard floor
MAX_REPROJECTION_PX = 12.0
MIN_QUAD_AREA_FRAC = 0.01        # projected pitch must not collapse to a sliver


@dataclass
class Homography:
    """A solved transform plus the evidence for trusting it."""

    matrix: np.ndarray
    reprojection_px: float
    n_points: int
    inliers: int

    def to_pitch(self, points) -> np.ndarray:
        """Image coordinates -> pitch metres."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.matrix)
        return out.reshape(-1, 2)

    def to_image(self, points) -> np.ndarray:
        """Pitch metres -> image coordinates."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        inv = np.linalg.inv(self.matrix)
        out = cv2.perspectiveTransform(pts, inv)
        return out.reshape(-1, 2)


def _is_convex_quad(pts: np.ndarray) -> bool:
    """True if the four points form a convex, non-self-intersecting quad.

    The pitch is a rectangle, so under any physically real camera it projects
    to a convex quadrilateral. A solution that folds it over is arithmetically
    valid and geometrically impossible.
    """
    # 2-D cross product written out: np.cross dropped 2-D vector support in
    # NumPy 2.0 and now raises on them.
    signs = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        u, v = b - a, c - b
        signs.append(np.sign(u[0] * v[1] - u[1] * v[0]))
    return abs(sum(signs)) == 4


def solve_homography(
    image_points: dict[str, tuple[float, float]],
    max_reprojection_px: float = MAX_REPROJECTION_PX,
    frame_shape: tuple[int, int] | None = None,
) -> Homography | None:
    """Solve image -> pitch from named landmark correspondences.

    Returns ``None`` rather than a suspect transform. Three things are checked,
    and each catches a different way of being wrong:

    - **Enough correspondences.** Four is the algebraic minimum; below that
      there is no solution at all.
    - **Reprojection error.** Maps the landmarks back and measures the miss.
      Catches mislabelled or badly localised keypoints, which otherwise produce
      a confident transform built on bad inputs.
    - **Projected pitch shape.** Transforms the pitch corners into the image and
      requires a convex quadrilateral of non-trivial area. Catches degenerate
      solutions from nearly-collinear points -- the common failure when a camera
      is tight on one touchline and every visible landmark lies on one line.
    """
    usable = {k: v for k, v in image_points.items() if k in LANDMARKS}
    if len(usable) < MIN_CORRESPONDENCES:
        log.debug("homography: %d correspondences, need %d",
                  len(usable), MIN_CORRESPONDENCES)
        return None

    names = sorted(usable)
    src = np.array([usable[n] for n in names], dtype=np.float64)
    dst = np.array([LANDMARKS[n] for n in names], dtype=np.float64)

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if matrix is None:
        return None

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
    if error > max_reprojection_px:
        log.debug("homography rejected: reprojection %.1f > %.1f",
                  error, max_reprojection_px)
        return None

    # Project the pitch corners back into the image and sanity-check the shape.
    corners_pitch = np.array(
        [[0, 0], [PITCH_LENGTH_M, 0], [PITCH_LENGTH_M, PITCH_WIDTH_M], [0, PITCH_WIDTH_M]],
        dtype=np.float64,
    )
    try:
        inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    corners_img = cv2.perspectiveTransform(
        corners_pitch.reshape(-1, 1, 2), inv
    ).reshape(-1, 2)

    if not np.all(np.isfinite(corners_img)) or not _is_convex_quad(corners_img):
        log.debug("homography rejected: pitch does not project to a convex quad")
        return None

    if frame_shape is not None:
        h, w = frame_shape[:2]
        area = 0.5 * abs(
            np.dot(corners_img[:, 0], np.roll(corners_img[:, 1], 1))
            - np.dot(corners_img[:, 1], np.roll(corners_img[:, 0], 1))
        )
        if area < MIN_QUAD_AREA_FRAC * w * h:
            log.debug("homography rejected: projected pitch area too small")
            return None

    return Homography(
        matrix=matrix,
        reprojection_px=error,
        n_points=len(names),
        inliers=int(mask.sum()) if mask is not None else len(names),
    )


class PitchMapper:
    """Per-frame homography with a fallback to the last good solution.

    Landmarks drop in and out as the camera moves, so a strict per-frame solve
    produces gaps. Reusing the previous transform bridges short gaps, but only
    briefly: the camera is moving, so a stale transform drifts steadily further
    from the truth. After ``max_stale_frames`` it is better to admit there is no
    usable mapping than to keep quietly reporting wrong positions.
    """

    def __init__(self, keypoint_model: KeypointModel, max_stale_frames: int = 5):
        self.keypoint_model = keypoint_model
        self.max_stale_frames = max_stale_frames
        self.last: Homography | None = None
        self._stale = 0
        self.solved = 0
        self.reused = 0
        self.failed = 0

    def update(self, frame: np.ndarray) -> Homography | None:
        points = self.keypoint_model(frame)
        homography = solve_homography(points, frame_shape=frame.shape)

        if homography is not None:
            self.last, self._stale = homography, 0
            self.solved += 1
            return homography

        if self.last is not None and self._stale < self.max_stale_frames:
            self._stale += 1
            self.reused += 1
            return self.last

        self.last = None
        self.failed += 1
        return None

    @property
    def coverage(self) -> float:
        """Fraction of frames with a usable mapping. Low means bad keypoints."""
        total = self.solved + self.reused + self.failed
        return (self.solved + self.reused) / total if total else 0.0


def on_pitch(xy, margin_m: float = 5.0) -> bool:
    """Whether a mapped point is plausibly on the pitch.

    A generous margin: players legitimately step over the touchline, and the
    ball goes out. This is a sanity filter for mapping blunders, not a rules
    check -- a point twenty metres off the pitch is a bad transform, not a
    throw-in.
    """
    x, y = xy
    return (
        -margin_m <= x <= PITCH_LENGTH_M + margin_m
        and -margin_m <= y <= PITCH_WIDTH_M + margin_m
    )


def attacking_third(team_attacks_x: float) -> tuple[float, float]:
    third = PITCH_LENGTH_M / 3
    return (PITCH_LENGTH_M - third, PITCH_LENGTH_M) if team_attacks_x else (0.0, third)
