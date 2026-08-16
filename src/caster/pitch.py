"""Pitch keypoints, homography, and canonical coordinates.

Broadcast cameras pan and zoom, so the homography must be re-solved per frame.
Once solved, every position becomes metres on a canonical 105x68 m pitch — which
is what makes distances, speeds, zones, and formation shape meaningful.

Zones defined here do not move with the camera. That is the entire point.
"""

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def solve_homography(frame):
    """Detect pitch keypoints and solve image -> pitch coordinate transform.

    Returns None when too few keypoints are visible (close-ups, crowd shots).
    A None here should propagate as "no usable state for this frame" rather
    than being papered over with the previous frame's transform.
    """
    raise NotImplementedError


def to_pitch(points, homography):
    """Map image-space points into pitch metres."""
    raise NotImplementedError


def attacking_third(team: str) -> tuple[float, float]:
    """x-range of the attacking third for the given team."""
    raise NotImplementedError


def in_penalty_area(xy) -> bool:
    raise NotImplementedError
