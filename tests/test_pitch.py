"""Tests for homography and pitch coordinates.

A bad homography does not raise -- it silently maps every player to a
plausible-looking wrong location, and the event layer reports shots and
turnovers derived from nonsense. So most of these tests are about *rejection*:
the transform must return None rather than a confident bad answer.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from caster.pitch import (
    LANDMARKS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    PitchMapper,
    on_pitch,
    solve_homography,
)


def make_camera(pitch_pts, image_pts):
    """Homography mapping pitch metres -> image pixels."""
    return cv2.findHomography(
        np.asarray(pitch_pts, np.float64), np.asarray(image_pts, np.float64)
    )[0]


@pytest.fixture
def camera():
    """A plausible broadcast view: pitch corners projected with perspective."""
    pitch = [[0, 0], [PITCH_LENGTH_M, 0], [PITCH_LENGTH_M, PITCH_WIDTH_M], [0, PITCH_WIDTH_M]]
    image = [[180, 620], [1740, 620], [1450, 240], [470, 240]]   # far side smaller
    return make_camera(pitch, image)


def project(camera, name):
    p = np.array([[LANDMARKS[name]]], dtype=np.float64)
    return tuple(cv2.perspectiveTransform(p, camera).reshape(2))


def observed(camera, names, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for n in names:
        x, y = project(camera, n)
        if noise:
            x += rng.normal(0, noise)
            y += rng.normal(0, noise)
        out[n] = (x, y)
    return out


GOOD_SET = [
    "corner_bl", "corner_br", "corner_tr", "corner_tl",
    "halfway_bottom", "halfway_top", "pen_spot_l", "pen_spot_r",
]


# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------


def test_landmarks_are_inside_the_pitch():
    for name, (x, y) in LANDMARKS.items():
        assert 0.0 <= x <= PITCH_LENGTH_M, name
        assert 0.0 <= y <= PITCH_WIDTH_M, name


def test_penalty_areas_are_symmetric():
    assert LANDMARKS["pen_l_bottom_outer"][0] == pytest.approx(16.5)
    assert LANDMARKS["pen_r_bottom_outer"][0] == pytest.approx(PITCH_LENGTH_M - 16.5)
    assert LANDMARKS["pen_spot_l"][0] == pytest.approx(11.0)
    assert LANDMARKS["pen_spot_r"][0] == pytest.approx(PITCH_LENGTH_M - 11.0)


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------


def test_clean_correspondences_recover_the_transform(camera):
    h = solve_homography(observed(camera, GOOD_SET))
    assert h is not None
    assert h.reprojection_px < 1.0

    # Round-trip a known landmark through the solved transform.
    got = h.to_pitch([project(camera, "centre_spot")])[0]
    assert got[0] == pytest.approx(52.5, abs=0.5)
    assert got[1] == pytest.approx(34.0, abs=0.5)


def test_round_trip_image_to_pitch_and_back(camera):
    h = solve_homography(observed(camera, GOOD_SET))
    image_pt = project(camera, "pen_spot_r")
    back = h.to_image(h.to_pitch([image_pt]))[0]
    assert back == pytest.approx(image_pt, abs=1.0)


def test_modest_keypoint_noise_is_tolerated(camera):
    h = solve_homography(observed(camera, GOOD_SET, noise=2.0, seed=1))
    assert h is not None
    got = h.to_pitch([project(camera, "centre_spot")])[0]
    assert got[0] == pytest.approx(52.5, abs=4.0)


def test_too_few_correspondences_is_rejected(camera):
    assert solve_homography(observed(camera, GOOD_SET[:3])) is None


def test_unknown_landmark_names_are_ignored(camera):
    points = observed(camera, GOOD_SET[:3])
    points["not_a_real_landmark"] = (10.0, 10.0)
    assert solve_homography(points) is None, "bogus name should not count"


def test_badly_localised_keypoints_are_rejected(camera):
    """Mislabelled keypoints otherwise yield a confident transform on bad input."""
    points = observed(camera, GOOD_SET)
    points["corner_bl"] = (1500.0, 100.0)      # wildly wrong
    points["corner_tr"] = (200.0, 700.0)
    points["halfway_top"] = (900.0, 900.0)
    assert solve_homography(points, max_reprojection_px=8.0) is None


def test_collinear_keypoints_are_rejected():
    """The classic degenerate case: camera tight on one touchline.

    Four points on a line admit an arithmetically valid homography that folds
    the pitch into a sliver. It does not raise -- it just produces nonsense.
    """
    points = {
        "corner_bl": (100.0, 500.0),
        "halfway_bottom": (500.0, 500.0),
        "corner_br": (900.0, 500.0),
        "pen_spot_l": (300.0, 500.0),
    }
    assert solve_homography(points) is None


def test_tiny_projected_pitch_is_rejected(camera):
    """A transform that shrinks the whole pitch to a few pixels is not real."""
    pitch = [[0, 0], [PITCH_LENGTH_M, 0], [PITCH_LENGTH_M, PITCH_WIDTH_M], [0, PITCH_WIDTH_M]]
    tiny = make_camera(pitch, [[10, 10], [30, 10], [30, 25], [10, 25]])
    points = observed(tiny, GOOD_SET)
    assert solve_homography(points, frame_shape=(1080, 1920)) is None


def test_solution_reports_its_evidence(camera):
    h = solve_homography(observed(camera, GOOD_SET))
    assert h.n_points == len(GOOD_SET)
    assert h.inliers >= 4
    assert h.reprojection_px >= 0.0


# --------------------------------------------------------------------------
# PitchMapper
# --------------------------------------------------------------------------


class FakeKeypoints:
    """Returns landmarks for some frames and nothing for others."""

    def __init__(self, camera, pattern):
        self.camera = camera
        self.pattern = pattern
        self.i = 0

    def __call__(self, frame):
        ok = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return observed(self.camera, GOOD_SET) if ok else {}


def frames(n, shape=(1080, 1920, 3)):
    return [np.zeros(shape, np.uint8) for _ in range(n)]


def test_mapper_solves_every_good_frame(camera):
    m = PitchMapper(FakeKeypoints(camera, [True]))
    for f in frames(10):
        assert m.update(f) is not None
    assert m.solved == 10 and m.coverage == 1.0


def test_mapper_bridges_a_short_dropout(camera):
    """Landmarks drop out as the camera moves; brief gaps should not lose the map."""
    m = PitchMapper(FakeKeypoints(camera, [True, False, False]), max_stale_frames=5)
    results = [m.update(f) for f in frames(9)]
    assert all(r is not None for r in results)
    assert m.reused == 6


def test_mapper_gives_up_on_a_long_dropout(camera):
    """A stale transform drifts -- the camera moved. Better to admit no mapping."""
    m = PitchMapper(FakeKeypoints(camera, [True] + [False] * 20), max_stale_frames=3)
    results = [m.update(f) for f in frames(10)]
    assert results[0] is not None
    assert results[-1] is None, "stale transform should have been abandoned"
    assert m.failed > 0


def test_mapper_recovers_after_a_failure(camera):
    m = PitchMapper(FakeKeypoints(camera, [True, False, False, False, False]),
                    max_stale_frames=1)
    results = [m.update(f) for f in frames(15)]
    assert results[-5] is not None or results[-4] is not None


def test_coverage_reports_the_mapping_rate(camera):
    m = PitchMapper(FakeKeypoints(camera, [True, False]), max_stale_frames=1)
    for f in frames(10):
        m.update(f)
    assert 0.0 < m.coverage <= 1.0


def test_mapper_with_no_keypoints_at_all():
    m = PitchMapper(lambda frame: {})
    assert m.update(np.zeros((1080, 1920, 3), np.uint8)) is None
    assert m.coverage == 0.0


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "xy, expected",
    [
        ((50.0, 34.0), True),
        ((-2.0, 34.0), True),        # player stepping over the touchline
        ((107.0, 34.0), True),
        ((-40.0, 34.0), False),      # mapping blunder
        ((50.0, 200.0), False),
    ],
)
def test_on_pitch(xy, expected):
    assert on_pitch(xy) is expected
