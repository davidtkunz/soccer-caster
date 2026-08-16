"""Tests for perception glue, using fake models.

The detection weights are the heavy, environment-specific part; the logic
around them -- slicing, tracking, clustering, outlier handling -- is ordinary
code, and it is where the bugs are. Injecting the models means all of it can be
exercised without a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from caster.perception import (
    FrameDetections,
    Perception,
    TeamAssigner,
    _kmeans,
    crop_detections,
    sample_frames_for_fitting,
)


def dets(boxes, confidence=None):
    xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    conf = (
        np.asarray(confidence, dtype=np.float32)
        if confidence is not None
        else np.ones(len(xyxy), dtype=np.float32)
    )
    return sv.Detections(xyxy=xyxy, confidence=conf,
                         class_id=np.zeros(len(xyxy), dtype=int))


class FakeTracker:
    """Assigns stable ids by box order -- enough to test the wiring."""

    def update(self, detections, frame=None, timestamp=None):
        detections.tracker_id = np.arange(1, len(detections) + 1)
        return detections


def colour_embedder(crops):
    """Mean BGR of each crop. Real enough: kit colour is the actual signal."""
    return np.stack([c.reshape(-1, 3).mean(axis=0) for c in crops])


def solid(colour, size=12):
    return np.full((size, size, 3), colour, np.uint8)


# --------------------------------------------------------------------------
# k-means
# --------------------------------------------------------------------------


def test_kmeans_separates_two_obvious_groups():
    x = np.array([[0.0, 0.0], [0.5, 0.2], [10.0, 10.0], [10.4, 9.8]])
    labels, centres = _kmeans(x, k=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert centres.shape == (2, 2)


def test_kmeans_is_deterministic():
    """Random init would make team labels differ run to run on the same match."""
    x = np.random.default_rng(0).normal(size=(40, 3))
    a, _ = _kmeans(x, k=2, seed=7)
    b, _ = _kmeans(x, k=2, seed=7)
    assert np.array_equal(a, b)


def test_kmeans_needs_enough_samples():
    with pytest.raises(ValueError):
        _kmeans(np.array([[1.0, 1.0]]), k=2)


# --------------------------------------------------------------------------
# Team assignment
# --------------------------------------------------------------------------


@pytest.fixture
def kits():
    red = [solid((30, 30, 200)) for _ in range(10)]
    blue = [solid((200, 40, 30)) for _ in range(10)]
    return red, blue


@pytest.fixture
def assigner(kits):
    red, blue = kits
    return TeamAssigner(colour_embedder).fit(red + blue)


def test_team_assignment_separates_kits(assigner, kits):
    red, blue = kits
    labels = assigner.predict(red[:3] + blue[:3])
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert set(labels) <= {"home", "away"}


def test_labels_are_stable_across_calls(assigner, kits):
    """Refitting per frame would swap labels and read as constant turnovers."""
    red, blue = kits
    first = assigner.predict(red[:2] + blue[:2])
    for _ in range(5):
        assert assigner.predict(red[:2] + blue[:2]) == first


def test_referee_is_returned_as_unknown_not_forced_into_a_team(assigner):
    """A yellow shirt matches neither kit -- better None than a wrong team."""
    labels = assigner.predict([solid((0, 230, 230))])
    assert labels == [None]


def test_outlier_scale_survives_perfectly_uniform_kits(kits):
    """Regression: the scale must come from kit separation, not cluster spread.

    Uniform kits give zero within-cluster spread, which makes a spread-derived
    threshold degenerate -- it either vanishes and rejects everyone, or is
    guarded to infinity and rejects no one. The gap between the two kits is
    always meaningful.
    """
    red, blue = kits
    a = TeamAssigner(colour_embedder).fit(red + blue)
    assert np.isfinite(a._limit) and a._limit > 0
    assert a.predict([solid((0, 230, 230))]) == [None]      # referee
    assert a.predict(red[:1]) != [None]                     # real player kept


def test_predict_before_fit_is_an_error(kits):
    with pytest.raises(RuntimeError, match="fit"):
        TeamAssigner(colour_embedder).predict(kits[0])


def test_predict_on_nothing():
    assert TeamAssigner(colour_embedder).predict([]) == []


# --------------------------------------------------------------------------
# Crops
# --------------------------------------------------------------------------


def test_crops_follow_the_boxes():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[10:30, 10:30] = (0, 0, 255)
    crops = crop_detections(frame, dets([[10, 10, 30, 30]]))
    assert crops[0].shape == (20, 20, 3)
    assert crops[0][..., 2].mean() == 255


def test_crops_are_clipped_to_the_frame():
    frame = np.zeros((50, 50, 3), np.uint8)
    crops = crop_detections(frame, dets([[40, 40, 200, 200], [-20, -20, 10, 10]]))
    assert all(c.size > 0 for c in crops)


def test_degenerate_box_does_not_crash():
    frame = np.zeros((50, 50, 3), np.uint8)
    assert crop_detections(frame, dets([[10, 10, 10, 10]]))[0].size > 0


# --------------------------------------------------------------------------
# Ball selection
# --------------------------------------------------------------------------


def test_highest_confidence_ball_is_used():
    fd = FrameDetections(
        players=dets([]),
        ball=dets([[0, 0, 10, 10], [90, 90, 100, 100]], confidence=[0.2, 0.9]),
    )
    assert fd.ball_xy == (95.0, 95.0)


def test_no_ball_detection_is_none_not_a_guess():
    assert FrameDetections(players=dets([]), ball=dets([])).ball_xy is None


# --------------------------------------------------------------------------
# Perception wiring
# --------------------------------------------------------------------------


@pytest.fixture
def perception(assigner):
    frame_players = dets([[10, 10, 22, 22], [60, 60, 72, 72]])

    return Perception(
        player_model=lambda f: frame_players,
        ball_model=lambda tile: dets([]),
        team_assigner=assigner,
        tracker=FakeTracker(),
        ball_slice_wh=(64, 64),
        ball_overlap_wh=(16, 16),
    )


def test_process_returns_tracked_players_with_teams(perception):
    frame = np.zeros((128, 128, 3), np.uint8)
    frame[10:22, 10:22] = (30, 30, 200)      # red kit
    frame[60:72, 60:72] = (200, 40, 30)      # blue kit

    out = perception.process(frame)
    assert len(out.players) == 2
    assert list(out.players.tracker_id) == [1, 2]
    assert out.teams[0] != out.teams[1]
    assert out.ball_xy is None


def test_teams_are_all_none_without_an_assigner():
    p = Perception(
        player_model=lambda f: dets([[0, 0, 10, 10]]),
        ball_model=lambda t: dets([]),
        team_assigner=None,
        tracker=FakeTracker(),
    )
    assert p.process(np.zeros((64, 64, 3), np.uint8)).teams == [None]


def test_ball_model_is_called_per_tile_not_per_frame():
    """Sliced inference is the point -- a single full-frame pass misses the
    ball at the far end of the pitch."""
    calls = []

    def ball_model(tile):
        calls.append(tile.shape)
        return dets([])

    p = Perception(
        player_model=lambda f: dets([]),
        ball_model=ball_model,
        tracker=FakeTracker(),
        ball_slice_wh=(64, 64),
        ball_overlap_wh=(0, 0),
    )
    p.detect_ball(np.zeros((128, 128, 3), np.uint8))
    assert len(calls) > 1, f"expected multiple tiles, got {len(calls)}"


def test_empty_frame_yields_nothing_invented():
    p = Perception(
        player_model=lambda f: dets([]),
        ball_model=lambda t: dets([]),
        tracker=FakeTracker(),
    )
    out = p.process(np.zeros((64, 64, 3), np.uint8))
    assert len(out.players) == 0
    assert out.ball_xy is None
    assert out.teams == []


# --------------------------------------------------------------------------
# Fitting sample
# --------------------------------------------------------------------------


def test_fitting_sample_is_spread_not_consecutive():
    """One passage of play can contain only one team, which clusters wrongly."""
    frames = list(range(1000))
    sample = sample_frames_for_fitting(frames, every=50, limit=10)
    assert sample == [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]


def test_fitting_sample_respects_the_limit():
    assert len(sample_frames_for_fitting(list(range(10000)), every=1, limit=25)) == 25
