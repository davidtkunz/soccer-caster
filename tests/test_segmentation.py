"""Tests for shot-boundary and replay detection.

The load-bearing case is `test_replay_is_excluded_from_live`. Everything else
in the pipeline trusts that replays never reach it, and a regression there
produces duplicate goals rather than an obvious crash.
"""

from __future__ import annotations

import numpy as np
import pytest

import synth
from caster.segmentation import (
    GREEN_SOME,
    GREEN_WIDE,
    ScoreboardDetector,
    Segmenter,
    ShotClass,
    classify,
    coverage,
    edge_density,
    find_cuts,
    frame_signature,
    green_ratio,
    live_segments,
    signature_distance,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def calibrated_detector(rng):
    det = ScoreboardDetector()
    assert det.calibrate([synth.pitch_frame(rng, with_bug=True) for _ in range(40)])
    return det


# --------------------------------------------------------------------------
# Frame features
# --------------------------------------------------------------------------


def test_green_ratio_separates_pitch_from_non_pitch(rng):
    pitch = np.mean([green_ratio(synth.pitch_frame(rng)) for _ in range(5)])
    crowd = np.mean([green_ratio(synth.crowd_frame(rng)) for _ in range(5)])
    graphic = np.mean([green_ratio(synth.graphic_frame(rng)) for _ in range(5)])

    assert pitch >= GREEN_WIDE
    assert crowd < GREEN_SOME
    assert graphic < GREEN_SOME


def test_closeup_lands_between_thresholds(rng):
    """A close-up shows pitch without being dominated by it."""
    g = np.mean([green_ratio(synth.closeup_frame(rng)) for _ in range(5)])
    assert GREEN_SOME < g < GREEN_WIDE


def test_edge_density_separates_crowd_from_graphic(rng):
    crowd = np.mean([edge_density(synth.crowd_frame(rng)) for _ in range(5)])
    graphic = np.mean([edge_density(synth.graphic_frame(rng)) for _ in range(5)])
    assert crowd > graphic * 2


def test_signature_distance_is_small_within_shot_and_large_across(rng):
    a, b = synth.pitch_frame(rng), synth.pitch_frame(rng)
    c = synth.crowd_frame(rng)
    within = signature_distance(frame_signature(a), frame_signature(b))
    across = signature_distance(frame_signature(a), frame_signature(c))
    assert within < across
    assert across > 0.5


# --------------------------------------------------------------------------
# Scoreboard
# --------------------------------------------------------------------------


def test_calibration_locates_the_bug(calibrated_detector):
    x, y, w, h = calibrated_detector.roi
    tx, ty, tw, th = synth.BUG_RECT

    # Overlap, not exact match: the detector finds the quiet region, which need
    # not align to the drawn rectangle's edges.
    ox = max(0, min(x + w, tx + tw) - max(x, tx))
    oy = max(0, min(y + h, ty + th) - max(y, ty))
    assert ox > 0 and oy > 0, "ROI does not overlap the drawn bug at all"
    assert (ox * oy) / (tw * th) > 0.25, "ROI covers too little of the bug"


def test_bug_present_on_live_absent_on_replay(calibrated_detector, rng):
    live = [calibrated_detector.score(synth.pitch_frame(rng, True)) for _ in range(10)]
    replay = [calibrated_detector.score(synth.replay_frame(rng)) for _ in range(10)]

    assert min(live) > calibrated_detector.present_threshold
    assert max(replay) < calibrated_detector.present_threshold
    assert min(live) - max(replay) > 0.3, "separation is too tight to be reliable"


def test_calibration_reports_failure_when_no_overlay_exists(rng):
    """Footage with no score bug must not yield a confident bogus ROI."""
    det = ScoreboardDetector()
    ok = det.calibrate([synth.replay_frame(rng) for _ in range(40)])
    assert ok is False
    assert not det.calibrated


def test_calibration_rejects_a_static_but_flat_region(rng):
    """A quiet region is not enough -- an overlay is quiet AND graphical.

    Regression guard. A variance threshold alone accepts any still corner of
    the pitch, which yields a bogus ROI whose presence score is then noise. On
    real footage that silently turns replay detection off.
    """
    frames = []
    for _ in range(40):
        f = synth.pitch_frame(rng, with_bug=False)
        f[0:30, 0:80] = (58, 138, 58)     # static, flat, no structure
        frames.append(f)

    det = ScoreboardDetector()
    assert det.calibrate(frames) is False
    assert not det.calibrated


def test_calibration_needs_enough_frames(rng):
    det = ScoreboardDetector()
    assert det.calibrate([synth.pitch_frame(rng) for _ in range(3)]) is False


# --------------------------------------------------------------------------
# Cut detection
# --------------------------------------------------------------------------


def test_find_cuts_on_flat_input_returns_nothing():
    assert find_cuts(np.zeros(200)) == []


def test_find_cuts_locates_boundaries(rng):
    plan = [("live", 40), ("crowd", 35), ("live", 40), ("graphic", 35)]
    frames, truth = synth.build_sequence(plan, seed=7)

    sigs = [frame_signature(f) for f in frames]
    dists = np.array(
        [signature_distance(sigs[i], sigs[i + 1]) for i in range(len(sigs) - 1)]
    )
    cuts = find_cuts(dists, min_shot_frames=12)

    assert len(cuts) == len(truth), f"expected {truth}, got {cuts}"
    for got, want in zip(cuts, truth):
        assert abs(got - want) <= 2, f"cut at {got}, expected near {want}"


def test_find_cuts_respects_minimum_shot_length():
    dists = np.full(200, 0.01)
    dists[[50, 52, 54]] = 0.9          # one transition smeared over 3 frames
    cuts = find_cuts(dists, min_shot_frames=12)
    assert len(cuts) == 1


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "features, expected",
    [
        ({"green_ratio": 0.8, "scoreboard": True}, ShotClass.LIVE_WIDE),
        ({"green_ratio": 0.8, "scoreboard": False}, ShotClass.REPLAY),
        ({"green_ratio": 0.3, "scoreboard": True}, ShotClass.CLOSE_UP),
        ({"green_ratio": 0.02, "edge_density": 0.2}, ShotClass.CROWD),
        ({"green_ratio": 0.02, "edge_density": 0.01}, ShotClass.GRAPHIC),
    ],
)
def test_classify_rules(features, expected):
    assert classify(features) is expected


def test_uncalibrated_classifier_assumes_live_rather_than_dropping_footage():
    """Without a bug to check, pitch footage is kept rather than discarded."""
    got = classify({"green_ratio": 0.8, "scoreboard": False},
                   scoreboard_available=False)
    assert got is ShotClass.LIVE_WIDE


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


@pytest.fixture
def analysed(calibrated_detector):
    plan = [
        ("live", 40),
        ("replay", 30),
        ("live", 40),
        ("crowd", 30),
        ("closeup", 30),
        ("live", 40),
        ("graphic", 30),
    ]
    frames, truth = synth.build_sequence(plan, seed=11)
    seg = Segmenter(min_shot_frames=12, feature_stride=5,
                    scoreboard=calibrated_detector)
    return seg.analyze(frames), plan, truth


def test_segment_count_and_classes(analysed):
    segments, plan, _ = analysed
    expected = [
        ShotClass.LIVE_WIDE,
        ShotClass.REPLAY,
        ShotClass.LIVE_WIDE,
        ShotClass.CROWD,
        ShotClass.CLOSE_UP,
        ShotClass.LIVE_WIDE,
        ShotClass.GRAPHIC,
    ]
    got = [s.kind for s in segments]
    assert got == expected, f"got {[g.value for g in got]}"


def test_segments_tile_the_footage_without_gaps(analysed):
    segments, plan, _ = analysed
    total = sum(n for _, n in plan)
    assert segments[0].start_frame == 0
    assert segments[-1].end_frame == total
    for a, b in zip(segments, segments[1:]):
        assert a.end_frame == b.start_frame


def test_replay_is_excluded_from_live(analysed):
    """The one that matters: a replay of pitch footage must not reach the
    event pipeline, or the goal it shows is counted twice."""
    segments, _, _ = analysed
    live = live_segments(segments)

    assert all(s.kind is ShotClass.LIVE_WIDE for s in live)
    assert len(live) == 3

    replay = next(s for s in segments if s.kind is ShotClass.REPLAY)
    assert replay not in live
    # And it really is pitch footage -- green ratio alone could not have caught it.
    assert replay.features["green_ratio"] >= GREEN_WIDE


def test_coverage_sums_to_one(analysed):
    segments, _, _ = analysed
    frac = coverage(segments)
    assert pytest.approx(sum(frac.values()), abs=1e-6) == 1.0
    assert frac[ShotClass.LIVE_WIDE] > frac.get(ShotClass.REPLAY, 0)


def test_empty_input():
    assert Segmenter().analyze([]) == []


def test_single_frame_input(rng):
    segments = Segmenter().analyze([synth.pitch_frame(rng)])
    assert len(segments) == 1
    assert segments[0].n_frames == 1
