"""Tests for game state: possession hysteresis, scoreboard reading, smoothing.

The two load-bearing behaviours are that possession does not flicker (every
flicker reads as a pass or turnover downstream) and that a single-frame digit
misread cannot move the score (which would read as a goal).
"""

from __future__ import annotations

import numpy as np
import pytest

import synth
from caster.segmentation import ScoreboardDetector
from caster.state import (
    FrameObservation,
    PossessionTracker,
    ScoreboardLayout,
    ScoreboardReader,
    StableValue,
    build_state,
    clock_transition_valid,
    read_number,
    score_transition_valid,
    segment_digits,
    synthesize_digit_templates,
)


@pytest.fixture
def rng():
    return np.random.default_rng(5)


@pytest.fixture
def templates():
    return synthesize_digit_templates()


@pytest.fixture
def detector(rng):
    det = ScoreboardDetector()
    assert det.calibrate([synth.scoreboard_frame(rng, 2, 1, m) for m in range(0, 80, 2)])
    return det


@pytest.fixture
def reader(detector, templates):
    # read_roi is the true bug rectangle; the detector's own ROI is only
    # accurate enough for the presence gate. See ScoreboardReader's docstring.
    return ScoreboardReader(
        detector,
        ScoreboardLayout(**synth.READABLE_LAYOUT),
        templates,
        confirm_frames=5,
        read_roi=synth.READABLE_BUG_RECT,
    )


# --------------------------------------------------------------------------
# StableValue
# --------------------------------------------------------------------------


def test_value_is_not_accepted_until_it_holds():
    sv = StableValue(confirm_frames=3)
    assert sv.update("a") is None
    assert sv.update("a") is None
    assert sv.update("a") == "a"


def test_flapping_readings_never_confirm():
    sv = StableValue(confirm_frames=3)
    for reading in ["a", "b", "a", "b", "a", "b"]:
        assert sv.update(reading) is None


def test_missing_readings_are_ignored_not_treated_as_change():
    sv = StableValue(confirm_frames=2)
    sv.update("a"), sv.update("a")
    assert sv.value == "a"
    for _ in range(20):
        assert sv.update(None) == "a"


def test_validator_rejects_impossible_transitions():
    sv = StableValue(confirm_frames=2, validator=score_transition_valid)
    sv.update((0, 0)), sv.update((0, 0))
    assert sv.value == (0, 0)

    for _ in range(30):
        sv.update((5, 0))           # impossible jump, sustained
    assert sv.value == (0, 0)
    assert sv.rejected == 30


def test_valid_transition_still_confirms_after_rejections():
    sv = StableValue(confirm_frames=2, validator=score_transition_valid)
    sv.update((0, 0)), sv.update((0, 0))
    sv.update((9, 9))               # rejected
    sv.update((1, 0)), sv.update((1, 0))
    assert sv.value == (1, 0)


# --------------------------------------------------------------------------
# Transition validators
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old, new, ok",
    [
        ((0, 0), (1, 0), True),
        ((2, 1), (2, 2), True),
        ((2, 1), (2, 1), True),
        ((2, 1), (1, 1), False),     # score cannot go down
        ((2, 1), (4, 1), False),     # two goals at once
        ((2, 1), (3, 2), False),     # both teams at once
    ],
)
def test_score_transition_rules(old, new, ok):
    assert score_transition_valid(old, new) is ok


@pytest.mark.parametrize(
    "old, new, ok",
    [(600, 601, True), (600, 600, True), (600, 599, False), (600, 4200, False)],
)
def test_clock_transition_rules(old, new, ok):
    assert clock_transition_valid(old, new) is ok


# --------------------------------------------------------------------------
# Possession
# --------------------------------------------------------------------------


def _players(**kw):
    """Helper: {track_id: (x, y, team)} from id=(x, y, team)."""
    return {int(k[1:]): v for k, v in kw.items()}


def test_possession_claimed_after_hold_frames():
    pt = PossessionTracker(hold_frames=3)
    players = _players(p1=(0.0, 0.0, "blue"))
    for _ in range(2):
        assert pt.update((0.2, 0.0), players)[0] is None
    assert pt.update((0.2, 0.0), players) == (1, "blue")


def test_possession_does_not_flicker_between_close_players():
    """The core reason this class exists.

    Two players either side of the ball, alternating by centimetres. Naive
    nearest-player assignment changes holder on most frames; each change would
    read as a pass downstream.
    """
    pt = PossessionTracker(hold_frames=5, challenge_margin_m=0.5)
    for _ in range(10):
        pt.update((0.0, 0.0), _players(p1=(0.4, 0.0, "blue"), p2=(0.5, 0.0, "red")))
    assert pt.holder == 1

    changes = 0
    prev = pt.holder
    for i in range(60):
        jitter = 0.06 * (1 if i % 2 else -1)
        holder, _ = pt.update(
            (0.0, 0.0),
            _players(p1=(0.45 + jitter, 0.0, "blue"), p2=(0.45 - jitter, 0.0, "red")),
        )
        changes += holder != prev
        prev = holder

    assert changes == 0, f"possession flickered {changes} times"


def test_clearly_closer_challenger_takes_possession():
    pt = PossessionTracker(hold_frames=3, challenge_margin_m=0.5)
    for _ in range(5):
        pt.update((0.0, 0.0), _players(p1=(0.3, 0.0, "blue"), p2=(1.8, 0.0, "red")))
    assert pt.holder == 1

    for _ in range(3):
        pt.update((0.0, 0.0), _players(p1=(1.8, 0.0, "blue"), p2=(0.2, 0.0, "red")))
    assert pt.holder == 2
    assert pt.team == "red"


def test_ball_out_of_range_goes_loose_after_delay():
    pt = PossessionTracker(hold_frames=2, loose_frames=4)
    for _ in range(4):
        pt.update((0.0, 0.0), _players(p1=(0.3, 0.0, "blue")))
    assert pt.holder == 1

    for _ in range(3):
        pt.update((50.0, 0.0), _players(p1=(0.3, 0.0, "blue")))
    assert pt.holder == 1, "released too eagerly"

    pt.update((50.0, 0.0), _players(p1=(0.3, 0.0, "blue")))
    assert pt.holder is None


def test_missing_ball_does_not_immediately_release():
    pt = PossessionTracker(hold_frames=2, loose_frames=5)
    for _ in range(3):
        pt.update((0.0, 0.0), _players(p1=(0.3, 0.0, "blue")))
    assert pt.holder == 1
    for _ in range(4):
        pt.update(None, _players(p1=(0.3, 0.0, "blue")))
    assert pt.holder == 1


# --------------------------------------------------------------------------
# Digit reading
# --------------------------------------------------------------------------


def test_single_digit_field_is_readable(rng, templates):
    """Regression: aspect-ratio filtering, not width-relative-to-crop.

    A tight single-digit field is barely wider than its digit, so a
    width-versus-crop rule discards the glyph it should keep -- and does so
    silently, leaving the score permanently unread while the wider clock field
    keeps working.
    """
    layout = ScoreboardLayout(**synth.READABLE_LAYOUT)
    frame = synth.scoreboard_frame(rng, home=7, away=3, minute=22)
    x, y, w, h = synth.READABLE_BUG_RECT
    roi = frame[y : y + h, x : x + w]

    assert len(segment_digits(layout.crop(roi, layout.home_score))) == 1
    assert read_number(layout.crop(roi, layout.home_score), templates) == 7
    assert read_number(layout.crop(roi, layout.away_score), templates) == 3
    assert read_number(layout.crop(roi, layout.clock), templates) == 22


def test_all_digits_read_correctly(rng, reader):
    misses = []
    for home in range(10):
        for away in (0, 1, 7):
            frame = synth.scoreboard_frame(rng, home, away, 45)
            got, _ = reader.read_raw(frame)
            if got != (home, away):
                misses.append(((home, away), got))
    assert not misses, f"{len(misses)} misreads: {misses[:5]}"


def test_missing_bug_reads_as_none(rng, reader):
    """A replay has no bug. Reading the pitch instead fabricates a scoreline.

    Regression. Without the presence gate the digit filters find contours in
    the grass and return a plausible-looking score. The transition validator
    cannot catch it either -- it only constrains changes from a known value, so
    a hallucinated *first* reading is accepted outright.
    """
    score, clock = reader.read_raw(synth.scoreboard_frame(rng, with_bug=False))
    assert score is None and clock is None


def test_reading_uses_the_authored_roi_not_the_detector_roi(detector, templates):
    """Regression: presence and reading need different rectangles.

    The detector's ROI is grown from whichever cells were quietest, so it is
    offset and clipped relative to the real bug -- fine for a whole-region
    presence correlation, useless for locating individual digit fields. If the
    reader silently falls back to it, every field crop lands on the wrong pixels
    and the score simply never reads.
    """
    assert detector.roi != synth.READABLE_BUG_RECT, "fixture no longer exercises this"

    reader = ScoreboardReader(
        detector,
        ScoreboardLayout(**synth.READABLE_LAYOUT),
        templates,
        read_roi=synth.READABLE_BUG_RECT,
    )
    assert reader.roi == synth.READABLE_BUG_RECT

    rng = np.random.default_rng(99)
    score, clock = reader.read_raw(synth.scoreboard_frame(rng, 4, 2, 63))
    assert score == (4, 2)
    assert clock == 63 * 60


def test_reader_requires_a_calibrated_detector(templates):
    with pytest.raises(ValueError, match="calibrated"):
        ScoreboardReader(
            ScoreboardDetector(), ScoreboardLayout(**synth.READABLE_LAYOUT), templates
        )


# --------------------------------------------------------------------------
# Reader + smoothing
# --------------------------------------------------------------------------


def test_goal_is_picked_up_after_confirmation(rng, reader):
    for _ in range(8):
        score, _ = reader.update(synth.scoreboard_frame(rng, 2, 1, 60))
    assert score == (2, 1)

    for _ in range(8):
        score, _ = reader.update(synth.scoreboard_frame(rng, 2, 2, 61))
    assert score == (2, 2)


def test_transient_misread_cannot_move_the_score(rng, reader):
    """A one-frame digit error must not look like a goal."""
    for _ in range(8):
        reader.update(synth.scoreboard_frame(rng, 2, 1, 60))
    assert reader.score.value == (2, 1)

    reader.update(synth.scoreboard_frame(rng, 8, 1, 60))       # single bad frame
    assert reader.score.value == (2, 1)

    for _ in range(6):
        reader.update(synth.scoreboard_frame(rng, 2, 1, 60))
    assert reader.score.value == (2, 1)


def test_frames_without_a_bug_do_not_clear_the_score(rng, reader):
    for _ in range(8):
        reader.update(synth.scoreboard_frame(rng, 3, 0, 70))
    assert reader.score.value == (3, 0)

    for _ in range(20):
        reader.update(synth.scoreboard_frame(rng, with_bug=False))
    assert reader.score.value == (3, 0)


# --------------------------------------------------------------------------
# build_state
# --------------------------------------------------------------------------


def test_build_state_produces_one_state_per_observation():
    obs = [
        FrameObservation(i, i / 25.0, (0.0, 0.0), _players(p1=(0.3, 0.0, "blue")))
        for i in range(20)
    ]
    states = list(build_state(obs))
    assert len(states) == 20
    assert states[-1].possessing_team == "blue"
    assert states[0].frame_idx == 0


def test_build_state_tracks_a_pass_as_one_change():
    """Ball moves from one player to a team-mate: exactly one holder change."""
    tracker = PossessionTracker(hold_frames=3, challenge_margin_m=0.5)
    players = _players(p1=(0.0, 0.0, "blue"), p2=(20.0, 0.0, "blue"))

    obs = []
    for i in range(40):
        bx = 0.0 if i < 20 else 20.0        # ball is played across
        obs.append(FrameObservation(i, i / 25.0, (bx, 0.0), players))

    holders = [s.possessing_track_id for s in build_state(obs, possession=tracker)]
    # Skip the leading Nones: first acquisition is not a pass, it is the ball
    # coming into play. Only holder-to-holder changes count.
    settled = [h for h in holders if h is not None]
    changes = sum(a != b for a, b in zip(settled, settled[1:]))
    assert changes == 1, f"expected one handover, saw {changes}"
    assert settled[0] == 1 and settled[-1] == 2


def test_clock_formatting():
    obs = [FrameObservation(0, 0.0, None, {})]
    state = next(iter(build_state(obs)))
    assert state.clock is None
    state.clock_s = 2715
    assert state.clock == "45:15"
