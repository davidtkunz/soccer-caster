"""Integration: state -> events -> summary over a synthetic match.

Each module is unit-tested in isolation; this checks they actually compose.
The gap between here and a full end-to-end run is perception (frames to pitch
coordinates), which is still a skeleton -- so this starts from tracked state.
"""

from __future__ import annotations

import pytest

from caster.events import EventDetector, EventType, PitchModel
from caster.state import (
    FrameObservation,
    PossessionTracker,
    build_state,
)
from caster.summary import build, format_report

DT = 1.0 / 25.0


def synthetic_match():
    """A short passage of play: two passes, a shot, and a goal."""
    obs: list[FrameObservation] = []
    t = 0.0
    score = (0, 0)

    def frame(ball, players):
        nonlocal t
        o = FrameObservation(int(t * 25), t, ball, players)
        t += DT
        obs.append(o)

    home_a = (40.0, 34.0, "home")
    home_b = (75.0, 30.0, "home")
    keeper = (102.0, 34.0, "away")

    players = {1: home_a, 2: home_b, 9: keeper}

    # Player 1 holds the ball
    for _ in range(20):
        frame((40.0, 34.0), players)

    # Pass to player 2 -- ball in flight, nobody possesses it
    for i in range(10):
        frame((40.0 + 3.5 * i, 34.0 - 0.4 * i), players)

    # Player 2 receives in the attacking third
    for _ in range(20):
        frame((75.0, 30.0), players)

    # Shot: fast, on target, from the attacking third
    for i in range(12):
        frame((75.0 + 20.0 * i * DT, 30.0 + 3.0 * i * DT), players)

    return obs, score


def test_state_events_and_summary_compose():
    obs, _ = synthetic_match()
    states = list(build_state(obs, possession=PossessionTracker(hold_frames=3)))

    detector = EventDetector(PitchModel())
    events = [e for s in states for e in detector.update(s)]

    kinds = {e.type for e in events}
    assert EventType.PASS in kinds, f"no pass detected in {events}"
    assert EventType.SHOT in kinds, f"no shot detected in {events}"

    summary = build(states, events)
    assert summary.possession_pct["home"] == pytest.approx(1.0)
    assert summary.players[1].touches >= 1
    assert summary.duration_s > 0

    report = format_report(summary)
    assert "Possession:" in report
    assert "shot" in report


def test_pipeline_survives_a_completely_empty_match():
    states = list(build_state([]))
    detector = EventDetector()
    events = [e for s in states for e in detector.update(s)]
    summary = build(states, events)
    assert summary.score is None
    assert format_report(summary)


def test_pipeline_survives_frames_with_no_detections():
    """Perception drops out -- no ball, no players. Nothing should be invented."""
    obs = [FrameObservation(i, i * DT, None, {}) for i in range(50)]
    states = list(build_state(obs))
    detector = EventDetector()
    events = [e for s in states for e in detector.update(s)]
    assert events == []
    assert build(states, events).possession_pct == {}
