"""Tests for the post-game summary.

The metric most likely to be quietly wrong is distance covered: tracking-ID
switches teleport an identity across the pitch, and without a guard the figure
is dominated by those jumps rather than by running.
"""

from __future__ import annotations

import numpy as np
import pytest

from caster.events import Event, EventType
from caster.state import GameState
from caster.summary import (
    MAX_PLAUSIBLE_SPEED_MPS,
    accumulate_player_stats,
    build,
    build_heatmaps,
    build_timeline,
    count_events,
    format_report,
    possession_share,
)

DT = 1.0 / 25.0


def st(t, players=None, team=None, holder=None, score=None):
    return GameState(
        t=t,
        frame_idx=int(t * 25),
        ball_xy=(50.0, 34.0),
        players=players or {},
        possessing_track_id=holder,
        possessing_team=team,
        score=score,
    )


def walk(track_id, team, start, speed_mps, n, t0=0.0):
    """States for one player moving in a straight line at a constant speed."""
    return [
        st(
            t0 + i * DT,
            players={track_id: (start[0] + speed_mps * i * DT, start[1], team)},
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Possession
# --------------------------------------------------------------------------


def test_possession_share_sums_to_one():
    states = [st(i * DT, team="home") for i in range(70)]
    states += [st((70 + i) * DT, team="away") for i in range(30)]
    share = possession_share(states)
    assert share["home"] == pytest.approx(0.7)
    assert share["away"] == pytest.approx(0.3)
    assert sum(share.values()) == pytest.approx(1.0)


def test_loose_ball_frames_are_excluded_not_charged_to_a_team():
    states = [st(i * DT, team="home") for i in range(50)]
    states += [st((50 + i) * DT, team=None) for i in range(200)]
    states += [st((250 + i) * DT, team="away") for i in range(50)]
    share = possession_share(states)
    assert share["home"] == pytest.approx(0.5)
    assert share["away"] == pytest.approx(0.5)


def test_possession_share_of_nothing():
    assert possession_share([st(i * DT) for i in range(10)]) == {}


# --------------------------------------------------------------------------
# Player stats
# --------------------------------------------------------------------------


def test_distance_matches_constant_speed_motion():
    states = walk(1, "home", (0.0, 34.0), speed_mps=5.0, n=51)   # 2 s at 5 m/s
    stats = accumulate_player_stats(states)
    assert stats[1].distance_m == pytest.approx(10.0, abs=0.2)
    assert stats[1].top_speed_mps == pytest.approx(5.0, abs=0.2)
    assert stats[1].team == "home"


def test_tracking_teleport_is_rejected_from_distance():
    """Regression: an ID switch must not read as several kilometres of running.

    Without the guard this single jump adds ~90 m in one frame -- more than the
    player covers in the entire rest of the sequence.
    """
    states = walk(1, "home", (0.0, 34.0), speed_mps=5.0, n=25)
    states.append(st(25 * DT, players={1: (95.0, 34.0, "home")}))   # switch
    states += [
        st((26 + i) * DT, players={1: (95.0 + 5.0 * i * DT, 34.0, "home")})
        for i in range(25)
    ]

    stats = accumulate_player_stats(states)
    assert stats[1].teleports == 1
    assert stats[1].distance_m < 15.0, "teleport leaked into distance"
    assert stats[1].top_speed_mps <= MAX_PLAUSIBLE_SPEED_MPS


def test_one_continuous_run_is_one_sprint():
    """A four-second run is one sprint, not four.

    Counting per sustained second would make this metric a restatement of
    high-speed distance rather than a count of efforts.
    """
    states = walk(1, "home", (0.0, 34.0), speed_mps=9.0, n=100)   # 4 s
    assert accumulate_player_stats(states)[1].sprints == 1


def test_separate_runs_count_separately():
    states = walk(1, "home", (0.0, 34.0), speed_mps=9.0, n=50, t0=0.0)
    slow = walk(1, "home", (18.0, 34.0), speed_mps=1.0, n=50, t0=2.0)
    fast = walk(1, "home", (20.0, 34.0), speed_mps=9.0, n=50, t0=4.0)
    assert accumulate_player_stats(states + slow + fast)[1].sprints == 2


def test_brief_burst_is_not_a_sprint():
    states = walk(1, "home", (0.0, 34.0), speed_mps=9.0, n=10)    # 0.4 s
    assert accumulate_player_stats(states)[1].sprints == 0


def test_jogging_is_not_a_sprint():
    states = walk(1, "home", (0.0, 34.0), speed_mps=3.0, n=100)
    assert accumulate_player_stats(states)[1].sprints == 0


def test_stats_track_multiple_players_independently():
    states = []
    for i in range(50):
        states.append(
            st(i * DT, players={
                1: (5.0 * i * DT, 34.0, "home"),
                2: (50.0, 20.0, "away"),          # stationary
            })
        )
    stats = accumulate_player_stats(states)
    assert stats[1].distance_m > 9.0
    assert stats[2].distance_m == pytest.approx(0.0, abs=1e-6)
    assert stats[2].team == "away"


# --------------------------------------------------------------------------
# Heatmaps and events
# --------------------------------------------------------------------------


def test_heatmap_shape_and_mass():
    states = walk(1, "home", (10.0, 34.0), speed_mps=5.0, n=40)
    maps = build_heatmaps(states, bins=(24, 16))
    assert maps[1].shape == (24, 16)
    assert maps[1].sum() == 40


def test_heatmap_puts_a_stationary_player_in_one_cell():
    states = [st(i * DT, players={1: (10.0, 20.0, "home")}) for i in range(30)]
    maps = build_heatmaps(states)
    assert np.count_nonzero(maps[1]) == 1


def test_event_counts_split_by_team():
    events = [
        Event(EventType.GOAL, 10.0, "home"),
        Event(EventType.SHOT, 9.0, "home"),
        Event(EventType.SHOT, 40.0, "away"),
    ]
    counts = count_events(events)
    assert counts["all"]["shot"] == 2
    assert counts["home"]["goal"] == 1
    assert counts["away"]["shot"] == 1
    assert "goal" not in counts["away"]


def test_timeline_drops_passes_but_keeps_notable_events():
    events = [
        Event(EventType.PASS, 1.0, "home"),
        Event(EventType.SHOT, 2.0, "home"),
        Event(EventType.PASS, 3.0, "home"),
        Event(EventType.GOAL, 4.0, "home"),
    ]
    kinds = [e.type for e in build_timeline(events)]
    assert kinds == [EventType.SHOT, EventType.GOAL]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@pytest.fixture
def summary():
    states = []
    for i in range(100):
        states.append(
            st(
                i * DT,
                players={1: (5.0 * i * DT, 34.0, "home"), 2: (60.0, 30.0, "away")},
                team="home" if i < 70 else "away",
                holder=1 if i < 70 else 2,
                score=(1, 0) if i > 50 else (0, 0),
            )
        )
    events = [
        Event(EventType.PASS, 0.5, "home", 1),
        Event(EventType.SHOT, 1.5, "home", 1),
        Event(EventType.GOAL, 2.0, "home", 1, meta={"score": (1, 0)}),
        Event(EventType.TURNOVER, 3.0, "away", 2),
    ]
    return build(states, events)


def test_summary_takes_the_final_score(summary):
    assert summary.score == (1, 0)


def test_summary_populates_every_section(summary):
    assert summary.possession_pct
    assert summary.players
    assert summary.heatmaps
    assert summary.timeline
    assert summary.duration_s == pytest.approx(99 * DT, abs=1e-6)


def test_touches_and_passes_come_from_events_not_frames(summary):
    """A touch is a confirmed possession, not every frame near the ball."""
    assert summary.players[1].touches == 3
    assert summary.players[1].passes == 1
    assert summary.players[2].touches == 1


def test_top_by_orders_correctly(summary):
    movers = summary.top_by("distance_m", 2)
    assert movers[0].track_id == 1
    assert movers[0].distance_m >= movers[1].distance_m


def test_report_renders_the_key_figures(summary):
    text = format_report(summary)
    assert "Final score: 1-0" in text
    assert "Possession:" in text
    assert "Timeline:" in text
    assert "goal" in text


def test_report_flags_unreliable_tracking():
    states = walk(1, "home", (0.0, 34.0), speed_mps=5.0, n=10)
    states.append(st(10 * DT, players={1: (95.0, 34.0, "home")}))
    text = format_report(build(states, []))
    assert "rejected jumps" in text


def test_build_on_empty_input():
    s = build([], [])
    assert s.score is None
    assert s.duration_s == 0.0
    assert format_report(s)          # must not raise
