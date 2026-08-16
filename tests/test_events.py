"""Tests for event detection.

Weighted toward the cases that produce *wrong* events rather than missing ones.
A missed event costs a quiet moment; a fabricated one is announced out loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from caster.events import (
    BallTrack,
    EventDetector,
    EventType,
    PitchModel,
    detect_events,
    trajectory_hits_goal,
)
from caster.state import GameState

FPS = 25.0
DT = 1.0 / FPS


def st(t, ball=None, holder=None, team=None, score=None, players=None):
    return GameState(
        t=t,
        frame_idx=int(t * FPS),
        ball_xy=ball,
        players=players or {},
        possessing_track_id=holder,
        possessing_team=team,
        score=score,
    )


def run(detector, states):
    return [e for s in states for e in detector.update(s)]


def types(events):
    return [e.type for e in events]


@pytest.fixture
def pitch():
    return PitchModel()          # home attacks x=105, away attacks x=0


@pytest.fixture
def det(pitch):
    return EventDetector(pitch)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_goal_mouth_is_centred_and_correct_width(pitch):
    lo, hi = pitch.goal_y
    assert hi - lo == pytest.approx(7.32)
    assert (lo + hi) / 2 == pytest.approx(pitch.width / 2)


def test_teams_defend_the_goal_they_do_not_attack(pitch):
    assert pitch.attack_x["home"] == 105.0 and pitch.defends_x("home") == 0.0
    assert pitch.attack_x["away"] == 0.0 and pitch.defends_x("away") == 105.0


@pytest.mark.parametrize(
    "team, xy, expected",
    [
        ("home", (90.0, 34.0), True),
        ("home", (50.0, 34.0), False),
        ("away", (10.0, 34.0), True),
        ("away", (50.0, 34.0), False),
    ],
)
def test_attacking_third(pitch, team, xy, expected):
    assert pitch.in_attacking_third(team, xy) is expected


@pytest.mark.parametrize(
    "defending_x, xy, expected",
    [
        (105.0, (95.0, 34.0), True),
        (105.0, (80.0, 34.0), False),      # outside the 16.5m box
        (105.0, (95.0, 5.0), False),       # too wide
        (0.0, (10.0, 34.0), True),
    ],
)
def test_penalty_area(pitch, defending_x, xy, expected):
    assert pitch.in_penalty_area(defending_x, xy) is expected


@pytest.mark.parametrize(
    "xy, out",
    [((50.0, 34.0), False), ((-1.0, 34.0), True), ((50.0, 70.0), True)],
)
def test_out_of_bounds(pitch, xy, out):
    assert pitch.out_of_bounds(xy) is out


# --------------------------------------------------------------------------
# Ball motion
# --------------------------------------------------------------------------


def test_velocity_recovers_constant_motion():
    bt = BallTrack()
    for i in range(7):
        bt.update(i * DT, (10.0 + 15.0 * i * DT, 34.0))
    vx, vy = bt.velocity()
    assert vx == pytest.approx(15.0, abs=0.2)
    assert vy == pytest.approx(0.0, abs=0.2)


def test_velocity_is_robust_to_detection_jitter():
    """A two-frame difference turns detection noise into phantom shots."""
    rng = np.random.default_rng(3)
    bt = BallTrack()
    for i in range(7):
        jitter = rng.normal(0, 0.15, 2)
        bt.update(i * DT, (10.0 + 15.0 * i * DT + jitter[0], 34.0 + jitter[1]))
    vx, _ = bt.velocity()
    assert vx == pytest.approx(15.0, abs=3.0)


def test_velocity_needs_history():
    bt = BallTrack()
    bt.update(0.0, (10.0, 34.0))
    assert bt.velocity() is None


@pytest.mark.parametrize(
    "pos, vel, hits",
    [
        ((85.0, 34.0), (15.0, 0.0), True),        # straight at goal
        ((85.0, 34.0), (15.0, 20.0), False),      # wide
        ((85.0, 34.0), (-15.0, 0.0), False),      # moving away
        ((85.0, 34.0), (0.0, 10.0), False),       # across, never arrives
    ],
)
def test_trajectory_hits_goal(pitch, pos, vel, hits):
    assert trajectory_hits_goal(pos, vel, 105.0, pitch.goal_y) is hits


# --------------------------------------------------------------------------
# Possession events
# --------------------------------------------------------------------------


def test_pass_is_detected_across_the_ball_being_loose(det):
    """The None gap is the normal case -- in flight, nobody possesses the ball."""
    states = (
        [st(i * DT, (30.0, 34.0), 1, "home") for i in range(10)]
        + [st((10 + i) * DT, (35.0, 34.0)) for i in range(10)]        # in flight
        + [st((20 + i) * DT, (40.0, 34.0), 2, "home") for i in range(5)]
    )
    events = [e for e in run(det, states) if e.type is EventType.PASS]
    assert len(events) == 1
    assert events[0].player_track_id == 1
    assert events[0].meta["to"] == 2


def test_long_loose_ball_is_not_a_pass(det):
    states = (
        [st(i * DT, (30.0, 34.0), 1, "home") for i in range(10)]
        + [st(1.0 + i * DT, (35.0, 34.0)) for i in range(200)]        # loose 8s
        + [st(10.0 + i * DT, (40.0, 34.0), 2, "home") for i in range(5)]
    )
    assert not [e for e in run(det, states) if e.type is EventType.PASS]


def test_change_of_team_is_a_turnover_not_a_pass(det):
    states = (
        [st(i * DT, (30.0, 34.0), 1, "home") for i in range(10)]
        + [st((10 + i) * DT, (40.0, 34.0), 2, "away") for i in range(5)]
    )
    events = run(det, states)
    assert types(events) == [EventType.TURNOVER]
    assert events[0].team == "away"
    assert events[0].meta["from_team"] == "home"


def test_holding_possession_emits_nothing(det):
    states = [st(i * DT, (30.0, 34.0), 1, "home") for i in range(50)]
    assert run(det, states) == []


# --------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------


def test_goal_comes_from_the_scoreboard(det):
    states = (
        [st(i * DT, (50.0, 34.0), 1, "home", score=(0, 0)) for i in range(5)]
        + [st((5 + i) * DT, (50.0, 34.0), 1, "home", score=(1, 0)) for i in range(5)]
    )
    goals = [e for e in run(det, states) if e.type is EventType.GOAL]
    assert len(goals) == 1
    assert goals[0].team == "home"
    assert goals[0].confidence == 1.0
    assert goals[0].meta["score"] == (1, 0)


def test_first_score_reading_is_a_baseline_not_a_goal(det):
    """Joining mid-match at 2-1 must not announce three goals."""
    states = [st(i * DT, (50.0, 34.0), 1, "home", score=(2, 1)) for i in range(10)]
    assert not [e for e in run(det, states) if e.type is EventType.GOAL]


def test_away_goal_is_attributed_to_away(det):
    states = (
        [st(i * DT, (50.0, 34.0), 1, "away", score=(1, 1)) for i in range(5)]
        + [st((5 + i) * DT, (50.0, 34.0), 1, "away", score=(1, 2)) for i in range(5)]
    )
    goals = [e for e in run(det, states) if e.type is EventType.GOAL]
    assert len(goals) == 1 and goals[0].team == "away"


# --------------------------------------------------------------------------
# Shots
# --------------------------------------------------------------------------


def _shot_states(vy=0.0, x0=82.0, speed=18.0, n=10, t0=0.0, holder=1, team="home"):
    return [
        st(t0 + i * DT, (x0 + speed * i * DT, 34.0 + vy * i * DT), holder, team)
        for i in range(n)
    ]


def test_shot_on_target_is_detected(det):
    events = [e for e in run(det, _shot_states()) if e.type is EventType.SHOT]
    assert len(events) == 1
    assert events[0].team == "home"


def test_one_strike_emits_one_shot(det):
    """Without a cooldown the ball emits a shot on every frame of its flight."""
    shots = [e for e in run(det, _shot_states(n=40)) if e.type is EventType.SHOT]
    assert len(shots) == 1


def test_ball_travelling_wide_is_not_a_shot(det):
    assert not [e for e in run(det, _shot_states(vy=25.0)) if e.type is EventType.SHOT]


def test_fast_ball_outside_the_attacking_third_is_not_a_shot(det):
    assert not [
        e for e in run(det, _shot_states(x0=20.0)) if e.type is EventType.SHOT
    ]


def test_slow_ball_towards_goal_is_not_a_shot(det):
    assert not [e for e in run(det, _shot_states(speed=3.0)) if e.type is EventType.SHOT]


def test_away_shoots_at_the_other_goal(det):
    states = [
        st(i * DT, (23.0 - 18.0 * i * DT, 34.0), 5, "away") for i in range(10)
    ]
    shots = [e for e in run(det, states) if e.type is EventType.SHOT]
    assert len(shots) == 1 and shots[0].team == "away"


# --------------------------------------------------------------------------
# Saves
# --------------------------------------------------------------------------


def test_save_is_detected_in_the_defended_penalty_area(det):
    """Regression: the relevant box is around the goal being shot at.

    Using the shooter's own defensive end put the check 105 metres away, so no
    save could ever be detected.
    """
    states = _shot_states()
    keeper_at = {9: (98.0, 34.0, "away")}
    states += [
        st(0.5 + i * DT, (98.0, 34.0), 9, "away", players=keeper_at)
        for i in range(5)
    ]
    events = run(det, states)
    saves = [e for e in events if e.type is EventType.SAVE]
    assert len(saves) == 1
    assert saves[0].team == "away" and saves[0].player_track_id == 9


def test_a_shot_that_scores_is_not_also_a_save(det):
    states = _shot_states()
    states = [st(s.t, s.ball_xy, s.possessing_track_id, s.possessing_team,
                 score=(0, 0)) for s in states]
    states += [
        st(0.5 + i * DT, (104.0, 34.0), None, None, score=(1, 0)) for i in range(5)
    ]
    kinds = types(run(det, states))
    assert EventType.GOAL in kinds
    assert EventType.SAVE not in kinds


def test_possession_outside_the_box_is_not_a_save(det):
    states = _shot_states()
    far = {9: (60.0, 34.0, "away")}
    states += [
        st(0.5 + i * DT, (60.0, 34.0), 9, "away", players=far) for i in range(5)
    ]
    assert EventType.SAVE not in types(run(det, states))


# --------------------------------------------------------------------------
# Restarts
# --------------------------------------------------------------------------


def test_ball_over_the_touchline_is_a_throw_in(det):
    states = [st(i * DT, (50.0, 34.0), 1, "home") for i in range(5)]
    states += [st((5 + i) * DT, (50.0, 70.0), None, None) for i in range(5)]
    restarts = [e for e in run(det, states) if e.type is EventType.RESTART]
    assert len(restarts) == 1
    assert restarts[0].meta["kind"] == "throw_in"


def test_attacker_putting_it_wide_gives_a_goal_kick(det):
    """Regression: this mapping was inverted.

    Home attacks x=105. Home puts it behind that line, so the defenders restart
    with a goal kick -- not a corner.
    """
    states = [st(i * DT, (100.0, 34.0), 1, "home") for i in range(5)]
    states += [st((5 + i) * DT, (106.0, 20.0), None, None) for i in range(5)]
    restarts = [e for e in run(det, states) if e.type is EventType.RESTART]
    assert len(restarts) == 1
    assert restarts[0].meta["kind"] == "goal_kick"


def test_defender_putting_it_behind_gives_a_corner(det):
    """Away defends x=105. Away puts it behind their own line -> corner."""
    states = [st(i * DT, (100.0, 34.0), 7, "away") for i in range(5)]
    states += [st((5 + i) * DT, (106.0, 20.0), None, None) for i in range(5)]
    restarts = [e for e in run(det, states) if e.type is EventType.RESTART]
    assert len(restarts) == 1
    assert restarts[0].meta["kind"] == "corner"


def test_ball_crossing_inside_the_mouth_is_left_to_the_scoreboard(det):
    """Geometry does not call goals -- the scoreboard does."""
    states = [st(i * DT, (100.0, 34.0), 1, "home") for i in range(5)]
    states += [st((5 + i) * DT, (106.0, 34.0), None, None) for i in range(5)]
    assert EventType.RESTART not in types(run(det, states))


def test_leaving_the_pitch_reports_once(det):
    states = [st(i * DT, (50.0, 34.0), 1, "home") for i in range(5)]
    states += [st((5 + i) * DT, (50.0, 72.0), None, None) for i in range(60)]
    restarts = [e for e in run(det, states) if e.type is EventType.RESTART]
    assert len(restarts) == 1


def test_returning_to_play_rearms_the_detector(det):
    states = [st(i * DT, (50.0, 34.0), 1, "home") for i in range(5)]
    states += [st((5 + i) * DT, (50.0, 72.0), None, None) for i in range(5)]
    states += [st((10 + i) * DT, (50.0, 34.0), 1, "home") for i in range(5)]
    states += [st((15 + i) * DT, (50.0, 72.0), None, None) for i in range(5)]
    restarts = [e for e in run(det, states) if e.type is EventType.RESTART]
    assert len(restarts) == 2


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_importance_ordering_matches_the_directors_gate():
    from caster.events import IMPORTANCE

    assert IMPORTANCE[EventType.GOAL] > IMPORTANCE[EventType.SHOT]
    assert IMPORTANCE[EventType.SHOT] > IMPORTANCE[EventType.TURNOVER]
    assert IMPORTANCE[EventType.TURNOVER] > IMPORTANCE[EventType.PASS]


def test_detect_events_streams_over_states(det):
    states = (
        [st(i * DT, (30.0, 34.0), 1, "home") for i in range(10)]
        + [st((10 + i) * DT, (40.0, 34.0), 2, "away") for i in range(5)]
    )
    assert types(list(detect_events(states, det))) == [EventType.TURNOVER]


def test_empty_stream():
    assert list(detect_events([])) == []
