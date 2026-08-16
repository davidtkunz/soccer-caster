"""Event detection -- rules over game-state transitions.

This is the core of the project. Perception gives positions; this module turns
positions into things worth talking about.

Design rule throughout: **precision over recall**. A missed event costs a quiet
moment. A fabricated event costs the system's credibility, and in the live
caster it costs it out loud. Where a rule is uncertain, it emits nothing rather
than guessing, and every event carries a confidence the narration layer can
gate on.

Two things are worth knowing before reading the code.

**A pass goes through nobody.** While the ball is in flight no player possesses
it, so possession reads A -> None -> B. The None is the normal case, not the
exception, and a detector that only looks for adjacent holder changes finds
almost no passes. What bounds it is time: if the ball is loose long enough, it
was a clearance or a scramble rather than a pass.

**The scoreboard outranks the vision pipeline.** Goals come from a score delta,
not from watching the ball cross the line. Line-crossing is used only to refine
the timestamp. The scoreboard is near-perfect ground truth and the geometry is
not, so where they disagree the scoreboard wins.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


class EventType(str, Enum):
    PASS = "pass"
    TURNOVER = "turnover"
    SHOT = "shot"
    GOAL = "goal"
    SAVE = "save"
    RESTART = "restart"
    FOUL = "foul"


#: Drives the director's speak/stay-silent gate and the TTS intensity level.
#: Tuned by listening, not by theory.
IMPORTANCE = {
    EventType.GOAL: 100,
    EventType.SHOT: 80,
    EventType.SAVE: 80,
    EventType.TURNOVER: 60,
    EventType.RESTART: 30,
    EventType.FOUL: 20,
    EventType.PASS: 5,
}


@dataclass
class Event:
    type: EventType
    t: float
    team: str | None = None
    player_track_id: int | None = None
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def importance(self) -> int:
        return IMPORTANCE.get(self.type, 0)

    def __repr__(self) -> str:
        who = f" p{self.player_track_id}" if self.player_track_id is not None else ""
        return f"<{self.type.value} t={self.t:.1f} {self.team or '-'}{who}>"


# --------------------------------------------------------------------------
# Pitch geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PitchModel:
    """Canonical pitch in metres, and which way each team attacks.

    ``attack_x`` maps a team name to the goal line it attacks: 0.0 or
    ``length``. Sides swap at half-time, so build a second model for the second
    half rather than mutating this one -- an event stream that silently changes
    its own coordinate convention halfway through is very hard to debug.
    """

    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32
    penalty_depth: float = 16.5
    penalty_width: float = 40.32
    attack_x: dict = field(default_factory=lambda: {"home": 105.0, "away": 0.0})

    @property
    def goal_y(self) -> tuple[float, float]:
        c = self.width / 2
        return (c - self.goal_width / 2, c + self.goal_width / 2)

    @property
    def penalty_y(self) -> tuple[float, float]:
        c = self.width / 2
        return (c - self.penalty_width / 2, c + self.penalty_width / 2)

    def defends_x(self, team: str) -> float:
        atk = self.attack_x[team]
        return 0.0 if atk == self.length else self.length

    def in_attacking_third(self, team: str, xy) -> bool:
        atk = self.attack_x[team]
        third = self.length / 3
        return xy[0] >= self.length - third if atk == self.length else xy[0] <= third

    def in_penalty_area(self, defending_x: float, xy) -> bool:
        lo, hi = self.penalty_y
        if not (lo <= xy[1] <= hi):
            return False
        return (
            xy[0] <= self.penalty_depth
            if defending_x == 0.0
            else xy[0] >= self.length - self.penalty_depth
        )

    def out_of_bounds(self, xy) -> bool:
        return not (0.0 <= xy[0] <= self.length and 0.0 <= xy[1] <= self.width)


# --------------------------------------------------------------------------
# Ball motion
# --------------------------------------------------------------------------


class BallTrack:
    """Smoothed ball velocity from a short history of positions.

    Ball detection is jittery, and a two-frame difference amplifies that jitter
    into velocities that swing wildly frame to frame. A least-squares fit over a
    short window is far steadier for the same latency, and shot detection keys
    directly off velocity direction -- so noise here becomes phantom shots.
    """

    def __init__(self, window: int = 7):
        self.window = window
        self._hist: deque[tuple[float, float, float]] = deque(maxlen=window)

    def update(self, t: float, xy) -> None:
        if xy is not None:
            self._hist.append((t, xy[0], xy[1]))

    def clear(self) -> None:
        self._hist.clear()

    @property
    def position(self):
        return (self._hist[-1][1], self._hist[-1][2]) if self._hist else None

    def velocity(self):
        """Metres per second as (vx, vy), or None without enough history."""
        if len(self._hist) < 3:
            return None
        arr = np.asarray(self._hist, dtype=np.float64)
        t = arr[:, 0] - arr[0, 0]
        if t[-1] - t[0] <= 0:
            return None
        vx = float(np.polyfit(t, arr[:, 1], 1)[0])
        vy = float(np.polyfit(t, arr[:, 2], 1)[0])
        return (vx, vy)

    def speed(self) -> float:
        v = self.velocity()
        return float(np.hypot(*v)) if v else 0.0


def trajectory_hits_goal(pos, vel, goal_x: float, goal_y: tuple[float, float]) -> bool:
    """Would the ball, continuing on this heading, cross inside the goal mouth?

    A straight-line projection. It ignores lift, spin, and deflection, so it is
    a test of *intent and direction* rather than a prediction of the outcome --
    which is what "was that a shot?" actually asks.
    """
    if vel is None:
        return False
    vx, vy = vel
    dx = goal_x - pos[0]
    if vx == 0 or np.sign(dx) != np.sign(vx):
        return False
    t = dx / vx
    if t <= 0:
        return False
    y = pos[1] + vy * t
    return goal_y[0] <= y <= goal_y[1]


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


@dataclass
class _PendingShot:
    event: Event
    deadline_t: float
    defending_x: float


class EventDetector:
    """Consumes a GameState stream and emits Events.

    Args:
        pitch: geometry and attacking directions.
        max_flight_s: longest ball-in-flight gap still treated as a pass. Beyond
            this the ball was loose, not played to someone.
        shot_speed_mps: minimum ball speed to qualify as a shot.
        shot_cooldown_s: suppression window so one strike emits one event
            rather than one per frame while the ball is still travelling.
        shot_window_s: how long to wait for a shot to resolve into goal or save.
    """

    def __init__(
        self,
        pitch: PitchModel | None = None,
        max_flight_s: float = 6.0,
        shot_speed_mps: float = 9.0,
        shot_cooldown_s: float = 2.0,
        shot_window_s: float = 4.0,
    ):
        self.pitch = pitch or PitchModel()
        self.max_flight_s = max_flight_s
        self.shot_speed_mps = shot_speed_mps
        self.shot_cooldown_s = shot_cooldown_s
        self.shot_window_s = shot_window_s

        self.ball = BallTrack()
        self._last_holder: int | None = None
        self._last_team: str | None = None
        self._last_holder_t: float | None = None
        self._last_score: tuple[int, int] | None = None
        self._last_shot_t: float = -1e9
        self._pending: _PendingShot | None = None
        self._was_out: bool = False

    # -- sub-detectors -----------------------------------------------------

    def _possession_events(self, s) -> list[Event]:
        holder, team = s.possessing_track_id, s.possessing_team
        if holder is None:
            return []

        if self._last_holder is None:
            self._last_holder, self._last_team = holder, team
            self._last_holder_t = s.t
            return []

        if holder == self._last_holder:
            self._last_holder_t = s.t
            return []

        gap = s.t - (self._last_holder_t or s.t)
        prev_team, prev_holder = self._last_team, self._last_holder
        self._last_holder, self._last_team = holder, team
        self._last_holder_t = s.t

        # Ball loose too long to call it a pass -- a clearance or a scramble.
        if gap > self.max_flight_s:
            return []

        if team == prev_team:
            return [
                Event(
                    EventType.PASS, s.t, team, prev_holder,
                    confidence=0.9,
                    meta={"to": holder, "flight_s": round(gap, 2)},
                )
            ]
        return [
            Event(
                EventType.TURNOVER, s.t, team, holder,
                confidence=0.9,
                meta={"from_team": prev_team, "from": prev_holder},
            )
        ]

    def _goal_events(self, s) -> list[Event]:
        if s.score is None:
            return []
        if self._last_score is None:
            self._last_score = s.score
            return []
        if s.score == self._last_score:
            return []

        prev, self._last_score = self._last_score, s.score
        # Confirmed by the scoreboard, so this is as certain as anything gets.
        team = "home" if s.score[0] > prev[0] else "away"
        return [
            Event(
                EventType.GOAL, s.t, team,
                confidence=1.0,
                meta={"score": s.score, "previous": prev},
            )
        ]

    def _shot_events(self, s) -> list[Event]:
        if s.t - self._last_shot_t < self.shot_cooldown_s:
            return []

        pos, vel = self.ball.position, self.ball.velocity()
        if pos is None or vel is None or self.ball.speed() < self.shot_speed_mps:
            return []

        # Attribute to whoever last had it -- at the moment of striking, the
        # ball has already left their feet and possession has gone to None.
        team = s.possessing_team or self._last_team
        if team is None or team not in self.pitch.attack_x:
            return []
        if not self.pitch.in_attacking_third(team, pos):
            return []

        goal_x = self.pitch.attack_x[team]
        if not trajectory_hits_goal(pos, vel, goal_x, self.pitch.goal_y):
            return []

        self._last_shot_t = s.t
        event = Event(
            EventType.SHOT, s.t, team,
            s.possessing_track_id or self._last_holder,
            confidence=0.65,
            meta={"speed_mps": round(self.ball.speed(), 1)},
        )
        # The penalty area a save would happen in is the one around the goal
        # being *shot at* -- attack_x[team] -- not the shooter's own end.
        self._pending = _PendingShot(
            event, s.t + self.shot_window_s, self.pitch.attack_x[team]
        )
        return [event]

    def _resolve_shot(self, s, goals: list[Event]) -> list[Event]:
        """Turn a pending shot into a save, or let it lapse."""
        if self._pending is None:
            return []

        if goals:
            self._pending = None          # the shot scored; GOAL already emitted
            return []

        holder, team = s.possessing_track_id, s.possessing_team
        if holder is not None and team != self._pending.event.team:
            pos = s.players.get(holder)
            if pos and self.pitch.in_penalty_area(self._pending.defending_x, pos[:2]):
                shot = self._pending
                self._pending = None
                return [
                    Event(
                        EventType.SAVE, s.t, team, holder,
                        confidence=0.5,
                        meta={"shot_t": shot.event.t},
                    )
                ]

        if s.t >= self._pending.deadline_t:
            self._pending = None
        return []

    def _restart_events(self, s) -> list[Event]:
        if s.ball_xy is None:
            return []

        out = self.pitch.out_of_bounds(s.ball_xy)
        if not out:
            self._was_out = False
            return []
        if self._was_out:
            return []                      # already reported this exit
        self._was_out = True

        x, y = s.ball_xy
        team = self._last_team

        if y < 0 or y > self.pitch.width:
            kind = "throw_in"
        else:
            # Left over a goal line. Who touched it last decides the restart.
            gy = self.pitch.goal_y
            if gy[0] <= y <= gy[1]:
                # Inside the mouth -- the scoreboard, not geometry, calls goals.
                return []
            # Which goal line did it cross, and was the last toucher defending
            # it? A defender putting it behind his own line concedes a corner;
            # an attacker putting it wide gives the defenders a goal kick.
            line_x = 0.0 if x < self.pitch.length / 2 else self.pitch.length
            kind = "corner"
            if team is not None and team in self.pitch.attack_x:
                own_line = self.pitch.defends_x(team) == line_x
                kind = "corner" if own_line else "goal_kick"

        return [
            Event(
                EventType.RESTART, s.t, team,
                confidence=0.8,
                meta={"kind": kind, "at": (round(x, 1), round(y, 1))},
            )
        ]

    # -- driver ------------------------------------------------------------

    def update(self, s) -> list[Event]:
        self.ball.update(s.t, s.ball_xy)

        events: list[Event] = []
        goals = self._goal_events(s)
        events += goals
        events += self._possession_events(s)
        events += self._shot_events(s)
        events += self._resolve_shot(s, goals)
        events += self._restart_events(s)
        return events


def detect_events(states, detector: EventDetector | None = None):
    """Yield Events from a sequence of GameState records.

    Only states from live-wide segments should reach here -- see
    :func:`caster.segmentation.live_segments`. A replay feeds the detector a
    discontinuous jump in ball and player positions, which reads as a turnover
    followed by a shot from nowhere.
    """
    detector = detector or EventDetector()
    for state in states:
        yield from detector.update(state)
