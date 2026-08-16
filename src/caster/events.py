"""Event detection — rules over game-state transitions.

This is the core of the project. Perception gives positions; this module turns
positions into things worth talking about.

Design rule: precision over recall. A missed event costs a quiet moment. A
fabricated event costs the whole system's credibility.
"""

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    PASS = "pass"
    TURNOVER = "turnover"
    SHOT = "shot"
    GOAL = "goal"
    SAVE = "save"
    RESTART = "restart"
    FOUL = "foul"


#: Importance drives the director's speak/stay-silent gate and the TTS intensity
#: level. Tuned by listening, not by theory.
IMPORTANCE = {
    EventType.GOAL: 100,
    EventType.SHOT: 80,
    EventType.SAVE: 80,
    EventType.TURNOVER: 60,
    EventType.RESTART: 30,
    EventType.PASS: 5,
    EventType.FOUL: 20,
}


@dataclass
class Event:
    type: EventType
    t: float                      # seconds from kickoff
    team: str | None = None
    player_track_id: int | None = None
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def importance(self) -> int:
        return IMPORTANCE.get(self.type, 0)


def detect_events(state_frames):
    """Yield Events from a sequence of per-frame GameState records.

    Args:
        state_frames: ordered iterable of state.GameState

    Detection notes per event type:

    pass / turnover
        Possession is the nearest player to the ball within a distance
        threshold, smoothed with hysteresis so it does not flicker frame to
        frame. A change of holder within a team is a pass; across teams, a
        turnover.

    shot
        Ball velocity vector directed at the goal mouth, speed above threshold,
        originating in the attacking third. Noisy — this is the row most worth
        tuning against the eval set.

    goal
        Primary signal is the *scoreboard delta*, not the vision pipeline.
        Goal-line crossing in pitch coordinates only refines the timestamp.
        This ordering matters: the scoreboard is near-perfect ground truth and
        costs almost nothing.

    save
        Shot + goalkeeper contact + no scoreboard change.

    restart
        Ball leaves pitch bounds; the re-entry location classifies it as a
        corner, throw-in, or goal kick.

    foul
        Weak. Proxy only: play stoppage inferred from a static ball and
        clustered players. Do not build commentary that depends on it.
    """
    raise NotImplementedError
