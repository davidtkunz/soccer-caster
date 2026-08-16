"""Per-frame game state.

Possession uses hysteresis so it does not flicker between two players standing
close together — a raw nearest-player assignment produces dozens of phantom
possession changes per minute.

Score and clock come from OCR of the broadcast scoreboard overlay rather than
from inference. It is the cheapest accuracy in the whole project: near-perfect
ground truth, no model to train, and it anchors goal detection.
"""

from dataclasses import dataclass, field

POSSESSION_RADIUS_M = 2.0     # ball within this distance = player has it
POSSESSION_HOLD_FRAMES = 5    # frames a new claim must hold before it counts


@dataclass
class GameState:
    t: float
    frame_idx: int
    ball_xy: tuple[float, float] | None       # pitch coordinates, metres
    players: dict = field(default_factory=dict)   # track_id -> (x, y, team)
    possessing_track_id: int | None = None
    possessing_team: str | None = None
    score: tuple[int, int] | None = None
    clock: str | None = None


def read_scoreboard(frame):
    """OCR the broadcast overlay for score and clock.

    Crop to the known overlay region once per broadcaster rather than scanning
    the whole frame — the bug sits in a fixed position for any given feed.
    Returns None when the overlay is absent, which is itself a useful signal:
    a missing scoreboard is decent evidence the current segment is a replay.
    """
    raise NotImplementedError


def resolve_possession(prev: GameState, ball_xy, players):
    """Nearest player within POSSESSION_RADIUS_M, with hold-frame hysteresis."""
    raise NotImplementedError


def build_state(tracked_frames):
    """Yield GameState per frame from tracked detections in pitch coordinates."""
    raise NotImplementedError
