"""Shot-boundary and replay detection.

Runs before everything else. Broadcast feeds cut between the main tactical
view, close-ups, crowd shots, and replays. Feed all of that into the event
pipeline and a replayed goal becomes a second goal.
"""

from dataclasses import dataclass
from enum import Enum


class ShotClass(str, Enum):
    LIVE_WIDE = "live_wide"     # the main tactical view — the only one we use
    CLOSE_UP = "close_up"
    CROWD = "crowd"
    REPLAY = "replay"
    GRAPHIC = "graphic"


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    kind: ShotClass


def find_cuts(frames, threshold: float = 0.35):
    """Hard-cut detection by frame-to-frame colour histogram distance."""
    raise NotImplementedError


def classify(segment_frames) -> ShotClass:
    """Classify one segment.

    Two heuristics carry most of the weight:

    1. The main tactical view is the segment where *pitch keypoints are
       detectable* — wide green field, many small player boxes.
    2. The *scoreboard overlay usually disappears during replays*. This is a
       strong, cheap signal and worth checking before anything more elaborate.

    Replays additionally tend to follow a key event within ~30 s, run in slow
    motion, and open with a branded wipe.
    """
    raise NotImplementedError


def live_segments(frames):
    """Yield only the segments the rest of the pipeline is allowed to see."""
    for seg in segment(frames):
        if seg.kind is ShotClass.LIVE_WIDE:
            yield seg


def segment(frames):
    raise NotImplementedError
