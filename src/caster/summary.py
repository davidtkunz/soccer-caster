"""Phase 1 deliverable — the post-game summary.

Same backbone as the live caster with the latency constraint removed. Build and
ship this first: it validates perception, state, and event detection against
recorded matches where accuracy can actually be measured, and its narrative
recap is the first draft of the live narration prompt.
"""

from dataclasses import dataclass, field


@dataclass
class MatchSummary:
    score: tuple[int, int]
    possession_pct: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    per_player: dict = field(default_factory=dict)   # distance, sprints, heatmap
    clips: dict = field(default_factory=dict)        # event id -> clip path


def build(events, states) -> MatchSummary:
    """Roll the event and state streams into match-level output."""
    raise NotImplementedError


def render_heatmaps(states, out_dir):
    """Per-player position heatmaps in pitch coordinates."""
    raise NotImplementedError


def export_clips(video_path, events, out_dir, pad: float = 4.0):
    """Cut a clip around each event for the timeline."""
    raise NotImplementedError


def narrative_recap(summary: MatchSummary) -> str:
    """Prose recap generated from the structured summary.

    Deliberately built on the derived stats rather than on raw footage: the
    perception layer produces facts, and the language layer only writes them up.
    Nothing here should assert something the event stream cannot support.
    """
    raise NotImplementedError
