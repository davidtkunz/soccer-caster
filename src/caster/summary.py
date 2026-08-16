"""Phase 1 deliverable -- the post-game summary.

Same backbone as the live caster with the latency constraint removed. Build and
ship this first: it validates perception, state, and event detection against
recorded matches where accuracy can actually be measured, and its narrative
recap is the first draft of the live narration prompt.

Everything here is derived from the event and state streams. Nothing in the
summary asserts something those streams cannot support -- the perception layer
produces facts and the language layer only writes them up. That separation is
what keeps the recap honest, and it is the same split the live caster uses.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .events import Event, EventType, PitchModel

log = logging.getLogger(__name__)

#: A player cannot move faster than this. Anything above it is a tracking-ID
#: switch teleporting one identity across the pitch, not a run -- Usain Bolt
#: peaks around 12 m/s and footballers do not reach that carrying a ball.
MAX_PLAUSIBLE_SPEED_MPS = 12.0

SPRINT_SPEED_MPS = 7.0
SPRINT_MIN_DURATION_S = 1.0


@dataclass
class PlayerStats:
    track_id: int
    team: str | None = None
    distance_m: float = 0.0
    top_speed_mps: float = 0.0
    sprints: int = 0
    frames: int = 0
    touches: int = 0
    passes: int = 0
    #: Rejected frame-to-frame jumps. A high count means tracking is switching
    #: identities, and every derived figure for this player is suspect.
    teleports: int = 0


@dataclass
class MatchSummary:
    score: tuple[int, int] | None = None
    possession_pct: dict[str, float] = field(default_factory=dict)
    event_counts: dict[str, Counter] = field(default_factory=dict)
    players: dict[int, PlayerStats] = field(default_factory=dict)
    timeline: list[Event] = field(default_factory=list)
    heatmaps: dict[int, np.ndarray] = field(default_factory=dict)
    duration_s: float = 0.0

    def top_by(self, attr: str, n: int = 5) -> list[PlayerStats]:
        return sorted(self.players.values(), key=lambda p: getattr(p, attr),
                      reverse=True)[:n]


# --------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------


def possession_share(states) -> dict[str, float]:
    """Fraction of *possessed* frames per team.

    Loose-ball frames are excluded rather than counted against anyone, so the
    figures sum to 1.0 and mean "share of the ball when someone had it" -- which
    is what a possession statistic is normally understood to report.
    """
    held = Counter(s.possessing_team for s in states if s.possessing_team)
    total = sum(held.values())
    if not total:
        return {}
    return {team: n / total for team, n in held.items()}


def accumulate_player_stats(states) -> dict[int, PlayerStats]:
    """Distance, top speed, and sprint counts per tracked player.

    Frame-to-frame displacement is rejected when it implies an impossible
    speed. Tracking swaps are common enough that without this guard, distance
    covered is dominated by teleports and reads as several kilometres of
    nonsense for players who barely moved.
    """
    stats: dict[int, PlayerStats] = {}
    last_pos: dict[int, tuple[float, float]] = {}
    last_t: dict[int, float] = {}
    sprint_since: dict[int, float | None] = defaultdict(lambda: None)
    #: A sprint is one continuous run above threshold, counted once when it
    #: passes the minimum duration -- not once per second sustained. A four
    #: second run is one sprint, and counting it as four would make the figure
    #: a restatement of high-speed distance rather than a count of efforts.
    sprint_counted: dict[int, bool] = defaultdict(bool)

    for s in states:
        for tid, entry in s.players.items():
            x, y, team = entry
            st = stats.setdefault(tid, PlayerStats(track_id=tid, team=team))
            st.frames += 1
            if team and not st.team:
                st.team = team

            if tid in last_pos:
                dt = s.t - last_t[tid]
                if dt > 0:
                    step = float(np.hypot(x - last_pos[tid][0], y - last_pos[tid][1]))
                    speed = step / dt
                    if speed > MAX_PLAUSIBLE_SPEED_MPS:
                        st.teleports += 1
                        sprint_since[tid] = None
                        sprint_counted[tid] = False
                    else:
                        st.distance_m += step
                        st.top_speed_mps = max(st.top_speed_mps, speed)
                        if speed >= SPRINT_SPEED_MPS:
                            if sprint_since[tid] is None:
                                sprint_since[tid] = s.t
                            elif (
                                not sprint_counted[tid]
                                and s.t - sprint_since[tid] >= SPRINT_MIN_DURATION_S
                            ):
                                st.sprints += 1
                                sprint_counted[tid] = True
                        else:
                            sprint_since[tid] = None
                            sprint_counted[tid] = False

            last_pos[tid], last_t[tid] = (x, y), s.t

    return stats


def build_heatmaps(states, pitch: PitchModel | None = None, bins=(24, 16)):
    """Per-player occupancy histograms in pitch coordinates.

    Pitch coordinates rather than image coordinates, so the result is
    independent of camera pan and directly comparable between players.
    """
    pitch = pitch or PitchModel()
    points: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for s in states:
        for tid, (x, y, _team) in s.players.items():
            points[tid].append((x, y))

    maps = {}
    for tid, pts in points.items():
        arr = np.asarray(pts)
        h, _, _ = np.histogram2d(
            arr[:, 0], arr[:, 1], bins=bins,
            range=[[0, pitch.length], [0, pitch.width]],
        )
        maps[tid] = h
    return maps


def count_events(events) -> dict[str, Counter]:
    """Event tallies per team, plus an ``all`` bucket."""
    out: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        out["all"][e.type.value] += 1
        if e.team:
            out[e.team][e.type.value] += 1
    return dict(out)


NOTABLE = {EventType.GOAL, EventType.SHOT, EventType.SAVE, EventType.TURNOVER}


def build_timeline(events, notable=None) -> list[Event]:
    """Events worth a line in the report, in order.

    Passes are excluded by default -- a match produces hundreds and they carry
    no narrative weight individually, though they matter in aggregate.
    """
    notable = notable or NOTABLE
    return [e for e in events if e.type in notable]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build(states, events, pitch: PitchModel | None = None) -> MatchSummary:
    """Roll the state and event streams into match-level output."""
    states = list(states)
    events = list(events)

    summary = MatchSummary(
        possession_pct=possession_share(states),
        event_counts=count_events(events),
        players=accumulate_player_stats(states),
        timeline=build_timeline(events),
        heatmaps=build_heatmaps(states, pitch),
        duration_s=(states[-1].t - states[0].t) if states else 0.0,
    )

    for s in reversed(states):
        if s.score is not None:
            summary.score = s.score
            break

    # Touches and passes come from the event stream rather than the state
    # stream: a touch is a possession the tracker confirmed, not every frame a
    # player happened to be near the ball.
    for e in events:
        if e.player_track_id in summary.players:
            p = summary.players[e.player_track_id]
            p.touches += 1
            if e.type is EventType.PASS:
                p.passes += 1

    return summary


def format_report(summary: MatchSummary) -> str:
    """Plain-text match report. The structured input to the narrative recap."""
    lines: list[str] = []
    if summary.score:
        lines.append(f"Final score: {summary.score[0]}-{summary.score[1]}")
    lines.append(f"Duration: {summary.duration_s / 60:.0f} min")

    if summary.possession_pct:
        share = "  ".join(
            f"{team} {pct:.0%}" for team, pct in sorted(summary.possession_pct.items())
        )
        lines.append(f"Possession: {share}")

    counts = summary.event_counts.get("all", Counter())
    if counts:
        tally = "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines.append(f"Events: {tally}")

    movers = summary.top_by("distance_m", 5)
    if movers:
        lines.append("")
        lines.append("Distance covered:")
        for p in movers:
            flag = f"  [!] {p.teleports} rejected jumps" if p.teleports else ""
            lines.append(
                f"  p{p.track_id} ({p.team or '?'}): {p.distance_m / 1000:.2f} km, "
                f"top {p.top_speed_mps:.1f} m/s, {p.sprints} sprints{flag}"
            )

    if summary.timeline:
        lines.append("")
        lines.append("Timeline:")
        for e in summary.timeline:
            mins, secs = divmod(int(e.t), 60)
            detail = e.meta.get("kind") or e.meta.get("score") or ""
            lines.append(
                f"  {mins:02d}:{secs:02d}  {e.type.value:<9} "
                f"{e.team or '-':<5} {detail}"
            )

    return "\n".join(lines)


RECAP_SYSTEM = """You write concise match reports from structured match data.

Report only what the data supports. Do not invent player names, tactical
intent, crowd reaction, or managerial decisions - none of that is in the data,
and inventing it is the failure mode that makes automated reports untrustworthy.

Where a figure is flagged as unreliable, either omit it or say plainly that it
is uncertain. Lead with the result and the shape of the match, then the
supporting detail."""


def narrative_recap(summary: MatchSummary, client=None, model="claude-opus-5") -> str:
    """Prose recap generated from the structured summary.

    Deliberately built on derived stats rather than raw footage: the perception
    layer produces facts and the language layer only writes them up. No latency
    pressure here, so this runs at high effort with thinking on -- the opposite
    end of the dial from the live caster's play-by-play path.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=RECAP_SYSTEM,
        messages=[{"role": "user", "content": format_report(summary)}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
