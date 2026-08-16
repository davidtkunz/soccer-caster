"""Tests for the delay buffer and the narration director (Phase 2).

The delay buffer is the design's central idea: narration reads from a cursor N
seconds behind the live edge, so a play's outcome is already known before it is
described. The property worth pinning is that the lookahead genuinely sees the
future relative to the narration cursor.
"""

from __future__ import annotations

import pytest

from caster.buffer import DelayBuffer, Tick
from caster.director import (
    COLOUR_AFTER_SILENCE,
    SPEAK_THRESHOLD,
    Director,
    intensity_for,
)
from caster.events import Event, EventType

FPS = 25.0
DT = 1.0 / FPS


def tick(t, events=None):
    return Tick(t=t, frame_idx=int(t * FPS), state=None, events=events or [])


def fill(buf, seconds, events_at=None):
    """Push ticks covering `seconds` of footage; events_at maps t -> [Event]."""
    events_at = events_at or {}
    n = int(seconds * FPS)
    for i in range(n):
        t = round(i * DT, 6)
        buf.push(tick(t, events_at.get(t, [])))
    return buf


# --------------------------------------------------------------------------
# Delay buffer
# --------------------------------------------------------------------------


def test_buffer_is_not_ready_until_the_delay_has_elapsed():
    buf = DelayBuffer(delay=8.0, fps=FPS)
    fill(buf, 3.0)
    assert not buf.ready()
    fill(buf, 0.0)
    assert buf.live_t < 8.0


def test_cursor_trails_the_live_edge_by_the_delay():
    buf = DelayBuffer(delay=8.0, fps=FPS)
    fill(buf, 12.0)
    assert buf.ready()
    assert buf.live_t - buf.cursor_t == pytest.approx(8.0)


def test_cursor_never_goes_negative():
    buf = DelayBuffer(delay=8.0, fps=FPS)
    fill(buf, 2.0)
    assert buf.cursor_t == 0.0


def test_pop_due_yields_only_aged_ticks():
    buf = DelayBuffer(delay=4.0, fps=FPS)
    fill(buf, 10.0)
    due = list(buf.pop_due())
    assert due, "expected aged ticks"
    assert max(t.t for t in due) <= buf.cursor_t + 1e-9


def test_popped_ticks_are_not_yielded_twice():
    buf = DelayBuffer(delay=4.0, fps=FPS)
    fill(buf, 10.0)
    first = list(buf.pop_due())
    second = list(buf.pop_due())
    assert first and not second


def test_lookahead_sees_the_future_relative_to_the_cursor():
    """The whole point: at narration time, the outcome is already in the buffer."""
    goal = Event(EventType.GOAL, 6.0, "home")
    buf = DelayBuffer(delay=8.0, fps=FPS)
    fill(buf, 12.0, events_at={6.0: [goal]})

    # The cursor trails the live edge by exactly the delay. The goal at t=6.0
    # is ahead of it -- not yet spoken, but already known.
    assert buf.cursor_t == pytest.approx(buf.live_t - 8.0)
    assert buf.cursor_t < goal.t < buf.live_t
    assert goal in buf.lookahead(buf.cursor_t, window=4.0)


def test_lookahead_excludes_events_outside_the_window():
    goal = Event(EventType.GOAL, 11.0, "home")
    buf = DelayBuffer(delay=8.0, fps=FPS)
    fill(buf, 12.0, events_at={11.0: [goal]})
    assert goal not in buf.lookahead(buf.cursor_t, window=2.0)


def test_lookahead_excludes_the_past():
    old = Event(EventType.SHOT, 1.0, "home")
    buf = DelayBuffer(delay=8.0, fps=FPS)
    fill(buf, 12.0, events_at={1.0: [old]})
    assert old not in buf.lookahead(buf.cursor_t, window=4.0)


def test_empty_buffer_is_safe():
    buf = DelayBuffer(delay=8.0, fps=FPS)
    assert buf.live_t == 0.0
    assert buf.cursor_t == 0.0
    assert not buf.ready()
    assert list(buf.pop_due()) == []
    assert buf.lookahead(0.0) == []


# --------------------------------------------------------------------------
# Director
# --------------------------------------------------------------------------


class FakeVoicePack:
    def __init__(self):
        self.asked = []

    def interjection_for(self, event_type):
        self.asked.append(event_type)
        return f"{event_type.value}.mp3"


@pytest.fixture
def pack():
    return FakeVoicePack()


@pytest.fixture
def director(pack):
    return Director(pack)


@pytest.mark.parametrize(
    "importance, expected",
    [(100, 3), (80, 2), (60, 1), (40, 1), (5, 0)],
)
def test_intensity_ladder(importance, expected):
    assert intensity_for(importance) == expected


def test_low_importance_events_stay_silent(director):
    decision = director.decide(Event(EventType.PASS, 10.0, "home"), now=10.0)
    assert decision.speak is False


def test_goal_speaks_at_full_intensity_and_preempts(director, pack):
    decision = director.decide(Event(EventType.GOAL, 10.0, "home"), now=10.0)
    assert decision.speak is True
    assert decision.intensity == 3
    assert decision.preempt is True
    assert decision.interjection == "goal.mp3"
    assert pack.asked == [EventType.GOAL]


def test_shot_speaks_without_preempting(director):
    decision = director.decide(Event(EventType.SHOT, 10.0, "home"), now=10.0)
    assert decision.speak is True
    assert decision.preempt is False
    assert decision.interjection is None


def test_silence_triggers_colour_commentary(director):
    director.decide(Event(EventType.GOAL, 0.0, "home"), now=0.0)
    quiet = director.decide(None, now=COLOUR_AFTER_SILENCE + 1.0)
    assert quiet.speak is True
    assert quiet.mode == "colour"
    assert quiet.intensity == 0


def test_brief_quiet_does_not_trigger_colour(director):
    director.decide(Event(EventType.GOAL, 0.0, "home"), now=0.0)
    assert director.decide(None, now=2.0).speak is False


def test_speaking_resets_the_silence_timer(director):
    director.decide(Event(EventType.GOAL, 0.0, "home"), now=0.0)
    director.decide(Event(EventType.SHOT, 10.0, "home"), now=10.0)
    # Silence is measured from the last utterance, not the last goal.
    assert director.decide(None, now=10.0 + COLOUR_AFTER_SILENCE - 1).speak is False


def test_threshold_is_the_documented_gate(director):
    """Events below SPEAK_THRESHOLD are silent; at or above it, spoken."""
    below = Event(EventType.PASS, 1.0, "home")
    above = Event(EventType.TURNOVER, 1.0, "home")
    assert below.importance < SPEAK_THRESHOLD <= above.importance
    assert director.decide(below, now=1.0).speak is False
    assert director.decide(above, now=1.0).speak is True
