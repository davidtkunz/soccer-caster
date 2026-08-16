"""The delay buffer — the central design decision.

Perception runs at the live edge. Narration reads from a cursor N seconds
behind it. Everything between the cursor and the live edge is *known future*:
when the caster describes a shot, the resolution of that shot is already in the
buffer.

This converts a hard real-time problem into a soft one. A slow generation
degrades into a slightly late line rather than a dropped one — and, more
importantly, the system can build tension toward an outcome it has confirmed
rather than one it is guessing at.
"""

from collections import deque
from dataclasses import dataclass

DELAY_SECONDS = 8.0


@dataclass
class Tick:
    t: float
    frame_idx: int
    state: object            # state.GameState
    events: list             # events.Event produced at this timestamp


class DelayBuffer:
    def __init__(self, delay: float = DELAY_SECONDS, fps: float = 25.0):
        self.delay = delay
        self.fps = fps
        self._ticks: deque[Tick] = deque(maxlen=int(delay * fps * 2))

    def push(self, tick: Tick) -> None:
        """Called at the live edge, once per processed frame."""
        self._ticks.append(tick)

    @property
    def live_t(self) -> float:
        return self._ticks[-1].t if self._ticks else 0.0

    @property
    def cursor_t(self) -> float:
        return max(0.0, self.live_t - self.delay)

    def ready(self) -> bool:
        return bool(self._ticks) and self.live_t >= self.delay

    def pop_due(self):
        """Yield ticks that have aged past the delay and are due for narration."""
        cutoff = self.cursor_t
        while self._ticks and self._ticks[0].t <= cutoff:
            yield self._ticks.popleft()

    def lookahead(self, from_t: float, window: float = 4.0):
        """Events between from_t and from_t + window.

        This is what makes honest tension possible: the narrator can see how a
        play resolves before describing its build-up.
        """
        return [
            e
            for tick in self._ticks
            if from_t < tick.t <= from_t + window
            for e in tick.events
        ]
