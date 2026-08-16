"""The director — decides *when* the caster speaks.

Deliberately plain code, not a model call. Letting the LLM decide when to talk
produces relentless, unlistenable position-reading. The gate belongs here.

Intensity is also computed here, from the event's importance score, and handed
to the voice layer. It is not asked of the model: forcing structured output
would mean streaming JSON into a text-to-speech engine.
"""

from dataclasses import dataclass

SPEAK_THRESHOLD = 40          # below this, stay quiet
COLOUR_AFTER_SILENCE = 15.0   # seconds of quiet before filler kicks in
COOLDOWN = 2.0                # per-zone debounce, seconds


@dataclass
class Decision:
    speak: bool
    intensity: int = 0        # 0-3, maps onto TTS voice settings
    mode: str = "play_by_play"  # or "colour"
    interjection: str | None = None
    preempt: bool = False     # cut whatever is currently playing


def intensity_for(importance: int) -> int:
    if importance >= 100:
        return 3
    if importance >= 70:
        return 2
    if importance >= 40:
        return 1
    return 0


class Director:
    def __init__(self, voice_pack):
        self.voice_pack = voice_pack
        self.last_spoke_at = 0.0
        self.last_event_zone = {}

    def decide(self, event, now: float) -> Decision:
        """Gate a single event into a speak/stay-silent decision."""
        if event is None:
            if now - self.last_spoke_at > COLOUR_AFTER_SILENCE:
                return Decision(speak=True, intensity=0, mode="colour")
            return Decision(speak=False)

        if event.importance < SPEAK_THRESHOLD:
            return Decision(speak=False)

        level = intensity_for(event.importance)

        # A goal preempts anything in progress. The pre-rendered interjection
        # fires at zero latency and covers the generation delay behind it.
        preempt = event.importance >= 100
        clip = self.voice_pack.interjection_for(event.type) if preempt else None

        self.last_spoke_at = now
        return Decision(
            speak=True,
            intensity=level,
            mode="play_by_play",
            interjection=clip,
            preempt=preempt,
        )
