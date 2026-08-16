"""Voice output — TTS streaming, interjections, and barge-in.

Two things matter here.

Pre-rendered interjections. The signature exclamations are a fixed vocabulary
that needs zero latency, so they ship as audio files rather than being
generated. The director fires one the instant a goal event lands, and the
generated sentence streams in behind it — the interjection is what hides the
generation delay from the listener.

Barge-in. A goal must be able to cut a filler line mid-word. That means a hard
stop that kills the current TTS stream *and* the pending generation. Build it
now; retrofitting it into a naive playback queue is genuinely painful.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class VoicePack:
    name: str
    persona_prompt: str
    signature_calls: list = field(default_factory=list)
    voice_id: str = ""
    settings_by_intensity: dict = field(default_factory=dict)
    interjections: dict = field(default_factory=dict)
    root: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path) -> "VoicePack":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        persona = data["persona"]
        voice = data["voice"]
        return cls(
            name=persona["name"],
            persona_prompt=persona["system_prompt"],
            signature_calls=persona.get("signature_calls", []),
            voice_id=voice["voice_id"],
            settings_by_intensity=voice.get("settings_by_intensity", {}),
            interjections=data.get("interjections", {}),
            root=path.parent,
        )

    def interjection_for(self, event_type) -> Path | None:
        """Pick a random clip for this event so the same shout doesn't repeat."""
        clips = self.interjections.get(str(event_type))
        if not clips:
            return None
        return self.root / "audio" / random.choice(clips)

    def settings(self, intensity: int) -> dict:
        return self.settings_by_intensity.get(intensity, {})


class VoiceOut:
    """Streaming TTS with preemption."""

    def __init__(self, pack: VoicePack, mixer):
        self.pack = pack
        self.mixer = mixer
        self._active = None

    def play_interjection(self, clip: Path) -> None:
        """Fire immediately, ahead of any pending generation."""
        raise NotImplementedError

    def open(self, intensity: int):
        """Open a streaming TTS session at the given intensity."""
        raise NotImplementedError

    def feed(self, text_chunk: str) -> None:
        """Push a chunk of generated text into the open TTS stream."""
        raise NotImplementedError

    def stop(self) -> None:
        """Hard stop — kills playback and the in-flight generation."""
        raise NotImplementedError
