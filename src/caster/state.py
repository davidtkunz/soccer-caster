"""Per-frame game state: possession, score, and clock.

Three ideas run through this module.

**Hysteresis everywhere.** Raw per-frame readings are noisy in ways that matter.
Nearest-player possession flickers dozens of times a minute between two players
standing close together, and every flicker looks like a pass to the event layer.
A digit misread on one frame looks like a goal. Both are fixed the same way:
a new reading has to hold for several consecutive frames, and beat the incumbent
by a margin, before it is accepted.

**Constraints reject nonsense for free.** A score cannot go down and cannot jump
by two. A clock cannot run backwards. Applying those rules costs nothing and
throws out most misreads before they reach the event layer.

**Template matching beats general OCR here.** The score bug uses one font, at one
size, in a fixed position, for the whole match. General-purpose OCR is built for
arbitrary fonts, sizes, and rotations -- it is slower, heavier, and less accurate
on this constrained problem than matching normalised digit glyphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


@dataclass
class FrameObservation:
    """What perception hands to this module, per frame."""

    frame_idx: int
    t: float
    ball_xy: tuple[float, float] | None                   # pitch metres
    players: dict[int, tuple[float, float, str]] = field(default_factory=dict)
    frame: np.ndarray | None = None                       # for scoreboard reads


@dataclass
class GameState:
    t: float
    frame_idx: int
    ball_xy: tuple[float, float] | None
    players: dict[int, tuple[float, float, str]] = field(default_factory=dict)
    possessing_track_id: int | None = None
    possessing_team: str | None = None
    score: tuple[int, int] | None = None
    clock_s: int | None = None

    @property
    def clock(self) -> str | None:
        if self.clock_s is None:
            return None
        return f"{self.clock_s // 60:02d}:{self.clock_s % 60:02d}"


# --------------------------------------------------------------------------
# Temporal confirmation
# --------------------------------------------------------------------------


class StableValue:
    """Accepts a new value only after it holds for several consecutive frames.

    Used for both score and clock. A validator can additionally reject
    transitions that are impossible given the previous value, which is where
    most of the accuracy comes from: a misread that violates monotonicity is
    discarded before it ever competes for confirmation.

    Missing readings (``None``) are ignored rather than treated as change --
    an unreadable frame is absence of evidence, not evidence of a new value.
    """

    def __init__(
        self,
        confirm_frames: int = 5,
        validator: Callable[[object, object], bool] | None = None,
    ):
        self.confirm_frames = confirm_frames
        self.validator = validator
        self.value = None
        self._candidate = None
        self._count = 0
        self.rejected = 0        # transitions thrown out by the validator

    def update(self, reading):
        if reading is None:
            return self.value

        if reading == self.value:
            self._candidate, self._count = None, 0
            return self.value

        if self.validator is not None and self.value is not None:
            if not self.validator(self.value, reading):
                self.rejected += 1
                return self.value

        if reading == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = reading, 1

        if self._count >= self.confirm_frames:
            self.value = self._candidate
            self._candidate, self._count = None, 0

        return self.value


def score_transition_valid(old: tuple[int, int], new: tuple[int, int]) -> bool:
    """A score never decreases, and only one goal is scored at a time."""
    if new[0] < old[0] or new[1] < old[1]:
        return False
    return (new[0] - old[0]) + (new[1] - old[1]) <= 1


def clock_transition_valid(old: int, new: int, max_jump_s: int = 180) -> bool:
    """A clock runs forward, and not in large leaps.

    The cap catches digit misreads: 12:34 misread as 72:34 is a 60-minute jump
    that no real clock makes between frames. Broadcasters that reset to 00:00 at
    half-time need this relaxed -- most continue counting to 90:00 instead.
    """
    return old <= new <= old + max_jump_s


# --------------------------------------------------------------------------
# Possession
# --------------------------------------------------------------------------


class PossessionTracker:
    """Who has the ball, with hysteresis to stop it flickering.

    Naive nearest-player assignment changes holder dozens of times a minute when
    two players contest the ball, and the event layer reads each change as a
    pass or turnover. Two rules fix it: a challenger must be *clearly* closer
    than the incumbent, and must stay clearly closer for several frames.

    Args:
        radius_m: ball must be within this distance to be possessed at all.
        challenge_margin_m: how much closer a challenger must be than the
            current holder. Prevents ties from oscillating.
        hold_frames: consecutive frames a challenger must lead before it counts.
        loose_frames: frames with nobody in range before the ball is loose.
    """

    def __init__(
        self,
        radius_m: float = 2.0,
        challenge_margin_m: float = 0.5,
        hold_frames: int = 5,
        loose_frames: int = 8,
    ):
        self.radius_m = radius_m
        self.challenge_margin_m = challenge_margin_m
        self.hold_frames = hold_frames
        self.loose_frames = loose_frames

        self.holder: int | None = None
        self.team: str | None = None
        self._challenger: int | None = None
        self._challenge_count = 0
        self._loose_count = 0

    def _release(self):
        self.holder, self.team = None, None
        self._challenger, self._challenge_count = None, 0

    def update(self, ball_xy, players) -> tuple[int | None, str | None]:
        if ball_xy is None or not players:
            self._loose_count += 1
            if self._loose_count >= self.loose_frames:
                self._release()
            return self.holder, self.team

        bx, by = ball_xy
        distances = {
            tid: float(np.hypot(px - bx, py - by))
            for tid, (px, py, _team) in players.items()
        }
        in_range = {t: d for t, d in distances.items() if d <= self.radius_m}

        if not in_range:
            self._loose_count += 1
            if self._loose_count >= self.loose_frames:
                self._release()
            return self.holder, self.team

        self._loose_count = 0
        nearest = min(in_range, key=in_range.get)

        if nearest == self.holder:
            self._challenger, self._challenge_count = None, 0
            return self.holder, self.team

        # Someone other than the incumbent is nearest. Require a clear margin,
        # otherwise a dead heat oscillates every frame.
        if self.holder is not None:
            holder_dist = distances.get(self.holder, float("inf"))
            if in_range[nearest] > holder_dist - self.challenge_margin_m:
                self._challenger, self._challenge_count = None, 0
                return self.holder, self.team

        if nearest == self._challenger:
            self._challenge_count += 1
        else:
            self._challenger, self._challenge_count = nearest, 1

        if self._challenge_count >= self.hold_frames:
            self.holder = nearest
            self.team = players[nearest][2]
            self._challenger, self._challenge_count = None, 0

        return self.holder, self.team


# --------------------------------------------------------------------------
# Scoreboard reading
# --------------------------------------------------------------------------

DIGIT_SIZE = (16, 24)         # width, height that every glyph is normalised to


def synthesize_digit_templates(
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.6,
    thickness: int = 2,
) -> dict[int, np.ndarray]:
    """Render 0-9 as normalised binary glyphs.

    A reasonable cold start: most broadcast bugs use a plain sans face, so a
    rendered set matches well enough to bootstrap. Replace with
    :func:`fit_digit_templates` once you have labelled crops from the actual
    broadcaster -- accuracy is meaningfully better with the real glyphs.
    """
    templates: dict[int, np.ndarray] = {}
    for d in range(10):
        canvas = np.zeros((80, 60), np.uint8)
        cv2.putText(canvas, str(d), (8, 62), font, scale, 255, thickness, cv2.LINE_AA)
        ys, xs = np.nonzero(canvas)
        if ys.size == 0:
            continue
        crop = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        templates[d] = cv2.resize(crop, DIGIT_SIZE, interpolation=cv2.INTER_AREA)
    return templates


def fit_digit_templates(samples: Iterable[tuple[np.ndarray, int]]) -> dict[int, np.ndarray]:
    """Build templates by averaging labelled glyph crops from real footage."""
    buckets: dict[int, list[np.ndarray]] = {}
    for crop, label in samples:
        norm = cv2.resize(crop, DIGIT_SIZE, interpolation=cv2.INTER_AREA)
        buckets.setdefault(int(label), []).append(norm.astype(np.float32))
    return {d: np.mean(v, axis=0).astype(np.uint8) for d, v in buckets.items()}


def _binarise(crop: np.ndarray) -> np.ndarray:
    """Threshold to white-text-on-black, whichever polarity the bug uses."""
    grey = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def segment_digits(
    crop: np.ndarray,
    min_height_frac: float = 0.35,
    max_aspect: float = 1.2,
    min_aspect: float = 0.05,
    max_width_frac: float = 0.9,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Isolate digit glyphs in a field crop, ordered left to right.

    Returns (bounding box, normalised glyph) pairs.

    Components are filtered on **aspect ratio**, not on width relative to the
    crop. Aspect is a property of the glyph itself -- no digit in a normal face
    is wider than it is tall -- whereas a width-versus-crop rule depends on how
    tightly the field happens to be cropped. A single-digit score field is
    barely wider than its digit, so a relative-width rule rejects exactly the
    glyph it is supposed to keep, while a two-digit clock field passes. That
    asymmetry is silent: scores read as None and the score simply never updates.

    Short components (separators, underlines) fall out on ``min_height_frac``;
    merged multi-digit blobs fall out on ``max_aspect``. ``max_width_frac`` is
    kept only as a loose backstop against a component spanning the whole field.
    """
    binary = _binarise(crop)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = binary.shape[:2]
    out = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < min_height_frac * h or bw > max_width_frac * w:
            continue
        if bw < 2 or bh < 4:
            continue
        aspect = bw / bh
        if aspect > max_aspect or aspect < min_aspect:
            continue
        glyph = cv2.resize(
            binary[y : y + bh, x : x + bw], DIGIT_SIZE, interpolation=cv2.INTER_AREA
        )
        out.append(((x, y, bw, bh), glyph))

    out.sort(key=lambda item: item[0][0])
    return out


def match_digit(
    glyph: np.ndarray, templates: dict[int, np.ndarray], min_score: float = 0.72
) -> int | None:
    """Nearest template by normalised agreement. None if nothing matches well."""
    best, best_score = None, -1.0
    g = glyph.astype(np.float32) / 255.0
    for digit, template in templates.items():
        t = template.astype(np.float32) / 255.0
        score = 1.0 - float(np.abs(g - t).mean())
        if score > best_score:
            best, best_score = digit, score
    return best if best_score >= min_score else None


def read_number(
    crop: np.ndarray, templates: dict[int, np.ndarray], max_digits: int = 2
) -> int | None:
    """Read a small integer from a field crop. None if any digit is unreadable.

    All-or-nothing on purpose. A partial read ("1" from "17") is worse than no
    read: the confirmation layer would treat it as a plausible value, whereas a
    None is correctly ignored as a missing observation.
    """
    glyphs = segment_digits(crop)
    if not glyphs or len(glyphs) > max_digits:
        return None

    digits = [match_digit(g, templates) for _, g in glyphs]
    if any(d is None for d in digits):
        return None
    return int("".join(str(d) for d in digits))


@dataclass
class ScoreboardLayout:
    """Where each field sits inside the bug ROI, as fractions of it.

    Fractions rather than pixels so the layout survives a change of resolution
    and can be reused across broadcasts of the same production.
    """

    home_score: tuple[float, float, float, float]   # x, y, w, h
    away_score: tuple[float, float, float, float]
    clock: tuple[float, float, float, float] | None = None

    def crop(self, roi_img: np.ndarray, field: tuple[float, float, float, float]):
        h, w = roi_img.shape[:2]
        x, y, fw, fh = field
        return roi_img[
            int(y * h) : int((y + fh) * h), int(x * w) : int((x + fw) * w)
        ]


class ScoreboardReader:
    """Reads score and clock from the ROI that segmentation already located.

    Takes the :class:`~caster.segmentation.ScoreboardDetector` rather than a
    bare rectangle, for two reasons. The ROI stays single-sourced, so the reader
    and the segmenter cannot disagree about where the bug is. And the detector
    already knows whether the bug is *present* on a given frame -- which turns
    out to be essential, not merely convenient.

    Without that presence gate the reader will happily find contours in the
    pitch itself, pass them through the digit filters, and return a score. On a
    replay -- where the bug is removed but the footage still looks like a match
    -- that is a fabricated scoreline. The transition validator does not save
    you either: it only constrains changes *from* a known value, so a
    hallucinated first reading is accepted outright.

    **Presence and reading use different rectangles, on purpose.** The
    detector's auto-calibrated ROI is approximate: it is grown from whichever
    cells were quietest, so it can be offset from the bug and clipped at its
    edges. That is entirely adequate for answering "is the overlay drawn?",
    which is a whole-region correlation. It is *not* adequate for locating
    individual fields, where being ten pixels out puts the crop on the wrong
    digit. So ``read_roi`` is authored per broadcaster alongside the layout,
    and the detector is used only for the presence gate. Defaulting one to the
    other is convenient and wrong -- it silently reads the wrong pixels.
    """

    def __init__(
        self,
        detector,
        layout: ScoreboardLayout,
        templates: dict[int, np.ndarray] | None = None,
        confirm_frames: int = 5,
        read_roi: tuple[int, int, int, int] | None = None,
    ):
        if not getattr(detector, "calibrated", False):
            raise ValueError("detector must be calibrated before reading")
        self.detector = detector
        self.layout = layout
        self.templates = templates or synthesize_digit_templates()
        self.score = StableValue(confirm_frames, validator=score_transition_valid)
        self.clock = StableValue(confirm_frames, validator=clock_transition_valid)

        if read_roi is None:
            log.warning(
                "no read_roi given; falling back to the detector's approximate "
                "ROI %s. Field crops will be misaligned unless the layout was "
                "authored against that exact rectangle.",
                detector.roi,
            )
        self._read_roi = read_roi or detector.roi

    @property
    def roi(self) -> tuple[int, int, int, int]:
        """Rectangle the field layout is expressed against."""
        return self._read_roi

    def _roi_of(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return frame[y : y + h, x : x + w]

    def read_raw(self, frame: np.ndarray):
        """Single-frame reading with no temporal smoothing. Mostly for debugging."""
        # No bug on this frame means no reading -- not a reading of the pitch.
        if not self.detector.present(frame):
            return None, None

        roi_img = self._roi_of(frame)
        if roi_img.size == 0:
            return None, None

        home = read_number(self.layout.crop(roi_img, self.layout.home_score),
                           self.templates)
        away = read_number(self.layout.crop(roi_img, self.layout.away_score),
                           self.templates)
        score = (home, away) if home is not None and away is not None else None

        clock = None
        if self.layout.clock is not None:
            mins = read_number(self.layout.crop(roi_img, self.layout.clock),
                               self.templates)
            if mins is not None:
                clock = mins * 60
        return score, clock

    def update(self, frame: np.ndarray):
        """Read this frame and fold it into the confirmed values."""
        raw_score, raw_clock = self.read_raw(frame)
        return self.score.update(raw_score), self.clock.update(raw_clock)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_state(
    observations: Iterable[FrameObservation],
    scoreboard: ScoreboardReader | None = None,
    possession: PossessionTracker | None = None,
) -> Iterable[GameState]:
    """Fold per-frame observations into a smoothed GameState stream.

    Only frames from live-wide segments should reach here -- see
    :func:`caster.segmentation.live_segments`. Feeding replays in corrupts both
    the score (the bug is absent, so reads fail) and possession (the tracker
    sees a discontinuous jump and treats it as a turnover).
    """
    possession = possession or PossessionTracker()

    for obs in observations:
        holder, team = possession.update(obs.ball_xy, obs.players)

        score, clock_s = None, None
        if scoreboard is not None and obs.frame is not None:
            score, clock_s = scoreboard.update(obs.frame)

        yield GameState(
            t=obs.t,
            frame_idx=obs.frame_idx,
            ball_xy=obs.ball_xy,
            players=obs.players,
            possessing_track_id=holder,
            possessing_team=team,
            score=score,
            clock_s=clock_s,
        )
