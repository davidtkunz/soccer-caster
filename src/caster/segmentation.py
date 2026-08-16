"""Shot-boundary and replay detection.

Runs before everything else in the pipeline. Broadcast feeds cut between the
main tactical view, close-ups, crowd shots, replays, and graphics. Feed all of
that downstream and a replayed goal registers as a second goal.

Two heuristics carry most of the weight:

1. The main tactical view is the one dominated by pitch green.
2. The scoreboard overlay is *removed during replays*.

**Known limitation.** Cut detection is colour-histogram based, so it does not
fire on a cut between two shots that look alike -- and a replay of wide pitch
play looks almost exactly like live wide pitch play. Confirmed on synthetic
footage: crowd and graphic boundaries are found cleanly, live/replay boundaries
are not. In real broadcasts this is usually masked, because replays are
introduced with a branded wipe that the histogram does catch. Where it is not,
the fix is to treat a *change in scoreboard presence* as a boundary in its own
right -- the bug vanishing is a stronger signal for this specific transition
than anything in the colour histogram. Left unimplemented until there is real
footage to validate it against.

That second one is the important one, and it is why this module bothers to
locate the scoreboard at all. Green ratio alone cannot separate a live wide
shot from a replay of a live wide shot -- both are pitch footage. The presence
or absence of the score bug can.

The scoreboard region is found automatically rather than configured per
broadcaster: the bug is a static overlay, so across a match its pixels vary far
less over time than the panning pitch behind it. A per-pixel temporal variance
map picks it out.

Typical use::

    from caster.segmentation import Segmenter

    seg = Segmenter()
    segments = seg.analyze_video("match.mp4")
    for s in segments:
        if s.kind is ShotClass.LIVE_WIDE:
            ...  # only these reach the event pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


class ShotClass(str, Enum):
    LIVE_WIDE = "live_wide"   # main tactical view -- the only one we use
    REPLAY = "replay"
    CLOSE_UP = "close_up"
    CROWD = "crowd"
    GRAPHIC = "graphic"


@dataclass
class Segment:
    start_frame: int
    end_frame: int            # exclusive
    kind: ShotClass
    features: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    def duration(self, fps: float) -> float:
        return self.n_frames / fps if fps else 0.0

    def __contains__(self, frame_idx: int) -> bool:
        return self.start_frame <= frame_idx < self.end_frame


# --------------------------------------------------------------------------
# Frame-level primitives
# --------------------------------------------------------------------------

#: OpenCV hue is 0-179. Pitch grass sits in roughly 35-85 across broadcasters,
#: floodlit or daylight. Saturation/value floors reject grey stands and shadow.
GREEN_LO = np.array([35, 40, 40], dtype=np.uint8)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)

SIG_SIZE = (160, 90)          # frames are downsampled before hashing
HIST_BINS = (16, 16)          # hue x saturation


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """Compact hue/saturation histogram used for shot-boundary detection.

    Value is deliberately excluded: it swings with floodlights and shadow while
    the underlying shot stays the same, which would produce false cuts.
    """
    small = cv2.resize(frame, SIG_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance in [0, 1]. Bounded, so thresholds stay meaningful."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def green_ratio(frame: np.ndarray) -> float:
    """Fraction of the frame that is pitch-green."""
    small = cv2.resize(frame, SIG_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    return float(np.count_nonzero(mask)) / mask.size


def edge_density(frame: np.ndarray) -> float:
    """Fraction of edge pixels. Crowd shots are high; pitch and graphics low."""
    small = cv2.resize(frame, SIG_SIZE, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 80, 200)
    return float(np.count_nonzero(edges)) / edges.size


# --------------------------------------------------------------------------
# Scoreboard localisation
# --------------------------------------------------------------------------


class ScoreboardDetector:
    """Locates the broadcast score bug, then reports its presence per frame.

    Calibration is automatic. The bug is a static overlay, so its pixels have
    much lower temporal variance than the panning pitch behind it. We build a
    variance map over sampled frames, score candidate cells in the top and
    bottom bands (where bugs live), and take the lowest-variance run.

    Presence is then normalised correlation against the calibrated median
    template. The clock digits change, but they are a small fraction of the
    bug's area, so correlation stays high while the overlay is drawn and
    collapses when it is removed for a replay.
    """

    #: Bands of the frame height searched for the bug, as (top, bottom) fractions.
    SEARCH_BANDS = ((0.0, 0.22), (0.78, 1.0))
    GRID = (24, 12)               # cells across, cells down
    SEED_CELLS = 2                # width of the initial quiet-run probe
    QUIET_FRACTION = 0.60         # cell counts as overlay below this x frame mean
    MAX_GROW_COLS = 16
    MAX_GROW_ROWS = 4

    #: A score bug is quiet *and graphical*. Empty turf near the frame edge is
    #: quiet and flat, and without these checks it gets mistaken for an overlay
    #: on footage that has none. Measured separation is wide -- a real bug runs
    #: std ~50 and edge density ~0.15; bare turf runs ~1 and ~0.00 -- so these
    #: thresholds sit far below anything a genuine overlay produces.
    MIN_TEMPLATE_STD = 12.0
    MIN_TEMPLATE_EDGES = 0.02

    def __init__(self, present_threshold: float = 0.45):
        self.present_threshold = present_threshold
        self.roi: tuple[int, int, int, int] | None = None   # x, y, w, h
        self.template: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self.roi is not None and self.template is not None

    def calibrate(self, frames: list[np.ndarray]) -> bool:
        """Find the bug from a sample of frames spread across the match.

        Returns False if no plausible static overlay was found, in which case
        the caller must fall back to green ratio alone.
        """
        if len(frames) < 8:
            log.warning("scoreboard calibration needs >=8 frames, got %d", len(frames))
            return False

        h, w = frames[0].shape[:2]
        greys = np.stack(
            [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
        )
        variance = greys.std(axis=0)

        cols, rows = self.GRID
        cell_w, cell_h = w // cols, h // rows
        if cell_w == 0 or cell_h == 0:
            return False

        def cell_var(r: int, c0: int, c1: int) -> float:
            return float(
                variance[
                    r * cell_h : (r + 1) * cell_h, c0 * cell_w : (c1 + 1) * cell_w
                ].mean()
            )

        frame_var = float(variance.mean())
        quiet_cut = self.QUIET_FRACTION * frame_var

        # Seed: the quietest short run of cells anywhere in the search bands.
        best = None
        for band_lo, band_hi in self.SEARCH_BANDS:
            r_lo = int(band_lo * rows)
            r_hi = max(r_lo + 1, int(band_hi * rows))
            for r in range(r_lo, min(r_hi, rows)):
                for start in range(cols - self.SEED_CELLS + 1):
                    v = cell_var(r, start, start + self.SEED_CELLS - 1)
                    if best is None or v < best[0]:
                        best = (v, r, start)

        if best is None:
            return False
        seed_var, r, start = best

        # Reject a "find" that is no quieter than the frame as a whole -- that
        # means there is no overlay, not that we found one.
        if seed_var > quiet_cut:
            log.warning(
                "no static overlay found (best region var %.1f vs frame %.1f)",
                seed_var,
                frame_var,
            )
            return False

        # Grow the seed outward while neighbouring cells are still overlay-quiet.
        # Growing beats tuning a width reward: it recovers the bug's true extent
        # instead of locking onto whichever few cells happen to be stillest, and
        # more area makes the presence correlation far more robust.
        c0, c1 = start, start + self.SEED_CELLS - 1
        while c0 > 0 and cell_var(r, c0 - 1, c0 - 1) < quiet_cut and (c1 - c0 + 1) < self.MAX_GROW_COLS:
            c0 -= 1
        while c1 < cols - 1 and cell_var(r, c1 + 1, c1 + 1) < quiet_cut and (c1 - c0 + 1) < self.MAX_GROW_COLS:
            c1 += 1

        r0 = r1 = r
        while r0 > 0 and cell_var(r0 - 1, c0, c1) < quiet_cut and (r1 - r0 + 1) < self.MAX_GROW_ROWS:
            r0 -= 1
        while r1 < rows - 1 and cell_var(r1 + 1, c0, c1) < quiet_cut and (r1 - r0 + 1) < self.MAX_GROW_ROWS:
            r1 += 1

        x, y = c0 * cell_w, r0 * cell_h
        bw, bh = (c1 - c0 + 1) * cell_w, (r1 - r0 + 1) * cell_h

        crops = np.stack([f[y : y + bh, x : x + bw] for f in frames])
        template = np.median(crops, axis=0).astype(np.uint8)

        # Structure gate. The median across frames washes out anything that
        # moved, so what survives is the genuinely static content -- an overlay
        # if there is one, flat turf if there is not. Testing the template for
        # graphical structure is what separates the two; a variance threshold
        # alone accepts any quiet corner of the pitch.
        grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        t_std = float(grey.std())
        t_edges = float(
            np.count_nonzero(cv2.Canny(grey, 60, 180))
        ) / max(1, grey.size)

        if t_std < self.MIN_TEMPLATE_STD or t_edges < self.MIN_TEMPLATE_EDGES:
            log.warning(
                "quiet region at (%d,%d) is not graphical (std %.1f, edges %.3f) "
                "-- treating as no overlay",
                x, y, t_std, t_edges,
            )
            return False

        self.roi = (x, y, bw, bh)
        self.template = template
        log.info(
            "scoreboard calibrated at x=%d y=%d w=%d h=%d (std %.1f, edges %.3f)",
            x, y, bw, bh, t_std, t_edges,
        )
        return True

    def score(self, frame: np.ndarray) -> float:
        """Correlation of this frame's ROI against the calibrated template."""
        if not self.calibrated:
            return 0.0
        x, y, w, h = self.roi
        crop = frame[y : y + h, x : x + w]
        if crop.shape[:2] != self.template.shape[:2]:
            return 0.0
        res = cv2.matchTemplate(crop, self.template, cv2.TM_CCOEFF_NORMED)
        return float(res.max())

    def present(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.present_threshold


# --------------------------------------------------------------------------
# Cut detection
# --------------------------------------------------------------------------


def find_cuts(
    distances: np.ndarray,
    min_shot_frames: int = 12,
    window: int = 25,
    k: float = 4.0,
    floor: float = 0.10,
) -> list[int]:
    """Frame indices where a new shot begins.

    ``distances[i]`` is the signature distance between frame ``i`` and ``i+1``,
    so a detection at ``i`` means the new shot starts at ``i+1``.

    The threshold is local rather than global: broadcast footage varies in how
    much motion it carries, and a threshold tuned on a static wide shot fires
    constantly during a handheld touchline shot. Each frame is compared against
    the mean and spread of its own neighbourhood, with an absolute floor so a
    very quiet passage cannot make trivial changes look like cuts.
    """
    distances = np.asarray(distances, dtype=np.float64)
    n = distances.size
    if n == 0:
        return []

    peaks: list[int] = []
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        neighbourhood = np.concatenate([distances[lo:i], distances[i + 1 : hi]])
        if neighbourhood.size == 0:
            continue
        threshold = max(floor, neighbourhood.mean() + k * neighbourhood.std())
        if distances[i] > threshold:
            peaks.append(i)

    # Collapse peaks that sit inside one transition, and enforce a minimum
    # shot length -- broadcast shots are rarely shorter than half a second.
    merged: list[int] = []
    for p in peaks:
        if merged and p - merged[-1] < min_shot_frames:
            if distances[p] > distances[merged[-1]]:
                merged[-1] = p
            continue
        merged.append(p)

    return [p + 1 for p in merged]


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

GREEN_WIDE = 0.45     # above this, we are looking at the pitch
GREEN_SOME = 0.15     # above this, pitch is visible but not dominant
EDGE_BUSY = 0.10      # crowd shots are texturally busy


def classify(features: dict, scoreboard_available: bool = True) -> ShotClass:
    """Map aggregated segment features onto a shot class.

    Deliberately a rule cascade rather than a learned model. The rules are
    inspectable, tunable against the eval set, and correct for the reason you
    think they are -- which matters more here than squeezing out the last few
    points of accuracy.
    """
    green = features.get("green_ratio", 0.0)
    board = features.get("scoreboard", 0.0)
    edges = features.get("edge_density", 0.0)

    if green >= GREEN_WIDE:
        # Pitch footage. Live or replay? This is the only signal that separates
        # them, which is the whole reason the scoreboard detector exists.
        if not scoreboard_available:
            # No calibration -- assume live rather than silently dropping the
            # match. The caller is warned separately.
            return ShotClass.LIVE_WIDE
        return ShotClass.LIVE_WIDE if board else ShotClass.REPLAY

    if green >= GREEN_SOME:
        return ShotClass.CLOSE_UP

    return ShotClass.CROWD if edges >= EDGE_BUSY else ShotClass.GRAPHIC


# --------------------------------------------------------------------------
# Segmenter
# --------------------------------------------------------------------------


class Segmenter:
    """Splits footage into shots and labels each one.

    Args:
        min_shot_frames: shortest run of frames treated as its own shot.
        feature_stride: sample every Nth frame for classification features.
            Signatures are computed on every frame (cuts need that resolution);
            features are far more expensive and change slowly within a shot.
        calibration_samples: frames sampled across the video to locate the bug.
    """

    def __init__(
        self,
        min_shot_frames: int = 12,
        feature_stride: int = 5,
        calibration_samples: int = 120,
        scoreboard: ScoreboardDetector | None = None,
    ):
        self.min_shot_frames = min_shot_frames
        self.feature_stride = feature_stride
        self.calibration_samples = calibration_samples
        self.scoreboard = scoreboard or ScoreboardDetector()

    # -- calibration -------------------------------------------------------

    def calibrate_from_video(self, path: str) -> bool:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total <= 0:
                frames = []
                while len(frames) < self.calibration_samples:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
            else:
                stride = max(1, total // self.calibration_samples)
                frames = []
                for idx in range(0, total, stride):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if ok:
                        frames.append(frame)
            return self.scoreboard.calibrate(frames)
        finally:
            cap.release()

    # -- analysis ----------------------------------------------------------

    def analyze(self, frames) -> list[Segment]:
        """Segment an iterable of BGR frames.

        The scoreboard detector should already be calibrated (see
        ``calibrate_from_video``). If it is not, classification falls back to
        green ratio alone and cannot distinguish replays -- a warning is logged.
        """
        if not self.scoreboard.calibrated:
            log.warning(
                "scoreboard not calibrated; replays will be labelled LIVE_WIDE"
            )

        signatures: list[np.ndarray] = []
        per_frame: dict[int, dict] = {}

        for idx, frame in enumerate(frames):
            signatures.append(frame_signature(frame))
            if idx % self.feature_stride == 0:
                per_frame[idx] = {
                    "green_ratio": green_ratio(frame),
                    "edge_density": edge_density(frame),
                    "scoreboard": self.scoreboard.score(frame),
                }

        n = len(signatures)
        if n == 0:
            return []
        if n == 1:
            return [Segment(0, 1, ShotClass.GRAPHIC, per_frame.get(0, {}))]

        distances = np.array(
            [signature_distance(signatures[i], signatures[i + 1]) for i in range(n - 1)]
        )
        cuts = find_cuts(distances, min_shot_frames=self.min_shot_frames)

        starts = [0] + cuts
        ends = cuts + [n]

        segments: list[Segment] = []
        for start, end in zip(starts, ends):
            samples = [f for i, f in per_frame.items() if start <= i < end]
            if not samples:
                # Shot shorter than the feature stride -- measure its midpoint.
                samples = [per_frame[min(per_frame, key=lambda i: abs(i - start))]]
            agg = {
                "green_ratio": float(np.mean([s["green_ratio"] for s in samples])),
                "edge_density": float(np.mean([s["edge_density"] for s in samples])),
                # Median, not mean: a few frames of a wipe transition should not
                # drag an otherwise clearly-present bug below threshold.
                "scoreboard": float(np.median([s["scoreboard"] for s in samples])),
            }
            board_present = agg["scoreboard"] >= self.scoreboard.present_threshold
            kind = classify(
                {**agg, "scoreboard": board_present},
                scoreboard_available=self.scoreboard.calibrated,
            )
            segments.append(Segment(start, end, kind, agg))

        return segments

    def analyze_video(self, path: str, calibrate: bool = True) -> list[Segment]:
        """Calibrate on a sample pass, then segment the whole video."""
        if calibrate:
            self.calibrate_from_video(path)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")

        def stream():
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    yield frame
            finally:
                cap.release()

        return self.analyze(stream())


def live_segments(segments: list[Segment]) -> list[Segment]:
    """The only segments the rest of the pipeline is allowed to see.

    Note that a close-up shown *during* a replay is labelled CLOSE_UP rather
    than REPLAY. That is fine -- both are filtered out here. The distinction
    only has to be right for wide pitch footage, which is the case that would
    otherwise produce a duplicate goal.
    """
    return [s for s in segments if s.kind is ShotClass.LIVE_WIDE]


def coverage(segments: list[Segment]) -> dict[ShotClass, float]:
    """Fraction of frames per class. A quick sanity check on a new broadcaster.

    Live wide should dominate a full match, typically 0.55-0.75. A much lower
    figure usually means calibration picked the wrong region, not that the
    broadcaster is unusual.
    """
    total = sum(s.n_frames for s in segments)
    if not total:
        return {}
    out: dict[ShotClass, float] = {}
    for s in segments:
        out[s.kind] = out.get(s.kind, 0.0) + s.n_frames / total
    return out


def dump_samples(video_path: str, segments, detector: ScoreboardDetector, out_dir: str,
                 per_class: int = 4) -> None:
    """Write sample frames per class plus the scoreboard evidence.

    Classification thresholds cannot be tuned from numbers alone -- you have to
    see what the segmenter called a replay. This writes enough to check the two
    things that actually go wrong: whether the ROI landed on the score bug, and
    whether live-wide and replay are being told apart.
    """
    import os
    from collections import defaultdict

    os.makedirs(out_dir, exist_ok=True)
    wanted: dict[ShotClass, list[int]] = defaultdict(list)
    for s in segments:
        if len(wanted[s.kind]) < per_class:
            wanted[s.kind].append((s.start_frame + s.end_frame) // 2)

    targets = {idx: kind for kind, idxs in wanted.items() for idx in idxs}
    cap = cv2.VideoCapture(video_path)
    written = 0
    try:
        for idx in sorted(targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            kind = targets[idx]
            annotated = frame.copy()
            if detector.calibrated:
                x, y, w, h = detector.roi
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 215, 255), 2)
                cv2.putText(annotated, f"board={detector.score(frame):.2f}",
                            (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 215, 255), 2)
            cv2.putText(annotated, f"{kind.value}  frame {idx}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out_dir, f"{kind.value}_{idx:07d}.jpg"), annotated)
            written += 1
    finally:
        cap.release()

    if detector.calibrated and detector.template is not None:
        scale = max(1, 240 // max(1, detector.template.shape[0]))
        big = cv2.resize(
            detector.template, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(os.path.join(out_dir, "scoreboard_template.png"), big)

    print(f"\nwrote {written} sample frames to {out_dir}/")
    if detector.calibrated:
        print("  scoreboard_template.png -- what calibration locked onto.")
        print("  Check it is the score bug and not a quiet patch of turf.")


def main(argv=None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Segment broadcast soccer footage.")
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=None,
                        help="override the video's reported frame rate")
    parser.add_argument("--dump", default=None, metavar="DIR",
                        help="write sample frames per class for inspection")
    parser.add_argument("--limit", type=int, default=None,
                        help="only analyse the first N frames")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    probe = cv2.VideoCapture(args.video)
    if not probe.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = args.fps or probe.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    print(f"{args.video}: {total} frames @ {fps:.2f} fps "
          f"({total / fps / 60:.1f} min)\n")

    seg = Segmenter()
    seg.calibrate_from_video(args.video)

    cap = cv2.VideoCapture(args.video)

    def stream():
        try:
            n = 0
            while args.limit is None or n < args.limit:
                ok, frame = cap.read()
                if not ok:
                    return
                n += 1
                yield frame
        finally:
            cap.release()

    segments = seg.analyze(stream())

    for s in segments:
        print(
            f"{s.start_frame:>7} - {s.end_frame:<7} "
            f"{s.duration(fps):>6.1f}s  {s.kind.value:<10} "
            f"green={s.features.get('green_ratio', 0):.2f} "
            f"board={s.features.get('scoreboard', 0):.2f}"
        )

    print("\ncoverage:")
    for kind, frac in sorted(coverage(segments).items(), key=lambda kv: -kv[1]):
        print(f"  {kind.value:<10} {frac:6.1%}")

    live = coverage(segments).get(ShotClass.LIVE_WIDE, 0.0)
    print()
    if not seg.scoreboard.calibrated:
        print("[!] no scoreboard found -- replays cannot be distinguished from")
        print("    live play, so every wide shot is labelled live_wide.")
    elif 0.55 <= live <= 0.80:
        print(f"[ok] live_wide {live:.0%} -- in the expected range for a match.")
    else:
        print(f"[!] live_wide {live:.0%} -- expected roughly 55-75%.")
        print("    Usually means calibration locked onto the wrong region,")
        print("    or GREEN_WIDE needs tuning for this broadcaster.")

    if args.dump:
        dump_samples(args.video, segments, seg.scoreboard, args.dump)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
