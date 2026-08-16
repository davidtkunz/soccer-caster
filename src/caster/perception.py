"""Detection, tracking, and team assignment.

Models are **injected, not imported**. The detection weights are the heavy,
environment-specific part of this pipeline; everything around them -- slicing,
tracking, clustering players into teams, filtering out referees -- is ordinary
logic that benefits from being testable without a GPU. Keeping the model behind
a callable boundary means the glue can be exercised with fakes, and the fakes
are where the bugs actually show up.

Ball tracking quality is the ceiling on the whole project. Nearly every
narratable event revolves around the ball, so if this runs at 70%, event
detection caps at 70% and commentary quality caps below that. Measure it first,
before tuning anything downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import supervision as sv

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Injected boundaries
# --------------------------------------------------------------------------


class DetectionModel(Protocol):
    """Anything that maps a frame to detections.

    Wrap whatever you are running -- an ultralytics model, a Roboflow inference
    client, a roboflow/sports checkpoint -- in a callable with this shape.
    """

    def __call__(self, frame: np.ndarray) -> sv.Detections: ...


class Embedder(Protocol):
    """Maps player crops to an (N, D) feature array for team clustering."""

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray: ...


# --------------------------------------------------------------------------
# Team assignment
# --------------------------------------------------------------------------


def _kmeans(x: np.ndarray, k: int = 2, iters: int = 50, seed: int = 0):
    """Minimal k-means. Deterministic seeding, which matters here.

    Two clusters over a few hundred crops does not justify a scikit-learn
    dependency, and a fixed seed means the same footage produces the same team
    labels on every run -- which a random init would not.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < k:
        raise ValueError(f"need at least {k} samples, got {n}")

    # k-means++ style seeding: first centre at random, rest far from the others.
    centres = [x[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(x - c, axis=1) for c in centres], axis=0)
        centres.append(x[int(np.argmax(d))])
    centres = np.stack(centres)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.stack([np.linalg.norm(x - c, axis=1) for c in centres], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            if np.any(labels == i):
                centres[i] = x[labels == i].mean(axis=0)
    return labels, centres


class TeamAssigner:
    """Clusters player crops into two teams.

    **Fit once per match, then only predict.** Re-clustering per frame makes the
    cluster-to-team mapping swap unpredictably, so player 7 is home in one frame
    and away in the next -- which the event layer reads as a continuous stream of
    turnovers. Fitting on a sample spread across the match and freezing it is
    what keeps labels stable.

    Goalkeepers and referees wear colours that match neither outfield kit, so
    they land far from both centroids. Rather than forcing them into a team,
    crops further than ``max_distance_frac`` of the *inter-kit separation* from
    the nearest centroid are returned as ``None``.

    The separation is the right scale to measure against, not the within-cluster
    spread. Spread collapses toward zero whenever the kits are uniform, which
    would make the outlier threshold vanish (or, if guarded, become infinite and
    catch nothing). The gap between the two kits is always meaningful, so "more
    than half way to the other kit, and not near either" is a stable rule.

    Returning ``None`` is the deliberately safe failure. An unassigned player
    leaves possession without a team, which downstream handles; a *wrongly*
    assigned player produces phantom turnovers.
    """

    def __init__(
        self,
        embedder: Embedder,
        team_names: tuple[str, str] = ("home", "away"),
        max_distance_frac: float = 0.6,
        seed: int = 0,
    ):
        self.embedder = embedder
        self.team_names = team_names
        self.max_distance_frac = max_distance_frac
        self.seed = seed
        self.centres: np.ndarray | None = None
        self._limit: float = float("inf")

    @property
    def fitted(self) -> bool:
        return self.centres is not None

    def fit(self, crops: list[np.ndarray]) -> "TeamAssigner":
        features = np.asarray(self.embedder(crops), dtype=np.float64)
        labels, centres = _kmeans(features, k=2, seed=self.seed)
        self.centres = centres

        separation = float(np.linalg.norm(centres[0] - centres[1]))
        self._limit = self.max_distance_frac * separation
        if separation == 0:
            log.warning(
                "the two kit clusters are identical; team assignment will be "
                "arbitrary. Check the embedder and the fitting sample."
            )
            self._limit = float("inf")
        return self

    def predict(self, crops: list[np.ndarray]) -> list[str | None]:
        if not crops:
            return []
        if not self.fitted:
            raise RuntimeError("TeamAssigner.fit must be called before predict")

        features = np.asarray(self.embedder(crops), dtype=np.float64)
        dists = np.stack(
            [np.linalg.norm(features - c, axis=1) for c in self.centres], axis=1
        )
        best = np.argmin(dists, axis=1)
        closest = dists[np.arange(len(features)), best]

        return [
            self.team_names[int(b)] if c <= self._limit else None
            for b, c in zip(best, closest)
        ]


def crop_detections(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """Pixel crops for each detection box, clipped to the frame."""
    h, w = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2 in detections.xyxy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            crops.append(frame[y1:y2, x1:x2])
        else:
            crops.append(np.zeros((1, 1, 3), np.uint8))
    return crops


# --------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------


@dataclass
class FrameDetections:
    players: sv.Detections
    ball: sv.Detections
    teams: list[str | None] = field(default_factory=list)

    @property
    def ball_xy(self) -> tuple[float, float] | None:
        """Highest-confidence ball centre in image coordinates, if any."""
        if len(self.ball) == 0:
            return None
        i = (
            int(np.argmax(self.ball.confidence))
            if self.ball.confidence is not None
            else 0
        )
        x1, y1, x2, y2 = self.ball.xyxy[i]
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def default_tracker(frame_rate: float = 25.0):
    """Tracker tuned for broadcast footage.

    BoTSORT rather than ByteTrack, specifically for **camera motion
    compensation**. A broadcast camera pans constantly, so between frames every
    box shifts from camera movement rather than player movement -- and an
    IoU-based association that does not account for that drops tracks on every
    pan. BoTSORT estimates the camera transform and compensates before matching.

    (``sv.ByteTrack`` is also deprecated as of supervision 0.28 and removed in
    0.31; the maintained implementations live in the ``trackers`` package.)
    """
    from trackers import BoTSORTTracker

    return BoTSORTTracker(frame_rate=frame_rate, enable_cmc=True)


class Perception:
    """Per-frame detection, tracking, and team labelling.

    Args:
        player_model: callable frame -> Detections for players.
        ball_model: callable frame -> Detections for the ball.
        team_assigner: fitted TeamAssigner, or None to skip team labelling.
        tracker: object with ``update(detections, frame) -> Detections``.
            Defaults to BoTSORT with camera motion compensation.
        ball_slice_wh: tile size for sliced ball inference. The ball is only a
            handful of pixels at the far end of the pitch, so a single
            full-frame pass misses it; slicing runs the model over overlapping
            tiles and merges the results.
    """

    def __init__(
        self,
        player_model: DetectionModel,
        ball_model: DetectionModel,
        team_assigner: TeamAssigner | None = None,
        tracker=None,
        ball_slice_wh: tuple[int, int] = (640, 640),
        ball_overlap_wh: tuple[int, int] = (128, 128),
        frame_rate: float = 25.0,
    ):
        self.player_model = player_model
        self.ball_model = ball_model
        self.team_assigner = team_assigner
        self.tracker = tracker if tracker is not None else default_tracker(frame_rate)
        self._slicer = sv.InferenceSlicer(
            callback=self._ball_callback,
            slice_wh=ball_slice_wh,
            overlap_wh=ball_overlap_wh,
        )

    def _ball_callback(self, tile: np.ndarray) -> sv.Detections:
        return self.ball_model(tile)

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """Detect and track players, producing persistent ``tracker_id``s."""
        detections = self.player_model(frame)
        return self.tracker.update(detections, frame)

    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        """Sliced inference -- a full-frame pass misses the distant ball."""
        return self._slicer(frame)

    def assign_teams(self, frame: np.ndarray, players: sv.Detections):
        if self.team_assigner is None or len(players) == 0:
            return [None] * len(players)
        return self.team_assigner.predict(crop_detections(frame, players))

    def process(self, frame: np.ndarray) -> FrameDetections:
        players = self.detect_players(frame)
        ball = self.detect_ball(frame)
        teams = self.assign_teams(frame, players)
        return FrameDetections(players=players, ball=ball, teams=teams)


def sample_frames_for_fitting(frames, every: int = 25, limit: int = 200):
    """Frames spread through the footage, for fitting the team assigner.

    Spread rather than consecutive: a single passage of play can easily contain
    only one team's players in shot, and clustering that produces two clusters
    of the same kit.
    """
    out = []
    for i, frame in enumerate(frames):
        if i % every == 0:
            out.append(frame)
            if len(out) >= limit:
                break
    return out
