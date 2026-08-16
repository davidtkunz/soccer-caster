"""Detection, tracking, and team assignment.

Wraps supervision plus the roboflow/sports models. Ball detection is separate
from player detection — the ball is small enough that it needs sliced inference
to be found reliably at the far end of the pitch.

Ball tracking quality is the ceiling on the entire project. Nearly every
narratable event revolves around the ball, so if this runs at 70%, event
detection caps at 70% and commentary quality caps below that. Measure it first.
"""

import supervision as sv


class Perception:
    def __init__(self, player_model, ball_model, device: str = "cuda"):
        self.player_model = player_model
        self.ball_model = ball_model
        self.tracker = sv.ByteTrack()

    def detect_players(self, frame) -> sv.Detections:
        """Full-frame player detection, then tracked for persistent IDs."""
        raise NotImplementedError

    def detect_ball(self, frame) -> sv.Detections:
        """Sliced inference — the ball is too small for a single full-frame pass.

        supervision's InferenceSlicer runs the model over overlapping crops and
        merges the results, which is what makes distant-ball detection viable.
        """
        raise NotImplementedError

    def assign_teams(self, frame, players: sv.Detections):
        """Cluster player crops into two teams.

        Crop each player box, embed the crop, reduce, then cluster into two
        groups. Goalkeepers and referees wear distinct colours and fall out
        either as their own clusters or as outliers — handle them explicitly
        rather than letting them contaminate the two outfield clusters.

        Fit the clustering once on a sample of frames from the match and reuse
        it, rather than re-clustering per frame: per-frame fitting makes team
        labels swap unpredictably.
        """
        raise NotImplementedError
