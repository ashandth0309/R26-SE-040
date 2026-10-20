"""Face following: lock one face track and keep it near the image centre.

Inputs are plain ``FaceFrame`` objects built from Sobiya's ``VisionPeople``
(bbox in pixels, ``person_key`` as the track id).  Only the track id and
the box geometry are used; names and identities are never read or kept.

Target selection never jumps between people:
* a locked track is kept while it is visible, even if another face is
  larger or closer to the centre;
* a new target needs ``acquire_frames`` consecutive sightings;
* the caller's track (when association is reliable) is preferred once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from buddy_core.head_tracking.config import FaceConfig
from buddy_core.head_tracking.servo import VisualServo2D
from buddy_core.head_tracking.models import FaceFrame, FaceObservation


class TrackStatus(str, Enum):
    TRACKING = "TRACKING"
    ACQUIRING = "ACQUIRING"
    HOLDING = "HOLDING"       # target briefly missing, pose is held
    LOST = "LOST"
    NO_TARGET = "NO_TARGET"
    STALE = "STALE"           # frame too old or repeated, ignored


@dataclass
class FollowResult:
    status: TrackStatus
    track_id: str | None = None
    pan: float | None = None       # absolute target, None = no command
    tilt: float | None = None
    ex: float = 0.0
    ey: float = 0.0


PoseAt = Callable[[float], tuple[float, float]]


def _center(f: FaceObservation, aim: float) -> tuple[float, float]:
    return f.x + f.w / 2.0, f.y + f.h * aim


class FaceFollower:
    def __init__(self, cfg: FaceConfig | None = None, pan_sign: int = 1,
                 tilt_sign: int = 1) -> None:
        self.cfg = cfg or FaceConfig()
        self.servo = VisualServo2D(self.cfg.servo, pan_sign, tilt_sign)
        self.locked: str | None = None
        self.preferred: str | None = None
        self._cand_id: str | None = None
        self._cand_n = 0
        self._last_seen: float | None = None
        self._last_frame_t: float | None = None
        self.last_ex = 0.0
        self.last_ey = 0.0
        self.last_face: FaceObservation | None = None

    # ------------------------------------------------------------ control
    def set_preferred(self, track_id: str | None) -> None:
        """Prefer this track (the caller) the next time it is visible."""

        self.preferred = track_id

    def reset(self) -> None:
        self.locked = None
        self._cand_id, self._cand_n = None, 0
        self._last_seen = None
        self.servo.reset()

    @property
    def has_target(self) -> bool:
        return self.locked is not None

    # ------------------------------------------------------------- update
    def _valid(self, faces) -> list[FaceObservation]:
        c = self.cfg
        return [
            f for f in faces
            if f.w >= c.min_face_px and f.h >= c.min_face_px
            and f.confidence >= c.min_confidence
        ]

    def _score(self, f: FaceObservation, fr: FaceFrame) -> float:
        cx, cy = _center(f, self.cfg.vertical_aim)
        dx = (cx / fr.width - 0.5) * 2
        dy = (cy / fr.height - 0.5) * 2
        return (dx * dx + dy * dy) ** 0.5 - 0.5 * (f.w * f.h) / (
            fr.width * fr.height)

    def update(self, frame: FaceFrame, now: float,
               pose_at: PoseAt) -> FollowResult:
        c = self.cfg
        if (now - frame.t > c.max_obs_age_s
                or (self._last_frame_t is not None
                    and frame.t <= self._last_frame_t)):
            return FollowResult(TrackStatus.STALE, self.locked)
        dt = 0.05 if self._last_frame_t is None else (
            frame.t - self._last_frame_t)
        self._last_frame_t = frame.t

        faces = self._valid(frame.faces)
        by_id = {f.track_id: f for f in faces}

        # caller preference: switch once, only if the caller is visible
        if self.preferred and self.preferred in by_id:
            if self.locked != self.preferred:
                self.locked = self.preferred
                self.servo.reset()
            self.preferred = None

        if self.locked and self.locked in by_id:
            self._last_seen = frame.t
            return self._track(by_id[self.locked], frame, pose_at, dt)

        if self.locked:                 # locked track not visible
            return self._missing(frame.t, faces, frame)

        return self._acquire(faces, frame, frame.t)

    def _track(self, face, frame, pose_at, dt) -> FollowResult:
        self.last_face = face
        cx, cy = _center(face, self.cfg.vertical_aim)
        out = self.servo.step(cx, cy, frame.width, frame.height, dt)
        self.last_ex, self.last_ey = out.ex, out.ey
        pan0, tilt0 = pose_at(frame.t)
        return FollowResult(
            TrackStatus.TRACKING, face.track_id,
            pan0 + out.d_pan_deg, tilt0 + out.d_tilt_deg, out.ex, out.ey)

    def _missing(self, t, faces, frame) -> FollowResult:
        if self._last_seen is not None and (
                t - self._last_seen <= self.cfg.lost_hold_s):
            return FollowResult(TrackStatus.HOLDING, self.locked)
        lost = self.locked
        self.reset()
        res = self._acquire(faces, frame, t)
        if res.status is TrackStatus.NO_TARGET:
            return FollowResult(TrackStatus.LOST, lost)
        return res

    def _acquire(self, faces, frame, t) -> FollowResult:
        if not faces:
            self._cand_id, self._cand_n = None, 0
            return FollowResult(TrackStatus.NO_TARGET)
        best = min(faces, key=lambda f: self._score(f, frame))
        if best.track_id == self._cand_id:
            self._cand_n += 1
        else:
            self._cand_id, self._cand_n = best.track_id, 1
        if self._cand_n >= self.cfg.acquire_frames:
            self.locked = best.track_id
            self._last_seen = t
            self.servo.reset()
            return FollowResult(TrackStatus.ACQUIRING, best.track_id)
        return FollowResult(TrackStatus.ACQUIRING, None)

    def check_timeout(self, now: float) -> FollowResult | None:
        """Call every tick: a locked target whose frames stopped arriving
        (camera died, node stalled) becomes HOLDING and then LOST."""

        if self.locked is None or self._last_seen is None:
            return None
        if now - self._last_seen <= self.cfg.lost_hold_s:
            if now - self._last_seen > self.cfg.max_obs_age_s:
                return FollowResult(TrackStatus.HOLDING, self.locked)
            return None
        lost = self.locked
        self.reset()
        return FollowResult(TrackStatus.LOST, lost)

    # --------------------------------------------------- caller association
    def associate_caller(self, frame: FaceFrame) -> str | None:
        """Pick the caller's track after Buddy turned toward the voice.

        Reliable only when exactly one valid face is visible, or the face
        closest to the image centre is clearly closer than the next one
        (``assoc_margin``).  Otherwise returns None: no guess is made.
        """

        faces = self._valid(frame.faces)
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0].track_id

        def off(f):
            cx, _ = _center(f, self.cfg.vertical_aim)
            return abs(cx / frame.width - 0.5) * 2 + 1e-3

        ranked = sorted(faces, key=off)
        if off(ranked[1]) / off(ranked[0]) >= self.cfg.assoc_margin:
            return ranked[0].track_id
        return None
