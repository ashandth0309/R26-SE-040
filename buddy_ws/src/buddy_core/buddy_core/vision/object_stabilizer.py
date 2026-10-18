"""Turn noisy per-frame detections into stable "an object is here" facts."""

from __future__ import annotations

from dataclasses import dataclass, field

from buddy_core.vision.objects import ObjectDetection
from buddy_core.vision.tracker import iou


@dataclass
class StabilizerConfig:
    min_hits: int = 3              # frames seen before it is reported
    max_gap_s: float = 2.0         # unseen this long -> gone
    match_iou: float = 0.3
    min_score: float = 0.40
    min_area_frac: float = 0.002   # ignore specks
    ema: float = 0.5               # box and score smoothing


@dataclass
class StableObject:
    object_id: str
    label: str
    category: str
    score: float
    box: tuple[float, float, float, float]
    first_seen: float
    last_seen: float
    hits: int
    confirmed: bool = False


@dataclass
class ObjectEvents:
    appeared: list[StableObject] = field(default_factory=list)
    disappeared: list[StableObject] = field(default_factory=list)


class ObjectStabilizer:
    def __init__(self, cfg: StabilizerConfig | None = None) -> None:
        self.cfg = cfg or StabilizerConfig()
        self._objs: dict[str, StableObject] = {}
        self._n = 0

    def update(self, dets: list[ObjectDetection], now: float,
               frame_area: float) -> ObjectEvents:
        cfg = self.cfg
        ev = ObjectEvents()
        used: set[str] = set()
        ranked = sorted((d for d in dets if d.score >= cfg.min_score
                         and d.area >= cfg.min_area_frac * frame_area),
                        key=lambda d: -d.score)

        for d in ranked:
            best, best_i = None, cfg.match_iou

            for oid, o in self._objs.items():
                if oid in used or o.label != d.label:
                    continue

                v = iou(o.box, d.box)

                if v >= best_i:
                    best, best_i = oid, v

            if best is None:
                self._n += 1
                oid = f"obj_{self._n}"
                self._objs[oid] = StableObject(
                    oid, d.label, d.category, d.score, d.box, now, now, 1)
                used.add(oid)

                continue

            o = self._objs[best]
            a = cfg.ema
            o.box = tuple(a * n + (1 - a) * p
                          for n, p in zip(d.box, o.box, strict=True))
            o.score = a * d.score + (1 - a) * o.score
            o.last_seen = now
            o.hits += 1
            used.add(best)

            if not o.confirmed and o.hits >= cfg.min_hits:
                o.confirmed = True
                ev.appeared.append(o)

        for oid in [k for k, o in self._objs.items()
                    if now - o.last_seen > cfg.max_gap_s]:
            o = self._objs.pop(oid)

            if o.confirmed:
                ev.disappeared.append(o)

        return ev

    def objects(self) -> list[StableObject]:
        return [o for o in self._objs.values() if o.confirmed]
