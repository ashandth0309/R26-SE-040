"""Frame container and a drop-oldest fan-out bus.

Many consumers (faces, objects, pose, ...) share one camera.  Each gets its
own bounded queue, so a slow consumer loses old frames instead of delaying
the camera or the other consumers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured image.

    ``image`` is BGR uint8, shape (H, W, 3), and read-only: it is shared by
    every subscriber, so nobody may modify it in place.  ``stamp`` is wall
    clock seconds at capture; ``mono`` is a monotonic time used for ages.
    """

    image: np.ndarray
    seq: int
    stamp: float
    mono: float
    source: str = "camera"

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


def make_frame(image: np.ndarray, seq: int, stamp: float, mono: float,
               source: str = "camera") -> Frame:
    """Validate and freeze an image into a Frame."""

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 \
            or image.ndim != 3 or image.shape[2] != 3 \
            or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("frame must be a uint8 HxWx3 BGR array")

    arr = np.ascontiguousarray(image)
    arr.flags.writeable = False

    return Frame(arr, seq, stamp, mono, source)


@dataclass
class Subscription:
    """A consumer's private, bounded view of the stream."""

    name: str
    maxlen: int = 1
    dropped: int = 0
    received: int = 0
    _queue: deque = field(default_factory=deque, repr=False)
    _cond: threading.Condition = field(default_factory=threading.Condition,
                                       repr=False)
    closed: bool = False

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self.closed:
                return

            if len(self._queue) >= self.maxlen:
                self._queue.popleft()
                self.dropped += 1

            self._queue.append(frame)
            self.received += 1
            self._cond.notify()

    def get(self, timeout: float = 0.0) -> Frame | None:
        """Oldest waiting frame, or None after ``timeout`` seconds."""

        end = time.monotonic() + timeout

        with self._cond:
            while not self._queue:
                left = end - time.monotonic()

                if left <= 0 or self.closed:
                    return None

                self._cond.wait(left)

            return self._queue.popleft()

    def latest(self) -> Frame | None:
        """Newest frame, discarding older ones (counted as dropped)."""

        with self._cond:
            if not self._queue:
                return None

            frame = self._queue[-1]
            self.dropped += len(self._queue) - 1
            self._queue.clear()

            return frame

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class FrameBus:
    """Publish once, deliver to every subscriber without blocking."""

    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()
        self.published = 0

    def subscribe(self, name: str, maxlen: int = 1) -> Subscription:
        sub = Subscription(name=name, maxlen=max(1, int(maxlen)))

        with self._lock:
            self._subs.append(sub)

        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.close()

        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def publish(self, frame: Frame) -> None:
        with self._lock:
            subs = list(self._subs)
            self.published += 1

        for sub in subs:
            sub.put(frame)

    def stats(self) -> dict:
        with self._lock:
            return {
                "published": self.published,
                "subscribers": {s.name: {"received": s.received,
                                         "dropped": s.dropped}
                                for s in self._subs},
            }
