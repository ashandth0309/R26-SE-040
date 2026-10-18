"""The single owner of the camera.

* One process-wide lock file stops a second owner (for example another
  camera node) from opening the same device.
* Capture failures are retried with exponential back-off; a stalled stream
  (frames stop but no error) is treated as a disconnect.
* Frames go to a ``FrameBus``; consumers never touch the hardware.
* ``set_enabled(False)`` releases the device so nothing is captured: this is
  the privacy "camera off" mode, and ``indicator`` is True only while the
  device is open and delivering frames.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from buddy_core.vision.camera_sources import (
    CameraError,
    CameraSource,
    build_source,
)
from buddy_core.vision.frames import FrameBus, make_frame
from buddy_core.vision.resources import ResourceMonitor

try:  # POSIX only; the lock degrades to "not enforced" elsewhere.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class CameraState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    STREAMING = "streaming"
    STALLED = "stalled"
    RECONNECTING = "reconnecting"
    DISABLED = "disabled"


@dataclass
class CameraConfig:
    backend: str = "mock"           # mock | video | opencv | picamera2
    width: int = 640
    height: int = 480
    fps: float = 10.0
    device_index: int = 0
    video_path: str = ""
    pixel_format: str = "RGB888"
    enabled: bool = True
    stale_after_s: float = 2.0
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 10.0
    lock_path: str = "/tmp/buddy_camera.lock"
    fps_window: int = 30


class CameraLock:
    """Exclusive, crash-safe lock: the OS drops it when the process dies."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True

        if fcntl is None:  # pragma: no cover
            return True

        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)

            return False

        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd

        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


class CameraService:
    """Capture loop with recovery.  ``step`` is one deterministic tick."""

    def __init__(self, config: CameraConfig,
                 bus: FrameBus | None = None,
                 source_factory: Callable[[], CameraSource] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self.config = config
        self.bus = bus or FrameBus()
        self._factory = source_factory or (lambda: build_source(
            config.backend, width=config.width, height=config.height,
            fps=config.fps, device_index=config.device_index,
            video_path=config.video_path,
            pixel_format=config.pixel_format))
        self._clock, self._wall = clock, wall
        self._source: CameraSource | None = None
        self._lock = CameraLock(config.lock_path)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._enabled = bool(config.enabled)
        self._monitor = ResourceMonitor(clock)
        self.state = CameraState.DISABLED if not self._enabled \
            else CameraState.STOPPED
        self.seq = 0
        self.reconnects = 0
        self.open_failures = 0
        self.bad_frames = 0
        self.last_error = ""
        self._last_frame_mono: float | None = None
        self._next_attempt = 0.0
        self._backoff = config.reconnect_initial_s
        self._stamps: deque[float] = deque(maxlen=max(2, config.fps_window))

    # ----- privacy / state -----

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def indicator(self) -> bool:
        """True while the camera is really capturing."""

        return self.state == CameraState.STREAMING and \
            self._source is not None and self._source.is_open

    def set_enabled(self, enabled: bool) -> None:
        """Turn the camera on or off.  Off closes the device at once."""

        self._enabled = bool(enabled)

        if not self._enabled:
            self._close()
            self.state = CameraState.DISABLED
        elif self.state == CameraState.DISABLED:
            self.state = CameraState.STOPPED
            self._next_attempt = 0.0
            self._backoff = self.config.reconnect_initial_s

    def frame_age(self) -> float | None:
        if self._last_frame_mono is None:
            return None

        return self._clock() - self._last_frame_mono

    def is_fresh(self, max_age_s: float | None = None) -> bool:
        """True when a recent frame exists.  Vision features must check
        this before acting on perception."""

        age = self.frame_age()
        limit = self.config.stale_after_s if max_age_s is None \
            else max_age_s

        return self.state == CameraState.STREAMING and \
            age is not None and age <= limit

    # ----- internals -----

    def _close(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass

            self._source = None

        self._lock.release()

    def _fail(self, message: str, now: float) -> None:
        self.last_error = message
        self._close()
        self.state = CameraState.RECONNECTING
        self._next_attempt = now + self._backoff
        self._backoff = min(self.config.reconnect_max_s,
                            self._backoff * 2.0)

    def _try_open(self, now: float) -> None:
        if not self._lock.acquire():
            self.open_failures += 1
            self._fail("camera is owned by another process "
                       f"({self.config.lock_path})", now)

            return

        try:
            source = self._factory()
            source.open()
        except Exception as exc:  # CameraError or backend-specific
            self.open_failures += 1
            self._fail(str(exc) or exc.__class__.__name__, now)

            return

        self._source = source
        self.state = CameraState.STARTING

        if self.open_failures or self.reconnects or self.seq:
            self.reconnects += 1

        self._backoff = self.config.reconnect_initial_s
        self._last_frame_mono = None
        self._opened_at = now

    def step(self) -> bool:
        """One capture tick.  Returns True if a frame was published."""

        now = self._clock()

        if not self._enabled:
            if self._source is not None:
                self._close()

            self.state = CameraState.DISABLED

            return False

        if self._source is None:
            if now < self._next_attempt:
                return False

            self._try_open(now)

            if self._source is None:
                return False

        try:
            image = self._source.read()
        except CameraError as exc:
            self._fail(str(exc), now)

            return False
        except Exception as exc:
            self._fail(f"{exc.__class__.__name__}: {exc}", now)

            return False

        if image is None:
            ref = self._last_frame_mono if self._last_frame_mono \
                is not None else self._opened_at

            if now - ref > self.config.stale_after_s:
                self.state = CameraState.STALLED
                self._fail("no frames received (stalled)", now)

            return False

        try:
            frame = make_frame(image, self.seq + 1, self._wall(), now,
                               self._source.name)
        except ValueError as exc:
            self.bad_frames += 1
            self.last_error = str(exc)

            return False

        self.seq += 1
        self._last_frame_mono = now
        self._stamps.append(now)
        self.state = CameraState.STREAMING
        self.bus.publish(frame)

        return True

    # ----- thread control -----

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="camera_service")
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / max(0.1, self.config.fps)

        while not self._stop.is_set():
            began = time.monotonic()
            got = self.step()
            spent = time.monotonic() - began

            if got:
                self._stop.wait(max(0.0, period - spent))
            else:
                self._stop.wait(min(0.05, period))

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._close()

        if self._enabled:
            self.state = CameraState.STOPPED

    # ----- diagnostics -----

    def measured_fps(self) -> float | None:
        """Frame rate over the recent window; None until two frames."""

        if len(self._stamps) < 2:
            return None

        span = self._stamps[-1] - self._stamps[0]

        return (len(self._stamps) - 1) / span if span > 0 else None

    def health(self) -> dict:
        info = {
            "state": self.state.value,
            "backend": self.config.backend,
            "enabled": self._enabled,
            "indicator": self.indicator,
            "frames": self.seq,
            "measured_fps": self.measured_fps(),
            "frame_age_s": self.frame_age(),
            "fresh": self.is_fresh(),
            "reconnects": self.reconnects,
            "open_failures": self.open_failures,
            "bad_frames": self.bad_frames,
            "last_error": self.last_error,
            "bus": self.bus.stats(),
        }
        info.update(self._monitor.sample())

        return info
