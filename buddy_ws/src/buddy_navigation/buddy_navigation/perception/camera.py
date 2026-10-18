"""Pi Camera interface with swappable backends.

Backends (all optional imports, chosen by name):

* ``picamera2``: libcamera via Picamera2, preferred on the Raspberry Pi;
* ``opencv``: ``cv2.VideoCapture`` fallback (USB cameras, some CSI setups);
* ``mock``: synthetic frames, the default, safe on any machine.

libcamera normally allows only one process to own the sensor.  Other
consumers (for example the face module) must therefore not open the camera
themselves.  This module exposes a ``FrameBus``: one owner captures, any
number of consumers subscribe.  Whether the face module will use the bus or
the other way round is a team decision recorded in the docs.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Capture settings."""

    backend: str = "mock"
    width: int = 640
    height: int = 480
    fps: float = 15.0
    device_index: int = 0
    frame_id: str = "camera_link"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Resolution must be positive.")

        if self.fps <= 0:
            raise ValueError("fps must be positive.")

        if self.backend not in ("mock", "picamera2", "opencv"):
            raise ValueError(f"Unknown camera backend '{self.backend}'.")


@dataclass(frozen=True)
class Frame:
    """One captured image.  ``data`` is HxWx3 uint8 in BGR order."""

    data: np.ndarray
    stamp_s: float
    sequence: int
    frame_id: str = "camera_link"

    @property
    def width(self) -> int:
        """Image width in pixels."""

        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        """Image height in pixels."""

        return int(self.data.shape[0])


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or read."""


class CameraBackend(ABC):
    """Minimal capture interface."""

    @abstractmethod
    def open(self, config: CameraConfig) -> None:
        """Acquire the camera."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return one BGR frame or raise ``CameraError``."""

    @abstractmethod
    def close(self) -> None:
        """Release the camera.  Must be safe to call twice."""


class MockCameraBackend(CameraBackend):
    """Synthetic frames: a gradient with a moving bright square."""

    def __init__(self) -> None:
        self._open = False
        self._tick = 0
        self.fail_after: int | None = None

    def open(self, config: CameraConfig) -> None:
        self._config = config
        self._open = True
        self._tick = 0

    def read(self) -> np.ndarray:
        if not self._open:
            raise CameraError("Mock camera is not open.")

        if self.fail_after is not None and self._tick >= self.fail_after:
            raise CameraError("Mock camera failure.")

        w, h = self._config.width, self._config.height
        x = np.linspace(0, 255, w, dtype=np.uint8)
        frame = np.tile(x, (h, 1))
        frame = np.dstack([frame, frame // 2, 255 - frame])
        size = max(8, min(w, h) // 6)
        left = (self._tick * 7) % max(1, w - size)
        frame[h // 2:h // 2 + size, left:left + size] = 255
        self._tick += 1

        return frame

    def close(self) -> None:
        self._open = False


class OpenCVBackend(CameraBackend):
    """``cv2.VideoCapture`` backend."""

    def __init__(self) -> None:
        self._capture = None

    def open(self, config: CameraConfig) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("OpenCV is not installed.") from exc

        capture = cv2.VideoCapture(config.device_index)

        if not capture.isOpened():
            raise CameraError(f"Cannot open camera {config.device_index}.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        capture.set(cv2.CAP_PROP_FPS, config.fps)
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Camera is not open.")

        ok, frame = self._capture.read()

        if not ok or frame is None:
            raise CameraError("Camera returned no frame.")

        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class Picamera2Backend(CameraBackend):
    """Picamera2 / libcamera backend (Raspberry Pi only)."""

    def __init__(self) -> None:
        self._camera = None

    def open(self, config: CameraConfig) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise CameraError(
                "picamera2 is not installed (sudo apt install "
                "python3-picamera2)."
            ) from exc

        try:
            camera = Picamera2()
            # "RGB888" is delivered as BGR-ordered bytes by libcamera.
            camera.configure(
                camera.create_video_configuration(
                    main={
                        "size": (config.width, config.height),
                        "format": "RGB888",
                    },
                    controls={"FrameRate": config.fps},
                )
            )
            camera.start()
        except Exception as exc:
            raise CameraError(f"Picamera2 failed to start: {exc}") from exc

        self._camera = camera

    def read(self) -> np.ndarray:
        if self._camera is None:
            raise CameraError("Camera is not open.")

        try:
            return self._camera.capture_array("main")
        except Exception as exc:
            raise CameraError(f"Picamera2 capture failed: {exc}") from exc

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            finally:
                self._camera = None


def create_backend(name: str) -> CameraBackend:
    """Return the backend for a configured name."""

    return {
        "mock": MockCameraBackend,
        "opencv": OpenCVBackend,
        "picamera2": Picamera2Backend,
    }[name]()


class FrameBus:
    """Fan-out of frames to any number of consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[int, Callable[[Frame], None]] = {}
        self._next_id = 0
        self.latest: Frame | None = None

    def subscribe(self, callback: Callable[[Frame], None]) -> int:
        """Register a consumer and return a token for ``unsubscribe``."""

        with self._lock:
            token = self._next_id
            self._next_id += 1
            self._subscribers[token] = callback

        return token

    def unsubscribe(self, token: int) -> None:
        """Remove a consumer."""

        with self._lock:
            self._subscribers.pop(token, None)

    def publish(self, frame: Frame) -> list[Exception]:
        """Deliver a frame; one failing consumer never affects the others."""

        with self._lock:
            self.latest = frame
            callbacks = list(self._subscribers.values())

        errors: list[Exception] = []

        for callback in callbacks:
            try:
                callback(frame)
            except Exception as exc:  # consumer bugs must not kill capture
                errors.append(exc)

        return errors


class CameraInterface:
    """Owns one camera, tracks health and publishes onto a ``FrameBus``."""

    def __init__(
        self,
        config: CameraConfig | None = None,
        backend: CameraBackend | None = None,
        bus: FrameBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_consecutive_failures: int = 5,
    ) -> None:
        self.config = config or CameraConfig()
        self.backend = backend or create_backend(self.config.backend)
        self.bus = bus or FrameBus()
        self._clock = clock
        self._max_failures = max_consecutive_failures
        self.opened = False
        self.sequence = 0
        self.consecutive_failures = 0
        self.total_failures = 0
        self.last_error = ""
        self._stamps: deque[float] = deque(maxlen=30)

    def start(self) -> bool:
        """Open the camera.  Returns False (never raises) on failure."""

        if self.opened:
            return True

        try:
            self.backend.open(self.config)
        except CameraError as exc:
            self.last_error = str(exc)

            return False

        self.opened = True
        self.consecutive_failures = 0

        return True

    def stop(self) -> None:
        """Release the camera.  Safe to call repeatedly."""

        try:
            self.backend.close()
        finally:
            self.opened = False

    def grab(self) -> Frame | None:
        """Capture one frame and publish it; None if capture failed."""

        if not self.opened:
            return None

        try:
            data = self.backend.read()
        except CameraError as exc:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_error = str(exc)

            if self.consecutive_failures >= self._max_failures:
                self.stop()

            return None

        self.consecutive_failures = 0
        now = self._clock()
        frame = Frame(data, now, self.sequence, self.config.frame_id)
        self.sequence += 1
        self._stamps.append(now)
        self.bus.publish(frame)

        return frame

    def measured_fps(self) -> float:
        """Frame rate over the recent window (0 if unknown)."""

        if len(self._stamps) < 2:
            return 0.0

        span = self._stamps[-1] - self._stamps[0]

        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def diagnostics(self) -> dict[str, str | float | int | bool]:
        """Health summary for ``/diagnostics``."""

        fps = self.measured_fps()

        return {
            "backend": self.config.backend,
            "opened": self.opened,
            "frames": self.sequence,
            "measured_fps": round(fps, 2),
            "target_fps": self.config.fps,
            "failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "healthy": self.opened
            and self.consecutive_failures == 0
            and (self.sequence < 5 or fps > 0.5 * self.config.fps),
        }


def frame_to_image_fields(frame: Frame) -> dict:
    """Fields for ``sensor_msgs/Image`` without needing cv_bridge."""

    data = np.ascontiguousarray(frame.data, dtype=np.uint8)

    return {
        "sec": int(math.floor(frame.stamp_s)),
        "nanosec": int((frame.stamp_s % 1.0) * 1e9),
        "frame_id": frame.frame_id,
        "height": frame.height,
        "width": frame.width,
        "encoding": "bgr8",
        "is_bigendian": 0,
        "step": frame.width * 3,
        "data": data.tobytes(),
    }
