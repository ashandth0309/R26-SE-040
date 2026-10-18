"""Camera back-ends behind one tiny interface.

Only ``MockCamera`` and ``VideoFileSource`` are exercised in the sandbox.
``OpenCvCamera`` and ``Picamera2Source`` are written for the Raspberry Pi
and are UNVERIFIED until run against the real Camera Module 2.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol

import numpy as np


class CameraError(RuntimeError):
    """The camera could not be opened or read."""


class CameraSource(Protocol):
    name: str

    def open(self) -> None: ...

    def read(self) -> np.ndarray | None:
        """A BGR frame, or None when no new frame is ready yet."""

    def close(self) -> None: ...

    @property
    def is_open(self) -> bool: ...


class MockCamera:
    """Deterministic synthetic video: a bright box moving over noise-free
    grey.  Optional scripted failures let tests exercise recovery."""

    name = "mock"

    def __init__(self, width: int = 320, height: int = 240,
                 fail_open: int = 0, fail_read_after: int | None = None,
                 empty_reads_after: int | None = None) -> None:
        self.width, self.height = width, height
        self.fail_open = fail_open
        self.fail_read_after = fail_read_after
        self.empty_reads_after = empty_reads_after
        self.opens = 0
        self.count = 0
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self.opens += 1

        if self.fail_open > 0:
            self.fail_open -= 1
            raise CameraError("mock camera unavailable")

        self._open = True

    def read(self) -> np.ndarray | None:
        if not self._open:
            raise CameraError("mock camera is closed")

        if self.fail_read_after is not None and \
                self.count >= self.fail_read_after:
            self._open = False
            raise CameraError("mock camera disconnected")

        if self.empty_reads_after is not None and \
                self.count >= self.empty_reads_after:
            return None

        img = np.full((self.height, self.width, 3), 90, np.uint8)
        x = (self.count * 7) % max(1, self.width - 40)
        img[60:100, x:x + 40] = (30, 220, 240)
        self.count += 1

        return img

    def close(self) -> None:
        self._open = False


class VideoFileSource:
    """Replay a recorded video (for repeatable tests and benchmarks)."""

    name = "video"

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path, self.loop = path, loop
        self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.path)

        if not cap.isOpened():
            raise CameraError(f"cannot open video {self.path}")

        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise CameraError("video is closed")

        ok, frame = self._cap.read()

        if not ok and self.loop:
            import cv2

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()

        if not ok:
            raise CameraError("end of video")

        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class OpenCvCamera:
    """USB or V4L2 camera through OpenCV.  UNVERIFIED on the robot."""

    name = "opencv"

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: float = 15.0) -> None:
        self.index, self.width, self.height, self.fps = \
            index, width, height, fps
        self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        import cv2

        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") \
            else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)

        if not cap.isOpened():
            raise CameraError(f"cannot open camera index {self.index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise CameraError("camera is closed")

        ok, frame = self._cap.read()

        if not ok:
            raise CameraError("camera read failed")

        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class Picamera2Source:
    """Raspberry Pi Camera Module 2 through Picamera2/libcamera.

    UNVERIFIED.  Picamera2's ``RGB888`` format stores bytes in B,G,R order,
    which is what OpenCV expects, so no channel swap is applied.  If colours
    look swapped on the robot, change ``pixel_format`` in the config.
    """

    name = "picamera2"

    def __init__(self, width: int = 640, height: int = 480,
                 fps: float = 15.0, pixel_format: str = "RGB888",
                 factory: Callable[[], object] | None = None) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.pixel_format = pixel_format
        self._factory = factory
        self._cam = None

    @property
    def is_open(self) -> bool:
        return self._cam is not None

    def open(self) -> None:
        try:
            if self._factory is not None:
                cam = self._factory()
            else:
                from picamera2 import Picamera2

                cam = Picamera2()

            cfg = cam.create_video_configuration(
                main={"size": (self.width, self.height),
                      "format": self.pixel_format},
                controls={"FrameRate": float(self.fps)})
            cam.configure(cfg)
            cam.start()
        except Exception as exc:
            raise CameraError(f"picamera2 failed: {exc}") from exc

        self._cam = cam

    def read(self) -> np.ndarray | None:
        if self._cam is None:
            raise CameraError("camera is closed")

        try:
            return self._cam.capture_array()
        except Exception as exc:
            raise CameraError(f"picamera2 read failed: {exc}") from exc

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass

            self._cam = None


BACKENDS = ("mock", "video", "opencv", "picamera2")


def build_source(backend: str, *, width: int = 640, height: int = 480,
                 fps: float = 15.0, device_index: int = 0,
                 video_path: str = "",
                 pixel_format: str = "RGB888") -> CameraSource:
    """Create a source by name.  Unknown names are an error, not a guess."""

    if backend == "mock":
        return MockCamera(width, height)

    if backend == "video":
        if not video_path:
            raise CameraError("video backend needs video_path")

        return VideoFileSource(video_path)

    if backend == "opencv":
        return OpenCvCamera(device_index, width, height, fps)

    if backend == "picamera2":
        return Picamera2Source(width, height, fps, pixel_format)

    raise CameraError(f"unknown camera backend {backend!r}")
