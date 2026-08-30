"""In-memory neck backend used for simulation and testing."""

from __future__ import annotations

from buddy_core.neck.base import NeckBackend


class MockNeckBackend(NeckBackend):
    """Neck backend that never accesses physical hardware."""

    def __init__(self) -> None:
        self.pan_angle_deg: float | None = None
        self.tilt_angle_deg: float | None = None
        self.released = False
        self.cleaned_up = False

    def set_pan(self, angle_deg: float) -> None:
        self.pan_angle_deg = float(angle_deg)
        self.released = False

    def set_tilt(self, angle_deg: float) -> None:
        self.tilt_angle_deg = float(angle_deg)
        self.released = False

    def release(self) -> None:
        self.released = True

    def cleanup(self) -> None:
        self.release()
        self.cleaned_up = True
