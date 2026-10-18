"""Tests for the camera interface, frame bus and landmark extraction."""

import numpy as np
import pytest

from buddy_navigation.perception.camera import (
    CameraBackend,
    CameraConfig,
    CameraInterface,
    Frame,
    FrameBus,
    MockCameraBackend,
    Picamera2Backend,
    create_backend,
    frame_to_image_fields,
)
from buddy_navigation.perception.landmarks import (
    CameraCalibration,
    LandmarkExtractor,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        self.now += 1.0 / 15.0

        return self.now


def make(**kw):
    clock = Clock()

    return CameraInterface(
        CameraConfig(**kw.pop("config", {})), clock=clock, **kw
    ), clock


def test_defaults_to_the_mock_backend_and_publishes_frames():
    camera, _ = make()

    assert camera.config.backend == "mock"
    assert camera.start()

    frame = camera.grab()

    assert frame.width == 640 and frame.height == 480
    assert frame.data.dtype == np.uint8 and frame.data.shape[2] == 3
    assert frame.sequence == 0
    assert camera.grab().sequence == 1
    assert camera.bus.latest.sequence == 1


def test_resolution_and_fps_are_configurable_and_measured():
    camera, _ = make(config={"width": 320, "height": 240, "fps": 15.0})
    camera.start()

    for _ in range(10):
        frame = camera.grab()

    assert (frame.width, frame.height) == (320, 240)
    assert camera.measured_fps() == pytest.approx(15.0, rel=0.01)
    assert camera.diagnostics()["healthy"]


def test_timestamps_increase():
    camera, _ = make()
    camera.start()
    stamps = [camera.grab().stamp_s for _ in range(5)]

    assert stamps == sorted(stamps) and len(set(stamps)) == 5


def test_grab_before_start_returns_none_and_stop_is_idempotent():
    camera, _ = make()

    assert camera.grab() is None

    camera.start()
    camera.stop()
    camera.stop()

    assert camera.grab() is None
    assert not camera.opened


def test_failure_counts_and_camera_closes_after_repeated_failures():
    backend = MockCameraBackend()
    backend.fail_after = 2
    camera, _ = make(backend=backend, max_consecutive_failures=3)
    camera.start()

    assert camera.grab() and camera.grab()

    for _ in range(3):
        assert camera.grab() is None

    diag = camera.diagnostics()

    assert not camera.opened
    assert diag["failures"] == 3
    assert not diag["healthy"]
    assert "failure" in diag["last_error"]


def test_missing_picamera2_fails_cleanly_without_raising():
    camera = CameraInterface(
        CameraConfig(backend="picamera2"), backend=Picamera2Backend()
    )

    try:
        import picamera2  # noqa: F401
    except ImportError:
        assert not camera.start()
        assert "picamera2" in camera.last_error
    else:  # pragma: no cover - only on a Pi
        camera.stop()


def test_config_validation_and_backend_factory():
    with pytest.raises(ValueError):
        CameraConfig(backend="webcam9000")

    with pytest.raises(ValueError):
        CameraConfig(width=0)

    with pytest.raises(ValueError):
        CameraConfig(fps=-1)

    assert isinstance(create_backend("mock"), CameraBackend)


def test_frame_bus_shared_by_many_consumers_and_isolates_failures():
    bus = FrameBus()
    got_a, got_b = [], []

    def bad(frame):
        raise RuntimeError("consumer bug")

    bus.subscribe(bad)
    token = bus.subscribe(got_a.append)
    bus.subscribe(got_b.append)

    camera = CameraInterface(CameraConfig(), bus=bus)
    camera.start()

    assert camera.grab() is not None
    assert len(got_a) == len(got_b) == 1

    bus.unsubscribe(token)
    camera.grab()

    assert len(got_a) == 1 and len(got_b) == 2


def test_image_message_fields_match_sensor_msgs_image_layout():
    frame = Frame(np.zeros((4, 6, 3), np.uint8), 12.5, 3, "camera_link")
    fields = frame_to_image_fields(frame)

    assert fields["sec"] == 12 and fields["nanosec"] == 500_000_000
    assert (fields["height"], fields["width"]) == (4, 6)
    assert fields["step"] == 18
    assert len(fields["data"]) == 4 * 6 * 3
    assert fields["encoding"] == "bgr8"


def _textured(seed=0, shift=0):
    rng = np.random.default_rng(seed)
    base = (rng.random((240, 320)) * 255).astype(np.uint8)
    base = np.roll(base, shift, axis=1)

    return np.dstack([base] * 3)


def test_landmarks_found_in_texture_and_not_in_a_blank_image():
    extractor = LandmarkExtractor(max_features=100)

    if not extractor.available:
        pytest.skip("OpenCV not installed")

    assert len(extractor.extract(_textured())) > 20
    assert extractor.extract(np.zeros((240, 320, 3), np.uint8)) == []


def test_matching_the_same_scene_beats_matching_different_scenes():
    extractor = LandmarkExtractor(max_features=150)

    if not extractor.available:
        pytest.skip("OpenCV not installed")

    a = extractor.extract(_textured(seed=1))
    same = extractor.extract(_textured(seed=1))
    other = extractor.extract(_textured(seed=2))

    assert extractor.match_count(a, same) > 0.9 * len(a)
    assert extractor.match_count(a, other) < 0.5 * len(a)


def test_calibration_bearing_sign_and_centre():
    cal = CameraCalibration(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    assert cal.bearing_rad(320.0) == pytest.approx(0.0)
    assert cal.bearing_rad(0.0) > 0.0  # left of centre is positive
    assert cal.bearing_rad(640.0) < 0.0
