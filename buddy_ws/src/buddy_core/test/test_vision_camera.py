"""Shared camera layer: frames, bus, sources, service, recovery, privacy."""

import threading
import time

import cv2
import numpy as np
import pytest

from buddy_core.vision.camera_service import (
    CameraConfig,
    CameraLock,
    CameraService,
    CameraState,
)
from buddy_core.vision.camera_sources import (
    CameraError,
    MockCamera,
    Picamera2Source,
    VideoFileSource,
    build_source,
)
from buddy_core.vision.frames import FrameBus, make_frame
from buddy_core.vision.resources import ResourceMonitor


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "cam.lock")


def service(lock_path, source=None, clock=None, **cfg):
    clock = clock or Clock()
    config = CameraConfig(lock_path=lock_path, **cfg)
    factory = (lambda: source) if source is not None else None

    return CameraService(config, source_factory=factory, clock=clock,
                         wall=lambda: 1000.0 + clock.t), clock


# ---------- frames and bus ----------

def img(v=10):
    return np.full((4, 6, 3), v, np.uint8)


def test_frame_is_validated_and_read_only():
    f = make_frame(img(), 1, 1.0, 1.0)
    assert (f.width, f.height) == (6, 4)
    with pytest.raises(ValueError):
        f.image[0, 0, 0] = 1
    for bad in (np.zeros((4, 6), np.uint8), np.zeros((4, 6, 3), np.float32),
                np.zeros((1, 1, 3), np.uint8), "frame", None):
        with pytest.raises(ValueError):
            make_frame(bad, 1, 1.0, 1.0)


def test_every_subscriber_gets_the_frame_and_slow_one_drops_old():
    bus = FrameBus()
    fast, slow = bus.subscribe("fast"), bus.subscribe("slow", maxlen=2)

    for i in range(5):
        bus.publish(make_frame(img(i), i + 1, 0.0, 0.0))
        assert fast.get().seq == i + 1

    assert slow.received == 5 and slow.dropped == 3
    assert [slow.get().seq, slow.get().seq] == [4, 5]
    assert slow.get() is None
    assert fast.dropped == 0


def test_latest_skips_stale_frames_and_counts_them():
    bus = FrameBus()
    sub = bus.subscribe("x", maxlen=5)

    for i in range(4):
        bus.publish(make_frame(img(), i + 1, 0.0, 0.0))

    assert sub.latest().seq == 4
    assert sub.dropped == 3 and sub.latest() is None


def test_get_blocks_until_a_frame_arrives():
    bus = FrameBus()
    sub = bus.subscribe("x")
    threading.Timer(0.05, lambda: bus.publish(
        make_frame(img(), 7, 0.0, 0.0))).start()
    assert sub.get(timeout=2.0).seq == 7


def test_unsubscribed_consumer_stops_receiving():
    bus = FrameBus()
    sub = bus.subscribe("x")
    bus.unsubscribe(sub)
    bus.publish(make_frame(img(), 1, 0.0, 0.0))
    assert sub.get() is None and bus.stats()["subscribers"] == {}


# ---------- sources ----------

def test_mock_camera_produces_moving_deterministic_frames():
    cam = MockCamera(160, 120)
    cam.open()
    a, b = cam.read(), cam.read()
    assert a.shape == (120, 160, 3) and not np.array_equal(a, b)
    again = MockCamera(160, 120)
    again.open()
    assert np.array_equal(a, again.read())


def test_video_file_source_replays_and_loops(tmp_path):
    path = str(tmp_path / "t.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10,
                             (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.release()
    src = VideoFileSource(path)
    src.open()
    frames = [src.read() for _ in range(8)]
    assert all(f.shape == (48, 64, 3) for f in frames)
    src.close()
    with pytest.raises(CameraError):
        VideoFileSource(str(tmp_path / "missing.avi")).open()


def test_build_source_rejects_unknown_or_incomplete_backends():
    assert build_source("mock").name == "mock"
    with pytest.raises(CameraError):
        build_source("webcam9000")
    with pytest.raises(CameraError):
        build_source("video")


def test_picamera2_adapter_with_a_fake_driver():
    calls = []

    class Fake:
        def create_video_configuration(self, **kw):
            calls.append(("cfg", kw))
            return kw

        def configure(self, c):
            calls.append("configure")

        def start(self):
            calls.append("start")

        def capture_array(self):
            return np.zeros((48, 64, 3), np.uint8)

        def stop(self):
            calls.append("stop")

        def close(self):
            calls.append("close")

    src = Picamera2Source(64, 48, 10, factory=Fake)
    src.open()
    assert src.read().shape == (48, 64, 3)
    assert calls[0][1]["main"]["format"] == "RGB888"
    src.close()
    assert not src.is_open and calls[-2:] == ["stop", "close"]

    def broken():
        raise RuntimeError("no camera")

    with pytest.raises(CameraError):
        Picamera2Source(factory=broken).open()


# ---------- service ----------

def test_streams_frames_with_increasing_seq_and_timestamps(lock_path):
    svc, clock = service(lock_path, MockCamera())
    sub = svc.bus.subscribe("t", maxlen=10)
    for _ in range(3):
        assert svc.step()
        clock.advance(0.1)
    frames = [sub.get(), sub.get(), sub.get()]
    assert [f.seq for f in frames] == [1, 2, 3]
    assert frames[0].stamp < frames[2].stamp
    assert svc.state is CameraState.STREAMING and svc.indicator
    assert svc.measured_fps() == pytest.approx(10.0)
    svc.stop()


def test_disconnect_recovers_with_backoff(lock_path):
    cams = [MockCamera(fail_read_after=2), MockCamera()]
    it = iter(cams)
    clock = Clock()
    svc = CameraService(CameraConfig(lock_path=lock_path),
                        source_factory=lambda: next(it), clock=clock)
    assert svc.step() and svc.step()
    assert not svc.step()
    assert svc.state is CameraState.RECONNECTING
    assert not svc.indicator and "disconnected" in svc.last_error
    assert not svc.step()  # still inside the back-off window
    clock.advance(0.6)
    assert svc.step() and svc.state is CameraState.STREAMING
    assert svc.reconnects == 1
    svc.stop()


def test_open_failures_back_off_exponentially_and_cap(lock_path):
    cam = MockCamera(fail_open=99)
    svc, clock = service(lock_path, cam, reconnect_initial_s=1.0,
                         reconnect_max_s=4.0)
    waits = []
    for _ in range(5):
        before = clock.t
        svc.step()
        waits.append(svc._next_attempt - before)
        clock.t = svc._next_attempt
    assert waits == [1.0, 2.0, 4.0, 4.0, 4.0]
    assert svc.open_failures == 5 and not svc.indicator


def test_stalled_stream_is_reopened(lock_path):
    cams = [MockCamera(empty_reads_after=1), MockCamera()]
    it = iter(cams)
    clock = Clock()
    svc = CameraService(CameraConfig(lock_path=lock_path, stale_after_s=1.0),
                        source_factory=lambda: next(it), clock=clock)
    assert svc.step()
    clock.advance(0.5)
    assert not svc.step() and svc.state is CameraState.STREAMING
    clock.advance(1.0)
    assert not svc.step()
    assert svc.state is CameraState.RECONNECTING
    assert "stalled" in svc.last_error
    clock.advance(1.0)
    assert svc.step()


def test_staleness_gate_for_consumers(lock_path):
    svc, clock = service(lock_path, MockCamera(), stale_after_s=2.0)
    assert not svc.is_fresh()
    svc.step()
    assert svc.is_fresh()
    clock.advance(3.0)
    assert not svc.is_fresh()
    assert svc.is_fresh(max_age_s=5.0)


def test_second_owner_is_refused(lock_path):
    a, _ = service(lock_path, MockCamera())
    b, _ = service(lock_path, MockCamera())
    assert a.step()
    assert not b.step()
    assert "another process" in b.last_error
    a.stop()
    b._next_attempt = 0.0
    assert b.step()


def test_lock_is_released_on_close_and_reacquirable(lock_path):
    lock = CameraLock(lock_path)
    other = CameraLock(lock_path)
    assert lock.acquire() and lock.held
    assert not other.acquire()
    lock.release()
    assert other.acquire()
    other.release()


def test_privacy_disable_closes_device_and_stops_frames(lock_path):
    cam = MockCamera()
    svc, clock = service(lock_path, cam)
    sub = svc.bus.subscribe("t", maxlen=10)
    assert svc.step() and svc.indicator
    svc.set_enabled(False)
    assert not cam.is_open and svc.state is CameraState.DISABLED
    assert not svc.indicator and not svc.is_fresh()
    published = svc.bus.published
    for _ in range(5):
        clock.advance(0.1)
        assert not svc.step()
    assert svc.bus.published == published and sub.received == 1
    # The lock is free, so no stale "owner" remains while disabled.
    probe = CameraLock(lock_path)
    assert probe.acquire()
    probe.release()
    svc.set_enabled(True)
    assert svc.step() and svc.indicator


def test_disabled_at_start_never_opens_the_camera(lock_path):
    cam = MockCamera()
    svc, _ = service(lock_path, cam, enabled=False)
    assert not svc.step() and cam.opens == 0
    assert svc.state is CameraState.DISABLED


def test_invalid_frames_are_counted_not_published(lock_path):
    class Bad(MockCamera):
        def read(self):
            return np.zeros((10, 10), np.uint8)

    svc, _ = service(lock_path, Bad())
    assert not svc.step()
    assert svc.bad_frames == 1 and svc.bus.published == 0


def test_unexpected_backend_exception_triggers_reconnect(lock_path):
    class Boom(MockCamera):
        def read(self):
            raise ValueError("driver crashed")

    svc, _ = service(lock_path, Boom())
    assert not svc.step()
    assert svc.state is CameraState.RECONNECTING
    assert "driver crashed" in svc.last_error


def test_background_thread_streams_and_stops_cleanly(lock_path):
    svc = CameraService(CameraConfig(lock_path=lock_path, fps=50.0),
                        source_factory=MockCamera)
    sub = svc.bus.subscribe("t", maxlen=3)
    svc.start()
    first = sub.get(timeout=2.0)
    assert first is not None
    time.sleep(0.2)
    svc.stop()
    assert svc.state is CameraState.STOPPED and not svc.indicator
    assert svc.seq >= 3
    assert threading.active_count() >= 1


def test_overload_does_not_block_the_camera(lock_path):
    svc, clock = service(lock_path, MockCamera())
    never_reads = svc.bus.subscribe("slow", maxlen=1)
    for _ in range(50):
        assert svc.step()
        clock.advance(0.02)
    assert never_reads.dropped == 49 and svc.seq == 50


def test_health_reports_only_measured_values(lock_path):
    svc, clock = service(lock_path, MockCamera())
    h0 = svc.health()
    assert h0["measured_fps"] is None and h0["frame_age_s"] is None
    svc.step()
    clock.advance(0.1)
    h = svc.health()
    for key in ("state", "indicator", "frames", "reconnects", "bus",
                "cpu_percent_of_one_core", "rss_mb", "temperature_c"):
        assert key in h
    assert h["frames"] == 1 and h["frame_age_s"] == pytest.approx(0.1)


def test_resource_monitor_never_invents_numbers(tmp_path, monkeypatch):
    from buddy_core.vision import resources

    assert resources.read_temperature_c(str(tmp_path / "none")) is None
    assert resources.read_rss_mb(str(tmp_path / "none")) is None
    assert resources.read_available_mem_mb(str(tmp_path / "none")) is None
    temp = tmp_path / "temp"
    temp.write_text("51500\n")
    assert resources.read_temperature_c(str(temp)) == 51.5
    clock = Clock()
    mon = ResourceMonitor(clock)
    assert mon.sample()["cpu_percent_of_one_core"] is None  # zero span
