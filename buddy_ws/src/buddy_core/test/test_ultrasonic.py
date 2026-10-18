"""Tests for status-aware ultrasonic measurement and sequential scanning.

A virtual clock and a fake GPIO that schedules echo pulses after the
trigger edge make the timing deterministic, so no test depends on the real
clock or on any hardware.
"""

import threading

import pytest

from buddy_core.sensors import ultrasonic as ultrasonic_module
from buddy_core.sensors.ultrasonic import (
    UltrasonicSensor,
    UltrasonicStatus,
)
from buddy_core.sensors.ultrasonic_array import UltrasonicArrayScanner
from ultrasonic_fakes import (
    WIRING,
    FakeClock,
    FakeEchoGPIO,
    FakeTimeModule,
)

SPEED = UltrasonicSensor.SPEED_OF_SOUND_MPS


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(
        ultrasonic_module,
        "time",
        FakeTimeModule(fake),
    )
    return fake


@pytest.fixture
def gpio(clock):
    return FakeEchoGPIO(clock, WIRING)


def make_sensor(gpio, name="front_left", **kwargs):
    trigger, echo = WIRING[name]
    return UltrasonicSensor(gpio, trigger, echo, **kwargs)


def make_array(gpio, **kwargs):
    return {
        name: make_sensor(gpio, name, **kwargs)
        for name in WIRING
    }


# ---------------------------------------------------------------------------
# Single sensor
# ---------------------------------------------------------------------------


def test_valid_distance_is_measured(gpio):
    gpio.set_scenario(22, "distance", 1.0)

    result = make_sensor(gpio).measure()

    assert result.status is UltrasonicStatus.OK
    assert result.has_distance
    assert result.raw_distance_m == pytest.approx(1.0, abs=0.02)
    assert result.filtered_distance_m == pytest.approx(1.0, abs=0.02)


def test_trigger_pulse_is_at_least_ten_microseconds(gpio):
    gpio.set_scenario(22, "distance", 0.5)

    make_sensor(gpio).measure()

    _, rose, fell = gpio.trigger_pulses[0]
    assert fell - rose >= 10e-6


def test_only_valid_samples_enter_the_median_filter(gpio):
    sensor = make_sensor(gpio)

    for distance in (1.0, 1.0, 2.0):
        gpio.set_scenario(22, "distance", distance)
        last = sensor.measure()

    assert last.filtered_distance_m == pytest.approx(1.0, abs=0.02)

    gpio.set_scenario(22, "none")
    failed = sensor.measure()

    assert failed.status is UltrasonicStatus.NO_RESPONSE
    assert failed.filtered_distance_m is None
    assert sensor.last_filtered_distance_m == pytest.approx(1.0, abs=0.02)


def test_no_response_when_echo_never_rises(gpio):
    gpio.set_scenario(22, "none")

    result = make_sensor(gpio).measure()

    assert result.status is UltrasonicStatus.NO_RESPONSE
    assert result.raw_distance_m is None


def test_clear_when_echo_never_falls_within_timeout(gpio):
    gpio.set_scenario(22, "clear")

    result = make_sensor(gpio).measure()

    assert result.status is UltrasonicStatus.CLEAR
    assert result.raw_distance_m is None


def test_return_beyond_maximum_range_is_clear(gpio):
    gpio.set_scenario(22, "distance", 4.5)

    result = make_sensor(gpio, maximum_distance_m=4.0).measure()

    assert result.status is UltrasonicStatus.CLEAR
    assert result.raw_distance_m is None
    assert result.pulse_seconds is not None


def test_return_closer_than_minimum_is_too_close(gpio):
    gpio.set_scenario(22, "distance", 0.004)

    result = make_sensor(gpio, minimum_distance_m=0.02).measure()

    assert result.status is UltrasonicStatus.TOO_CLOSE
    assert result.filtered_distance_m is None


def test_stuck_high_echo_is_reported_not_treated_as_clear(gpio):
    gpio.set_scenario(22, "stuck")

    result = make_sensor(gpio).measure()

    assert result.status is UltrasonicStatus.STUCK_HIGH


@pytest.mark.parametrize(
    "kind, distance",
    [
        ("none", None),
        ("clear", None),
        ("stuck", None),
        ("distance", 0.004),
        ("distance", 4.5),
        ("distance", 1.0),
    ],
)
def test_every_outcome_finishes_within_a_bounded_time(
    clock,
    gpio,
    kind,
    distance,
):
    gpio.set_scenario(22, kind, distance)
    sensor = make_sensor(gpio, timeout_seconds=0.03)

    before = clock.now
    sensor.measure(idle_settle_seconds=0.045)

    assert clock.now - before <= 0.045 + 2 * 0.03 + 0.005


def test_back_to_back_measurements_do_not_misread_late_echo_as_stuck(gpio):
    gpio.set_scenario(22, "clear")
    sensor = make_sensor(gpio)

    first = sensor.measure()
    second = sensor.measure()

    assert first.status is UltrasonicStatus.CLEAR
    assert second.status is UltrasonicStatus.CLEAR


def test_legacy_read_distance_is_unchanged(gpio):
    gpio.set_scenario(22, "distance", 1.5)
    sensor = make_sensor(gpio)

    assert sensor.read_distance_m() == pytest.approx(1.5, abs=0.02)

    gpio.set_scenario(22, "none")

    assert sensor.read_distance_m() is None


# ---------------------------------------------------------------------------
# Array scanner
# ---------------------------------------------------------------------------


def make_scanner(clock, gpio, interval=0.06, **kwargs):
    return UltrasonicArrayScanner(
        make_array(gpio),
        min_trigger_interval_seconds=interval,
        clock=clock.monotonic,
        sleep=clock.sleep,
        **kwargs,
    )


def test_scan_measures_every_sensor_once_in_order(clock, gpio):
    for echo, distance in ((22, 0.5), (24, 0.8), (26, 1.1), (14, 1.4)):
        gpio.set_scenario(echo, "distance", distance)

    scanner = make_scanner(clock, gpio)
    results = scanner.scan()

    assert list(results) == [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ]
    assert results["front_left"].raw_distance_m == pytest.approx(
        0.5, abs=0.02
    )
    assert results["rear_right"].raw_distance_m == pytest.approx(
        1.4, abs=0.02
    )
    assert scanner.scan_index == 1


def test_sensors_never_fire_at_the_same_time(clock, gpio):
    for echo in (22, 24, 26, 14):
        gpio.set_scenario(echo, "distance", 1.0)

    scanner = make_scanner(clock, gpio, interval=0.06)

    for _ in range(3):
        scanner.scan()

    history = scanner.trigger_history

    assert len(history) == 12

    for earlier, later in zip(history, history[1:]):
        assert later.started >= earlier.finished
        assert later.started - earlier.finished >= 0.06 - 1e-9


def test_step_refuses_to_fire_before_the_interval_has_elapsed(
    clock,
    gpio,
):
    gpio.set_scenario(22, "distance", 1.0)
    gpio.set_scenario(24, "distance", 1.0)
    scanner = make_scanner(clock, gpio, interval=0.06)

    first = scanner.step()
    too_soon = scanner.step()

    assert first is not None
    assert first.name == "front_left"
    assert too_soon is None
    assert scanner.seconds_until_ready() > 0.0

    clock.now += 0.07
    second = scanner.step()

    assert second is not None
    assert second.name == "front_right"


def test_step_reports_scan_completion_and_index(clock, gpio):
    scanner = make_scanner(clock, gpio)

    flags = [
        scanner.step(wait=True).scan_complete
        for _ in range(8)
    ]

    assert flags == [False, False, False, True] * 2
    assert scanner.scan_index == 2


def test_failed_sensor_does_not_block_the_others(clock, gpio):
    gpio.set_scenario(22, "distance", 1.0)
    gpio.set_scenario(24, "none")
    gpio.set_scenario(26, "stuck")
    gpio.set_scenario(14, "distance", 2.0)

    scanner = make_scanner(clock, gpio)

    before = clock.now
    results = scanner.scan()
    elapsed = clock.now - before

    assert results["front_left"].status is UltrasonicStatus.OK
    assert results["front_right"].status is UltrasonicStatus.NO_RESPONSE
    assert results["rear_left"].status is UltrasonicStatus.STUCK_HIGH
    assert results["rear_right"].status is UltrasonicStatus.OK

    worst_case = 4 * (0.045 + 2 * 0.03 + 0.06) + 0.05
    assert elapsed <= worst_case


def test_gpio_exception_becomes_error_status_not_a_crash(clock, gpio):
    gpio.fail_pins.add(22)
    scanner = make_scanner(clock, gpio)

    result = scanner.step()

    assert result.measurement.status is UltrasonicStatus.ERROR


def test_concurrent_callers_are_serialised(clock, gpio):
    for echo in (22, 24, 26, 14):
        gpio.set_scenario(echo, "distance", 1.0)

    gpio.yield_on_read = True
    scanner = make_scanner(clock, gpio, interval=0.06)
    errors = []

    def worker():
        try:
            for _ in range(3):
                scanner.step(wait=True)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    history = scanner.trigger_history
    assert len(history) == 12

    for earlier, later in zip(history, history[1:]):
        assert later.started >= earlier.finished


def test_scanner_rejects_bad_configuration(clock, gpio):
    sensors = make_array(gpio)

    with pytest.raises(ValueError):
        UltrasonicArrayScanner({})

    with pytest.raises(ValueError):
        UltrasonicArrayScanner(sensors, min_trigger_interval_seconds=-1)

    with pytest.raises(ValueError):
        UltrasonicArrayScanner(
            sensors,
            order=["front_left", "front_left", "rear_left", "rear_right"],
        )

    with pytest.raises(ValueError):
        UltrasonicArrayScanner(sensors, order=["front_left"])
