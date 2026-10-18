"""HC-SR04 ultrasonic distance sensor support."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median


class UltrasonicStatus(str, Enum):
    """Outcome of one status-aware ultrasonic measurement.

    The legacy ``read_distance_m`` API collapses every failure into
    ``None``.  These values keep the failure modes apart because they have
    different safety meaning:

    * ``OK``: a distance inside the configured range was measured.
    * ``CLEAR``: the module started a measurement but no echo returned
      within the configured range.  Nothing was detected, but a soft or
      steeply angled obstacle can also produce this result.
    * ``TOO_CLOSE``: the echo pulse was shorter than the minimum range.
      An obstacle is present and closer than the sensor can resolve.
    * ``NO_RESPONSE``: the echo line never went high after the trigger
      pulse.  Wiring, power or module fault (or a module variant that
      does not raise echo when nothing is detected).
    * ``STUCK_HIGH``: the echo line was already high before triggering and
      did not settle.  Wiring or module fault.
    * ``ERROR``: the GPIO backend raised an exception.  Only produced by
      ``UltrasonicArrayScanner``, never by ``measure`` itself.
    """

    OK = "ok"
    CLEAR = "clear"
    TOO_CLOSE = "too_close"
    NO_RESPONSE = "no_response"
    STUCK_HIGH = "stuck_high"
    ERROR = "error"


@dataclass(frozen=True)
class UltrasonicMeasurement:
    """Result of one ``UltrasonicSensor.measure`` call."""

    status: UltrasonicStatus
    raw_distance_m: float | None
    filtered_distance_m: float | None
    pulse_seconds: float | None
    started_monotonic: float
    finished_monotonic: float

    @property
    def has_distance(self) -> bool:
        """Return whether a distance was actually measured."""

        return self.status is UltrasonicStatus.OK


class UltrasonicSensor:
    """Read and filter one HC-SR04 ultrasonic distance sensor."""

    SPEED_OF_SOUND_MPS = 343.0

    def __init__(
        self,
        gpio,
        trigger_pin: int,
        echo_pin: int,
        *,
        timeout_seconds: float = 0.03,
        minimum_distance_m: float = 0.02,
        maximum_distance_m: float = 4.0,
        filter_window: int = 3,
    ) -> None:
        if trigger_pin == echo_pin:
            raise ValueError(
                "Ultrasonic trigger and echo pins must be different."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if minimum_distance_m <= 0:
            raise ValueError(
                "minimum_distance_m must be greater than zero."
            )

        if maximum_distance_m <= minimum_distance_m:
            raise ValueError(
                "maximum_distance_m must be greater than "
                "minimum_distance_m."
            )

        if filter_window <= 0:
            raise ValueError(
                "filter_window must be greater than zero."
            )

        self._gpio = gpio
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.timeout_seconds = timeout_seconds
        self.minimum_distance_m = minimum_distance_m
        self.maximum_distance_m = maximum_distance_m
        self.filter_window = filter_window

        self._samples: deque[float] = deque(
            maxlen=filter_window
        )

        self._gpio.configure_output(
            self.trigger_pin,
            initial=False,
        )
        self._gpio.configure_input(
            self.echo_pin,
        )

    def _wait_for_state(
        self,
        expected_state: bool,
    ) -> float | None:
        """Wait for the echo GPIO to reach the requested state."""

        deadline = time.monotonic() + self.timeout_seconds

        while bool(self._gpio.read(self.echo_pin)) != expected_state:
            if time.monotonic() >= deadline:
                return None

        return time.monotonic()

    def read_raw_distance_m(self) -> float | None:
        """Take one physical measurement and return metres."""

        self._gpio.write(self.trigger_pin, False)
        time.sleep(0.000002)

        self._gpio.write(self.trigger_pin, True)
        time.sleep(0.00001)
        self._gpio.write(self.trigger_pin, False)

        echo_start = self._wait_for_state(True)

        if echo_start is None:
            return None

        echo_end = self._wait_for_state(False)

        if echo_end is None:
            return None

        pulse_seconds = echo_end - echo_start

        distance_m = (
            pulse_seconds * self.SPEED_OF_SOUND_MPS / 2.0
        )

        if not (
            self.minimum_distance_m
            <= distance_m
            <= self.maximum_distance_m
        ):
            return None

        return distance_m

    def read_distance_m(self) -> float | None:
        """
        Take one measurement and return the median filtered distance.

        Invalid measurements are not added to the filter.
        """

        distance = self.read_raw_distance_m()

        if distance is None:
            return None

        self._samples.append(distance)

        return float(median(self._samples))

    def measure(
        self,
        *,
        idle_settle_seconds: float = 0.045,
    ) -> UltrasonicMeasurement:
        """Take one measurement and report why it succeeded or failed.

        Unlike ``read_distance_m`` this does not discard the failure
        reason.  Only ``OK`` readings enter the median filter, exactly as
        in ``read_distance_m``, so ``filtered_distance_m`` is the same
        value the legacy API would have returned.

        Worst-case duration is bounded by
        ``idle_settle_seconds + 2 * timeout_seconds``.
        """

        started = time.monotonic()

        # A healthy module holds echo LOW between measurements.  A line
        # that stays HIGH would otherwise look like an endless echo and be
        # misread as "nothing in range".
        if bool(self._gpio.read(self.echo_pin)):
            settle_deadline = started + idle_settle_seconds

            while bool(self._gpio.read(self.echo_pin)):
                if time.monotonic() >= settle_deadline:
                    return UltrasonicMeasurement(
                        UltrasonicStatus.STUCK_HIGH,
                        None,
                        None,
                        None,
                        started,
                        time.monotonic(),
                    )

        self._gpio.write(self.trigger_pin, False)
        time.sleep(0.000002)

        self._gpio.write(self.trigger_pin, True)
        time.sleep(0.00001)
        self._gpio.write(self.trigger_pin, False)

        echo_start = self._wait_for_state(True)

        if echo_start is None:
            return UltrasonicMeasurement(
                UltrasonicStatus.NO_RESPONSE,
                None,
                None,
                None,
                started,
                time.monotonic(),
            )

        echo_end = self._wait_for_state(False)
        finished = time.monotonic()

        if echo_end is None:
            return UltrasonicMeasurement(
                UltrasonicStatus.CLEAR,
                None,
                None,
                None,
                started,
                finished,
            )

        pulse_seconds = echo_end - echo_start
        distance_m = pulse_seconds * self.SPEED_OF_SOUND_MPS / 2.0

        if distance_m < self.minimum_distance_m:
            return UltrasonicMeasurement(
                UltrasonicStatus.TOO_CLOSE,
                distance_m,
                None,
                pulse_seconds,
                started,
                finished,
            )

        if distance_m > self.maximum_distance_m:
            return UltrasonicMeasurement(
                UltrasonicStatus.CLEAR,
                None,
                None,
                pulse_seconds,
                started,
                finished,
            )

        self._samples.append(distance_m)

        return UltrasonicMeasurement(
            UltrasonicStatus.OK,
            distance_m,
            float(median(self._samples)),
            pulse_seconds,
            started,
            finished,
        )

    @property
    def has_samples(self) -> bool:
        """Return whether the filter currently contains valid samples."""

        return bool(self._samples)

    @property
    def last_filtered_distance_m(self) -> float | None:
        """Return the current median without taking a new measurement."""

        if not self._samples:
            return None

        return float(median(self._samples))

    def clear_filter(self) -> None:
        """Remove all previous distance samples."""

        self._samples.clear()

    def shutdown(self) -> None:
        """Leave the trigger output in its safe LOW state."""

        self._gpio.write(self.trigger_pin, False)
