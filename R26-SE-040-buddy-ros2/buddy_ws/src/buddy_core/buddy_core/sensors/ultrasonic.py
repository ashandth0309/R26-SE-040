"""HC-SR04 ultrasonic distance sensor support."""

from __future__ import annotations

import time
from collections import deque
from statistics import median


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
