"""Strictly sequential, bounded-time scanning of several ultrasonic sensors.

Ultrasonic modules hear each other's bursts.  Triggering two modules at the
same time, or triggering the next one before the previous echo has died
away, produces false short ranges.  This scanner is the single place that
decides when a sensor may fire:

* only one sensor is ever measured at a time (enforced by a lock, so even
  concurrent callers are serialised);
* consecutive triggers are separated by at least
  ``min_trigger_interval_seconds``;
* every ``step`` has a worst-case duration set by the sensor timeouts, so
  a missing or broken sensor can never block the caller indefinitely.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from buddy_core.sensors.ultrasonic import (
    UltrasonicMeasurement,
    UltrasonicSensor,
    UltrasonicStatus,
)


@dataclass(frozen=True)
class SensorScanResult:
    """One sensor's measurement within the round-robin scan."""

    name: str
    measurement: UltrasonicMeasurement
    scan_index: int
    scan_complete: bool


@dataclass(frozen=True)
class TriggerRecord:
    """When a sensor was fired, for interference diagnostics and tests."""

    name: str
    started: float
    finished: float


class UltrasonicArrayScanner:
    """Round-robin scheduler for a set of ``UltrasonicSensor`` objects."""

    def __init__(
        self,
        sensors: Mapping[str, UltrasonicSensor],
        *,
        order: Iterable[str] | None = None,
        min_trigger_interval_seconds: float = 0.06,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        history_length: int = 64,
    ) -> None:
        if not sensors:
            raise ValueError("At least one ultrasonic sensor is required.")

        if min_trigger_interval_seconds < 0:
            raise ValueError(
                "min_trigger_interval_seconds must not be negative."
            )

        names = tuple(order) if order is not None else tuple(sensors)

        if len(set(names)) != len(names):
            raise ValueError("Scan order contains a duplicate sensor.")

        if set(names) != set(sensors):
            raise ValueError(
                "Scan order must list exactly the configured sensors."
            )

        self._sensors = dict(sensors)
        self._names = names
        self._interval = float(min_trigger_interval_seconds)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_index = 0
        self._scan_index = 0
        self._last_finished: float | None = None
        self._history: deque[TriggerRecord] = deque(maxlen=history_length)

    @property
    def names(self) -> tuple[str, ...]:
        """Sensor names in firing order."""

        return self._names

    @property
    def scan_index(self) -> int:
        """Number of completed full scans."""

        return self._scan_index

    @property
    def trigger_history(self) -> tuple[TriggerRecord, ...]:
        """Most recent trigger windows, oldest first."""

        with self._lock:
            return tuple(self._history)

    def seconds_until_ready(self) -> float:
        """Return how long until the next sensor may be triggered."""

        if self._last_finished is None:
            return 0.0

        remaining = self._last_finished + self._interval - self._clock()

        return max(0.0, remaining)

    def step(self, *, wait: bool = False) -> SensorScanResult | None:
        """Measure the next sensor in the round-robin order.

        With ``wait=False`` this returns ``None`` immediately when the
        minimum trigger interval has not yet elapsed, so a ROS timer can
        call it at any rate without ever breaking the spacing rule.  With
        ``wait=True`` it sleeps for the remaining interval instead.
        """

        with self._lock:
            remaining = self.seconds_until_ready()

            if remaining > 0.0:
                if not wait:
                    return None

                self._sleep(remaining)

            name = self._names[self._next_index]
            sensor = self._sensors[name]

            started = self._clock()

            try:
                measurement = sensor.measure()
            except Exception:
                finished = self._clock()
                measurement = UltrasonicMeasurement(
                    UltrasonicStatus.ERROR,
                    None,
                    None,
                    None,
                    started,
                    finished,
                )

            finished = self._clock()
            self._last_finished = finished
            self._history.append(TriggerRecord(name, started, finished))

            self._next_index += 1
            scan_complete = self._next_index >= len(self._names)

            if scan_complete:
                self._next_index = 0
                self._scan_index += 1

            return SensorScanResult(
                name=name,
                measurement=measurement,
                scan_index=self._scan_index,
                scan_complete=scan_complete,
            )

    def scan(self) -> dict[str, UltrasonicMeasurement]:
        """Run one blocking full scan and return every measurement."""

        results: dict[str, UltrasonicMeasurement] = {}

        for _ in self._names:
            result = self.step(wait=True)

            if result is not None:
                results[result.name] = result.measurement

        return results

    def shutdown(self) -> None:
        """Leave every trigger output LOW."""

        for sensor in self._sensors.values():
            try:
                sensor.shutdown()
            except Exception:
                pass
