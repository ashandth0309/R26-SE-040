"""Mock ultrasonic array driven by a simulated world (simulation only)."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.sensor_model import (
    RangeSample,
    SensorSpec,
    SensorStatus,
)
from buddy_navigation.sim.world import World2D

FAULT_MODES = ("no_response", "stuck_high", "specular", "frozen")


class MockUltrasonicArray:
    """Sequentially triggered mock sensors with injectable failures.

    * ``no_response`` / ``stuck_high``: sensor reports FAULT;
    * ``specular``: sound bounces away, so a present obstacle looks CLEAR
      (the dangerous silent failure);
    * ``frozen``: the sensor keeps repeating its last value.
    """

    def __init__(
        self,
        world: World2D,
        specs: Iterable[SensorSpec],
        *,
        order: Iterable[str] | None = None,
        trigger_interval_s: float = 0.06,
        noise_std_m: float = 0.0,
        dropout_probability: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.world = world
        self.specs = {spec.name: spec for spec in specs}
        self.order = tuple(order or self.specs)
        self.trigger_interval_s = trigger_interval_s
        self.noise_std_m = noise_std_m
        self.dropout_probability = dropout_probability
        self.faults: dict[str, str] = {}
        self._rng = random.Random(seed)
        self._index = 0
        self._last_trigger = -math.inf
        self._last_value: dict[str, float | None] = {}
        self.trigger_log: list[tuple[float, str]] = []

    def inject_fault(self, name: str, mode: str) -> None:
        """Make a sensor misbehave."""

        if name not in self.specs or mode not in FAULT_MODES:
            raise ValueError(f"Unknown sensor or fault mode: {name}, {mode}")

        self.faults[name] = mode

    def clear_fault(self, name: str) -> None:
        """Restore a sensor."""

        self.faults.pop(name, None)

    def measure(self, name: str, pose: Pose2D, now: float) -> RangeSample:
        """Measure one sensor right now."""

        spec = self.specs[name]
        fault = self.faults.get(name)

        if fault in ("no_response", "stuck_high"):
            return RangeSample(name, SensorStatus.FAULT, None, now)

        if fault == "frozen" and name in self._last_value:
            return self._sample(name, self._last_value[name], spec, now)

        if self._rng.random() < self.dropout_probability:
            return RangeSample(name, SensorStatus.FAULT, None, now)

        true = self.world.beam_distance(pose, spec)

        if fault == "specular":
            true = math.inf

        if math.isfinite(true) and self.noise_std_m > 0.0:
            true += self._rng.gauss(0.0, self.noise_std_m)

        value = true if math.isfinite(true) else None
        self._last_value[name] = value

        return self._sample(name, value, spec, now)

    @staticmethod
    def _sample(
        name: str,
        value: float | None,
        spec: SensorSpec,
        now: float,
    ) -> RangeSample:
        if value is None:
            return RangeSample(name, SensorStatus.CLEAR, None, now)

        if value < spec.min_range_m:
            return RangeSample(name, SensorStatus.TOO_CLOSE, None, now)

        return RangeSample(name, SensorStatus.OK, value, now)

    def step(self, pose: Pose2D, now: float) -> RangeSample | None:
        """Trigger the next sensor if the spacing interval has elapsed."""

        if now - self._last_trigger < self.trigger_interval_s - 1e-9:
            return None

        name = self.order[self._index]
        self._index = (self._index + 1) % len(self.order)
        self._last_trigger = now
        self.trigger_log.append((now, name))

        return self.measure(name, pose, now)

    def min_trigger_gap(self) -> float:
        """Smallest observed spacing between triggers (for tests)."""

        times = [t for t, _ in self.trigger_log]

        if len(times) < 2:
            return math.inf

        return min(b - a for a, b in zip(times, times[1:]))
