"""Deterministic fakes for ultrasonic driver and node tests.

A virtual clock and a fake GPIO that schedules echo pulses after the trigger
edge, so timing is reproducible and no hardware or real time is needed.
"""

import time as real_time

from buddy_core.sensors.ultrasonic import UltrasonicSensor

SPEED = UltrasonicSensor.SPEED_OF_SOUND_MPS
ECHO_LATENCY_S = 0.0005
NO_OBSTACLE_PULSE_S = 0.038


class FakeClock:
    """Virtual monotonic clock advanced by sleeps and GPIO polling."""

    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeTimeModule:
    """Stand-in for the ``time`` module inside the ultrasonic driver."""

    def __init__(self, clock):
        self.monotonic = clock.monotonic
        self.sleep = clock.sleep


class FakeEchoGPIO:
    """GPIO fake that behaves like several HC-SR04 modules."""

    def __init__(self, clock, wiring, poll_cost=20e-6):
        self.clock = clock
        self.poll_cost = poll_cost
        self.echo_for_trigger = {
            trigger: echo for trigger, echo in wiring.values()
        }
        self.scenarios = {}
        self.windows = {}
        self.trigger_high_since = {}
        self.trigger_pulses = []
        self.fail_pins = set()
        self.yield_on_read = False

    def set_scenario(self, echo_pin, kind, distance_m=None):
        self.scenarios[echo_pin] = (kind, distance_m)

    def configure_output(self, pin, initial=False):
        pass

    def configure_input(self, pin, pull=None):
        pass

    def write(self, pin, value):
        if pin not in self.echo_for_trigger:
            return

        now = self.clock.now

        if value:
            self.trigger_high_since[pin] = now
            return

        rose = self.trigger_high_since.pop(pin, None)

        if rose is None:
            return

        self.trigger_pulses.append((pin, rose, now))
        echo_pin = self.echo_for_trigger[pin]
        kind, distance_m = self.scenarios.get(echo_pin, ("none", None))
        start = now + ECHO_LATENCY_S

        if kind == "distance":
            self.windows[echo_pin] = (
                start,
                start + 2.0 * distance_m / SPEED,
            )
        elif kind == "clear":
            self.windows[echo_pin] = (start, start + NO_OBSTACLE_PULSE_S)
        else:
            self.windows.pop(echo_pin, None)

    def read(self, pin):
        if pin in self.fail_pins:
            raise RuntimeError("simulated GPIO failure")

        if self.yield_on_read:
            # Let threads genuinely interleave if nothing serialises them.
            real_time.sleep(0)

        self.clock.now += self.poll_cost
        kind, _ = self.scenarios.get(pin, ("none", None))

        if kind == "stuck":
            return True

        window = self.windows.get(pin)

        if window is None:
            return False

        return window[0] <= self.clock.now < window[1]


WIRING = {
    "front_left": (21, 22),
    "front_right": (23, 24),
    "rear_left": (25, 26),
    "rear_right": (27, 14),
}
