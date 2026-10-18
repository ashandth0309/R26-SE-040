"""CPU, memory and temperature sampling.

Returns ``None`` for anything the machine cannot report.  Nothing here is
estimated or defaulted to a plausible-looking number.
"""

from __future__ import annotations

import os
import resource
import time
from collections.abc import Callable
from pathlib import Path


def read_temperature_c(path: str = "/sys/class/thermal/thermal_zone0/temp"
                       ) -> float | None:
    try:
        return int(Path(path).read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_rss_mb(status_path: str = "/proc/self/status") -> float | None:
    try:
        for line in Path(status_path).read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass

    return None


def read_available_mem_mb(path: str = "/proc/meminfo") -> float | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass

    return None


class ResourceMonitor:
    """Process CPU share between samples, plus memory and temperature."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_wall = clock()
        self._last_cpu = self._cpu_seconds()

    @staticmethod
    def _cpu_seconds() -> float:
        usage = resource.getrusage(resource.RUSAGE_SELF)

        return usage.ru_utime + usage.ru_stime

    def sample(self) -> dict:
        wall, cpu = self._clock(), self._cpu_seconds()
        span = wall - self._last_wall
        cpu_pct = None

        if span > 0:
            # 100 percent equals one fully used core.
            cpu_pct = 100.0 * (cpu - self._last_cpu) / span

        self._last_wall, self._last_cpu = wall, cpu

        return {
            "cpu_percent_of_one_core": cpu_pct,
            "rss_mb": read_rss_mb(),
            "mem_available_mb": read_available_mem_mb(),
            "temperature_c": read_temperature_c(),
            "cpu_count": os.cpu_count(),
        }
