"""Online/offline state with a circuit breaker.

Cloud use is opt-in.  While it is disabled, nothing here probes the network
at all.  When enabled:

* Calls get a bounded number of attempts inside a time budget.
* After repeated failures the link is marked OFFLINE and requests are not
  attempted until an exponentially growing retry time, so a dead link does
  not slow every reply.
* A later success restores ONLINE.

Callers always have a local fallback; this class never raises just because
the network is down.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Link(str, Enum):
    """Cloud reachability."""

    DISABLED = "disabled"
    UNKNOWN = "unknown"
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True)
class ConnectivityConfig:
    """Breaker settings."""

    failure_threshold: int = 2
    backoff_base_s: float = 5.0
    backoff_max_s: float = 120.0
    probe_interval_s: float = 30.0
    max_attempts: int = 2
    attempt_pause_s: float = 0.2


class ConnectivityManager:
    """Tracks whether the cloud can be used."""

    def __init__(
        self,
        *,
        enabled: Callable[[], bool] = lambda: False,
        probe: Callable[[], bool] | None = None,
        config: ConnectivityConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._enabled = enabled
        self._probe = probe
        self.config = config or ConnectivityConfig()
        self._clock = clock
        self._sleep = sleep
        self._link = Link.UNKNOWN
        self._failures = 0
        self._backoff = self.config.backoff_base_s
        self._next_try = 0.0
        self._next_probe = 0.0
        self.total_failures = 0

    @property
    def link(self) -> Link:
        """Current state; DISABLED whenever the user has not opted in."""

        if not self._enabled():
            return Link.DISABLED

        return self._link

    def allow_request(self) -> bool:
        """May a cloud request be attempted right now?"""

        if not self._enabled():
            return False

        if self._link is Link.OFFLINE:
            return self._clock() >= self._next_try

        return True

    def record_success(self) -> None:
        """A cloud call worked."""

        self._failures = 0
        self._backoff = self.config.backoff_base_s
        self._link = Link.ONLINE

    def record_failure(self) -> None:
        """A cloud call failed."""

        self.total_failures += 1
        self._failures += 1

        if self._failures >= self.config.failure_threshold:
            self._link = Link.OFFLINE
            self._next_try = self._clock() + self._backoff
            self._backoff = min(self._backoff * 2.0,
                                self.config.backoff_max_s)

    def tick(self) -> None:
        """Run the optional reachability probe when due."""

        if not self._enabled() or self._probe is None:
            return

        now = self._clock()

        if now < self._next_probe:
            return

        self._next_probe = now + self.config.probe_interval_s

        if self._link is Link.OFFLINE and now < self._next_try:
            return

        try:
            ok = bool(self._probe())
        except Exception:
            ok = False

        if ok:
            self.record_success()
        else:
            self.record_failure()

    def call(self, fn: Callable[[], object],
             budget_s: float = 6.0):
        """Run ``fn`` with bounded attempts; raise the last error.

        Not allowed attempts raise ``ConnectionError`` immediately.
        """

        if not self.allow_request():
            raise ConnectionError("cloud not available")

        deadline = self._clock() + budget_s
        last: Exception | None = None

        for attempt in range(self.config.max_attempts):
            if attempt and self._clock() >= deadline:
                break

            try:
                result = fn()
            except Exception as exc:
                last = exc
                self.record_failure()

                if not self.allow_request():
                    break

                if attempt + 1 < self.config.max_attempts:
                    self._sleep(self.config.attempt_pause_s)

                continue

            self.record_success()

            return result

        raise last or ConnectionError("cloud call failed")
