"""Thread that drives a ``Companion``: microphone in, engine, timers.

All engine calls happen on this one thread.  Other threads (ROS callbacks)
hand work over with ``submit``, which is safe from any thread.  This keeps
``VoiceEngine`` single-threaded and the audio loop free of locks.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from buddy_core.voice.audio_manager import FrameChunker
from buddy_core.voice.engine_factory import (
    WARM_PHRASES,
    Companion,
    current_hour,
)
from buddy_core.voice.proactive import SuggestContext
from buddy_core.voice.speaker_manager import Priority


class EngineRunner:
    """Drive the companion."""

    def __init__(
        self,
        companion: Companion,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        hour: Callable[[], int] = current_hour,
        tick_s: float = 0.05,
        proactive_s: float = 30.0,
        max_frames_per_step: int = 8,
        mic_loss_stop_s: float = 1.0,
    ) -> None:
        self.c = companion
        self._clock = clock
        self._sleep = sleep
        self._hour = hour
        self._tick_s = tick_s
        self._max_frames = max_frames_per_step
        self._mic_loss_stop_s = mic_loss_stop_s
        self._no_mic_since: float | None = None
        self._proactive_s = proactive_s
        self._chunker = FrameChunker(320)
        self._inbox: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._next_tick = 0.0
        self._next_proactive = clock() + proactive_s
        self._awake_since: float | None = None
        self._last_user_speech = -1e9
        self.loop_errors = 0

    # ----- lifecycle -----

    def start(self) -> None:
        """Warm up and start the loop thread."""

        self.c.engine.warm_cache(WARM_PHRASES)
        started = self.c.mics.start()

        if started == 0:
            self.c.notes.append("no microphone could be started")

        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="voice_engine")
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the loop, the robot and the devices."""

        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self.c.router.stop("shutdown")
        self.c.speaker.cancel_all()
        self.c.mics.stop()

    def submit(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the engine thread."""

        self._inbox.put(fn)

    # ----- loop -----

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.step()
            except Exception:
                self.loop_errors += 1
                busy = False

            if not busy:
                self._sleep(0.005)

    def step(self) -> bool:
        """One pass.  Returns True if audio was processed."""

        self._drain_inbox()
        busy = False

        for _ in range(self._max_frames):
            frame = self.c.mics.pump()

            if frame is None:
                break

            busy = True

            for block in self._chunker.push(frame.samples):
                self.c.engine.process_frame(block)

        now = self._clock()

        if now >= self._next_tick:
            self._next_tick = now + self._tick_s
            self.c.engine.tick()
            self._track_awake(now)
            self._watch_microphone(now)

        if now >= self._next_proactive:
            self._next_proactive = now + self._proactive_s
            self._proactive(now)

        return busy

    def _watch_microphone(self, now: float) -> None:
        """If every microphone is gone, a spoken stop is impossible, so
        stop the robot rather than let it run on."""

        if self.c.mics.active is not None:
            self._no_mic_since = None

            return

        if self._no_mic_since is None:
            self._no_mic_since = now

        if (
            now - self._no_mic_since >= self._mic_loss_stop_s
            and self.c.router.moving
        ):
            self.c.router.stop("microphone lost")

    def _drain_inbox(self) -> None:
        while True:
            try:
                fn = self._inbox.get_nowait()
            except queue.Empty:
                return

            try:
                fn()
            except Exception:
                self.loop_errors += 1

    def _track_awake(self, now: float) -> None:
        awake = self.c.turn.awake

        if awake and self._awake_since is None:
            self._awake_since = now
        elif not awake:
            self._awake_since = None

    def _proactive(self, now: float) -> None:
        sug = self.c.suggester
        engine = self.c.engine

        if sug is None or self._awake_since is None:
            return

        ctx = SuggestContext(
            hour=self._hour(),
            awake_minutes=(now - self._awake_since) / 60.0,
            emotion=self.c.emotion.hint(),
            silenced=engine.quiet_mode,
            muted=engine.muted,
            busy=engine.speaker.speaking or self.c.router.moving
            or self.c.turn.state.value != "AWAKE",
            seconds_since_user_spoke=now - engine.last_speech_time,
        )
        choice = sug.maybe_suggest(ctx)

        if choice is not None:
            engine.say(choice.text, Priority.LOW)
