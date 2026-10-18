"""The voice engine: wires audio, recognition, dialogue and speech.

One ``process_frame`` call per 20 ms microphone frame and one ``tick`` per
~50 ms timer.  Everything is synchronous and driven from outside, which
keeps it testable and makes the safety order explicit:

**A spoken stop never waits for dialogue, an LLM, the network or speech
synthesis.**  ``_do_stop`` first publishes STOP through the router, then
cancels speech, then resets recognisers, and only then queues the spoken
acknowledgement (from a pre-synthesised cache when possible).

Muting is a safety event too: once the microphone is ignored a spoken stop
can no longer be heard, so muting also stops the robot.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np

from buddy_core.voice.action_gate import ActionGate
from buddy_core.voice.backchannel import BackchannelManager
from buddy_core.voice.command_router import (
    CommandRouter,
    RouteStatus,
)
from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueManager,
)
from buddy_core.voice.dsp import dbfs, envelope, rms, to_float
from buddy_core.voice.echo import EchoGuard
from buddy_core.voice.speaker_manager import Priority, SpeakerManager
from buddy_core.voice.stt import Recognizer, Transcript
from buddy_core.voice.tts_backends import SpeechStyle, Synthesizer
from buddy_core.voice.turn_manager import TurnManager, TurnState
from buddy_core.voice.vad import VoiceActivityDetector
from buddy_core.voice.voice_commands import (
    ParsedCommand,
    is_sleep,
    is_stop,
    is_wake,
    parse_command,
)

FRAME_S = 0.02


class InlineExecutor:
    """Runs work immediately on the caller's thread (tests, simple use).

    Production uses a thread so a slow model cannot stall the audio loop.
    """

    def submit(self, fn, *args) -> Future:
        """Run ``fn`` now and return a finished future."""

        future: Future = Future()

        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)

        return future


@dataclass(frozen=True)
class EngineConfig:
    """Engine timing and wording."""

    turn_end_silence_s: float = 0.8
    preroll_frames: int = 20
    partial_check_every: int = 4
    wake_reply: str = "Yes?"
    sleep_reply: str = "Okay, going to sleep."
    unclear_reply: str = "Sorry, I didn't catch that."
    failure_reply: str = "Sorry, I could not do that."
    min_turn_s: float = 0.25


class VoiceEngine:
    """Offline-capable voice companion core."""

    def __init__(
        self,
        *,
        vad: VoiceActivityDetector,
        wake_rec: Recognizer,
        command_rec: Recognizer,
        general_rec: Recognizer,
        router: CommandRouter,
        speaker: SpeakerManager,
        synth: Synthesizer,
        guard: EchoGuard,
        turn: TurnManager,
        dialogue: DialogueManager,
        backchannel: BackchannelManager | None = None,
        context: ConversationContext | None = None,
        config: EngineConfig | None = None,
        known_locations: tuple[str, ...] = (),
        style: SpeechStyle | None = None,
        clock: Callable[[], float] = time.monotonic,
        emotion_hint: Callable[[], str | None] = lambda: None,
        utterance_observer: Callable[[np.ndarray], None] | None = None,
        style_provider: Callable[[], SpeechStyle] | None = None,
        executor=None,
        action_gate: ActionGate | None = None,
    ) -> None:
        self.vad = vad
        self.wake_rec = wake_rec
        self.command_rec = command_rec
        self.general_rec = general_rec
        self.router = router
        self.speaker = speaker
        self.synth = synth
        self.guard = guard
        self.turn = turn
        self.dialogue = dialogue
        self.backchannel = backchannel or BackchannelManager()
        self.context = context or ConversationContext()
        self.config = config or EngineConfig()
        self.known_locations = known_locations
        self.style = style or SpeechStyle()
        self._clock = clock
        self.emotion_hint = emotion_hint
        self.utterance_observer = utterance_observer
        self.style_provider = style_provider
        self.executor = executor or InlineExecutor()
        self.action_gate = action_gate or ActionGate(clock=clock)
        self._gen = 0
        self._inflight: tuple[Future, int] | None = None

        self.muted = False
        self.quiet_mode = False
        self.tick_hooks: list[Callable[[], None]] = []
        self.events: deque = deque(maxlen=200)
        self.errors = 0

        self._preroll: deque = deque(maxlen=self.config.preroll_frames)
        self._cache: dict = {}
        self._user_active = False
        self._wake_context = False
        self._utt_start = 0.0
        self._last_speech = 0.0
        self._frame_count = 0
        self._stop_fired = False
        self._cmd_texts: list[Transcript] = []
        self._gen_texts: list[Transcript] = []
        self._wake_texts: list[Transcript] = []
        self._utt_audio: list[np.ndarray] = []
        self._playback_env: np.ndarray | None = None
        self._playback_t0 = 0.0

        speaker.on_start = self._on_speaker_start
        speaker.on_finish = self._on_speaker_finish
        turn.on_transition = self._on_transition

    # ----- bookkeeping -----

    def _log(self, kind: str, detail: str = "") -> None:
        self.events.append((self._clock(), kind, detail))

    def _on_transition(self, old: TurnState, new: TurnState, event: str
                       ) -> None:
        self._log("state", f"{old.value}->{new.value} ({event})")

        if new in (TurnState.LISTENING_FOR_WAKE, TurnState.SLEEPING,
                   TurnState.ERROR_RECOVERY) and self.router.moving:
            self.router.stop(f"state {new.value}")

    def _on_speaker_start(self, utt) -> None:
        self.guard.playback_started()
        env = envelope(to_float(utt.samples), max(1, int(utt.rate * FRAME_S)))
        gain_db = 20.0 * np.log10(max(self.speaker.effective_gain(
            utt.volume), 1e-3))
        self._playback_env = np.array(
            [dbfs(float(v)) for v in env]) + gain_db
        self._playback_t0 = self._clock()

        if utt.priority is not Priority.LOW:
            self.turn.buddy_started()

        self._log("speak", utt.text)

    def _on_speaker_finish(self, utt, completed: bool) -> None:
        self.guard.playback_stopped()

        if self.turn.state is TurnState.BUDDY_SPEAKING \
                and not self.speaker.speaking:
            self.turn.buddy_finished()

    # ----- speech output -----

    def _audio_for(self, text: str, style: SpeechStyle):
        key = (text, style)

        if key not in self._cache:
            self._cache[key] = self.synth.synthesize(text, style)

        return self._cache[key]

    def warm_cache(self, phrases: list[str]) -> None:
        """Pre-synthesise short phrases so acknowledgements are instant."""

        for text in phrases:
            try:
                self._audio_for(text, self.style)
            except Exception:
                self.errors += 1

    def say(self, text: str, priority: Priority = Priority.HIGH,
            style: SpeechStyle | None = None) -> bool:
        """Synthesize and queue ``text``.  Returns True if queued."""

        if not text:
            return False

        style = style or self.style

        try:
            samples, rate = self._audio_for(text, style)
        except Exception:
            self.errors += 1
            self._log("tts_error", text)

            return False

        uid = self.speaker.say(
            text, samples, rate, priority=priority, volume=style.volume,
        )
        self.speaker.tick()

        return uid is not None

    # ----- control -----

    def set_muted(self, muted: bool) -> None:
        """Privacy mute.  Also stops motion, since stop can no longer be
        heard."""

        if muted == self.muted:
            return

        self.muted = muted

        if muted:
            self.router.stop("muted")
            self.speaker.cancel_all()
            self._abort_user_turn()
            self.turn.mute()
        else:
            self.turn.unmute()

        self._log("mute", str(muted))

    def set_quiet(self, quiet: bool) -> None:
        """Quiet mode: no backchannels, no unprompted speech."""

        self.quiet_mode = quiet

    def _abort_user_turn(self) -> None:
        self._user_active = False
        self._stop_fired = False
        self._cmd_texts.clear()
        self._gen_texts.clear()
        self._wake_texts.clear()
        self._utt_audio.clear()
        self._reset_recognizers()

    def _reset_recognizers(self) -> None:
        for rec in (self.wake_rec, self.command_rec, self.general_rec):
            rec.reset()

    def _do_stop(self, reason: str) -> None:
        """Highest-priority path.  Order is deliberate."""

        result = self.router.stop(reason)
        self.action_gate.pending = None
        self._cancel_reply()
        self.speaker.cancel_all()
        self._abort_user_turn()
        self.turn.stop()
        self.vad.reset()
        self._log("stop", reason)
        self.say(result.ack or "Stopping.", Priority.CRITICAL)

    # ----- frames -----

    def process_frame(self, frame: np.ndarray) -> None:
        """Handle one microphone frame (int16, 16 kHz)."""

        now = self._clock()
        self._frame_count += 1
        self._preroll.append(frame)

        if self.muted or self.turn.state is TurnState.SLEEPING:
            return

        if self.guard.blocked or self.speaker.speaking:
            self._update_playback_level(now)

            if self.speaker.speaking and self.guard.observe(frame):
                self._handle_barge_in()
            else:
                return

        if self.guard.blocked:
            return

        self._listen(frame, now)

    def _update_playback_level(self, now: float) -> None:
        env = self._playback_env

        if env is None or env.size == 0:
            return

        idx = min(int((now - self._playback_t0) / FRAME_S), env.size - 1)
        self.guard.playback_level_db(float(env[max(idx, 0)]))

    def _handle_barge_in(self) -> None:
        self.speaker.cancel_all()

        if self.turn.state is TurnState.BUDDY_SPEAKING:
            self.turn.barge_in()

        self._log("barge_in")

        for old in list(self._preroll):
            self._listen(old, self._clock(), replay=True)

    def _listen(self, frame: np.ndarray, now: float, replay: bool = False
                ) -> None:
        event = self.vad.process(frame)

        if event.is_speech_frame:
            self._last_speech = now

        if not self._user_active:
            if event.speech_started:
                self._start_user_turn(now)

                if not replay:
                    for old in list(self._preroll)[:-1]:
                        self._feed(old)
            else:
                return

        self._feed(frame)
        self._check_partial_stop()

        if self._stop_fired:
            return

        pause = now - self._last_speech

        if pause >= self.config.turn_end_silence_s:
            self._end_user_turn(now)

            return

        self._maybe_backchannel(now, pause)

    def _start_user_turn(self, now: float) -> None:
        if self.turn.state is TurnState.PROCESSING:
            self._cancel_reply()
            self.turn.nothing_to_say()

        self._wake_context = self.turn.wants_wake_word
        self._user_active = True
        self._stop_fired = False
        self._utt_start = now
        self._last_speech = now
        self._cmd_texts.clear()
        self._gen_texts.clear()
        self._wake_texts.clear()
        self._utt_audio.clear()
        self._reset_recognizers()
        self.backchannel.on_user_resumed()
        self.turn.user_started()

    def _feed(self, frame: np.ndarray) -> None:
        data = frame.tobytes()

        # Held in memory for this utterance only; see _end_user_turn.
        if self.utterance_observer and not self._wake_context \
                and len(self._utt_audio) < 1000:
            self._utt_audio.append(frame)

        if self._wake_context:
            result = self.wake_rec.accept(data)

            if result:
                self._wake_texts.append(result)

            return

        result = self.command_rec.accept(data)

        if result:
            self._cmd_texts.append(result)

        result = self.general_rec.accept(data)

        if result:
            self._gen_texts.append(result)

    def _check_partial_stop(self) -> None:
        if self._frame_count % self.config.partial_check_every:
            return

        rec = self.wake_rec if self._wake_context else self.command_rec

        if is_stop(rec.partial()) and not self._stop_fired:
            self._stop_fired = True
            self._do_stop("spoken stop (partial)")

    def _maybe_backchannel(self, now: float, pause: float) -> None:
        if self._wake_context or self.quiet_mode:
            return

        rec = self.general_rec
        line = self.backchannel.consider(
            now,
            user_speaking_ms=int((self._last_speech - self._utt_start)
                                 * 1000),
            pause_ms=int(pause * 1000),
            partial_text=rec.partial(),
            emotion=self.emotion_hint(),
            buddy_speaking=self.speaker.speaking,
            silenced=self.quiet_mode or self.muted,
            command_active=self.router.moving
            or self.backchannel.looks_like_command(
                self.command_rec.partial()),
        )

        if line:
            self.say(line, Priority.LOW)

    # ----- end of turn -----

    @staticmethod
    def _join(parts: list[Transcript]) -> Transcript | None:
        parts = [p for p in parts if p and p.text]

        if not parts:
            return None

        confs = [p.confidence for p in parts if p.confidence is not None]

        return Transcript(
            " ".join(p.text for p in parts),
            sum(confs) / len(confs) if confs else None,
            parts[0].language,
        )

    def _end_user_turn(self, now: float) -> None:
        self._user_active = False
        wake_context = self._wake_context
        already_stopped = self._stop_fired
        self._stop_fired = False
        self.speaker.cancel_low_priority()

        if already_stopped:
            return

        if wake_context:
            self._wake_texts.append(self.wake_rec.finalize())
            heard = self._join(self._wake_texts)
            self.turn.wake_utterance_ended()
            self._handle_sleeping_text(heard)

            return

        self._cmd_texts.append(self.command_rec.finalize())
        self._gen_texts.append(self.general_rec.finalize())
        cmd = self._join(self._cmd_texts)
        gen = self._join(self._gen_texts)
        too_short = (now - self._utt_start) < self.config.min_turn_s

        if (cmd is None and gen is None) or too_short and gen is None:
            self.turn.user_ended(False)

            return

        self._observe_utterance()
        self.turn.user_ended(True)
        self._handle_awake_text(cmd, gen)

    def _observe_utterance(self) -> None:
        """Give the utterance audio to the observer, then discard it."""

        audio, self._utt_audio = self._utt_audio, []

        if not self.utterance_observer or not audio:
            return

        try:
            self.utterance_observer(np.concatenate(audio))
        except Exception:
            self.errors += 1

    def _handle_sleeping_text(self, heard: Transcript | None) -> None:
        if heard is None:
            return

        text = heard.text
        self._log("wake_heard", text)

        if is_stop(text):
            self._do_stop("spoken stop (wake)")

            return

        if is_wake(text):
            self.turn.wake()
            self.say(self.config.wake_reply, Priority.HIGH)

    def _handle_awake_text(self, cmd: Transcript | None,
                           gen: Transcript | None) -> None:
        primary = cmd or gen
        text = (primary.text if primary else "")
        self._log("heard", text)

        if any(t and is_stop(t.text) for t in (cmd, gen)):
            self._do_stop("spoken stop")

            return

        if any(t and is_sleep(t.text) for t in (cmd, gen)):
            self.router.stop("sleep")
            self.turn.sleep()
            self.say(self.config.sleep_reply, Priority.HIGH)

            return

        parsed = self._find_command(cmd, gen)
        confirmed = self.action_gate.confirm(text)

        if confirmed is not None:
            parsed = (confirmed, primary)

        if parsed is not None:
            command, source = parsed
            result = self.router.handle(
                command, awake=self.turn.awake,
                confidence=source.confidence,
            )

            if self._speak_route_result(result):
                return

        if gen is None:
            self.turn.nothing_to_say()

            return

        self._reply(gen.text)

    def _find_command(self, cmd: Transcript | None,
                      gen: Transcript | None
                      ) -> tuple[ParsedCommand, Transcript] | None:
        for source in (cmd, gen):
            if source is None:
                continue

            parsed = parse_command(source.text, self.known_locations)

            if parsed is not None:
                return parsed, source

        return None

    def _speak_route_result(self, result) -> bool:
        """Speak the outcome.  Returns True if the turn is finished."""

        status = result.status

        if status is RouteStatus.NOT_APPLICABLE:
            return False

        if status is RouteStatus.DUPLICATE:
            self.turn.nothing_to_say()

            return True

        if status is RouteStatus.LOW_CONFIDENCE:
            text = self.config.unclear_reply
        elif status is RouteStatus.FAILED:
            text = result.ack or self.config.failure_reply
        else:
            text = result.ack

        if not self.say(text, Priority.HIGH):
            self.turn.nothing_to_say()

        return True

    def _reply(self, text: str) -> None:
        """Ask dialogue for a reply on the executor, never blocking audio."""

        self._cancel_reply()
        self._gen += 1
        future = self.executor.submit(
            self.dialogue.respond, text, self.context)
        self._inflight = (future, self._gen)
        self._poll_reply()

    def _cancel_reply(self) -> None:
        """Forget any reply still being prepared."""

        self._gen += 1

        if self._inflight is not None:
            self._inflight[0].cancel()
            self._inflight = None

    def _poll_reply(self) -> None:
        if self._inflight is None:
            return

        future, gen = self._inflight

        if not future.done():
            return

        self._inflight = None

        if gen != self._gen or self.turn.state is not TurnState.PROCESSING:
            self._log("stale_reply_dropped")

            return

        try:
            reply = future.result()
        except Exception:
            self.errors += 1
            self._log("dialogue_error")
            self.turn.error()

            return

        text = reply.text
        question = self.action_gate.propose(reply.suggested_action)

        if question:
            text = f"{text} {question}"

        style = self.style_provider() if self.style_provider else None

        if not self.say(text, Priority.NORMAL, style):
            self.turn.nothing_to_say()

    # ----- timer -----

    def tick(self) -> None:
        """Advance timers.  Call about every 50 ms."""

        self._poll_reply()
        self.speaker.tick()
        self.router.tick()
        self.turn.tick()

        for hook in self.tick_hooks:
            hook()

        if (
            self._user_active
            and self._clock() - self._utt_start
            >= self.turn.config.user_speaking_timeout_s
        ):
            self._end_user_turn(self._clock())

    def level_dbfs(self, frame: np.ndarray) -> float:
        """Convenience for diagnostics."""

        return dbfs(rms(frame))
