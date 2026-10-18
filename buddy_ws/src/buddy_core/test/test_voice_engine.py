"""Phase 4 tests: turn manager, backchannel, dialogue and the engine."""

import datetime
import random

import numpy as np
import pytest

from buddy_core.voice import command_router as cr
from buddy_core.voice.backchannel import (
    BackchannelConfig,
    BackchannelKind,
    BackchannelManager,
)
from buddy_core.voice.command_router import (
    CommandRouter,
    RecordingSink,
    RouterConfig,
)
from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueManager,
    DialogueReply,
    MlIntentClassifier,
    clean_llm_text,
)
from buddy_core.voice.echo import EchoGuard
from buddy_core.voice.speaker_manager import (
    Priority,
    SpeakerConfig,
    SpeakerManager,
)
from buddy_core.voice.stt import ScriptedRecognizer, Transcript
from buddy_core.voice.tts_backends import FakeSynthesizer
from buddy_core.voice.turn_manager import (
    TRANSITIONS,
    TurnConfig,
    TurnManager,
    TurnState,
)
from buddy_core.voice.vad import VoiceActivityDetector
from buddy_core.voice.voice_engine import EngineConfig, VoiceEngine

RATE = 16000
BLOCK = 320
S = TurnState


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# ----- turn manager -----

def test_turn_starts_listening_for_wake():
    assert TurnManager(clock=Clock()).state is S.LISTENING_FOR_WAKE


def test_every_event_from_every_state_is_defined_or_counted():
    for event, table in TRANSITIONS.items():
        for state in S:
            tm = TurnManager(clock=Clock())
            tm.state = state
            ok = tm._go(event)
            assert ok == (state in table)

            if not ok:
                assert tm.state is state and tm.illegal_events == 1


def test_happy_path_conversation():
    clock = Clock()
    tm = TurnManager(clock=clock)
    assert tm.user_started() and tm.state is S.USER_SPEAKING
    tm.wake_utterance_ended()
    assert tm.wake() and tm.awake
    assert tm.user_started()
    assert tm.user_ended() and tm.state is S.PROCESSING
    assert tm.buddy_started() and tm.state is S.BUDDY_SPEAKING
    assert tm.buddy_finished() and tm.state is S.COOLDOWN
    clock.advance(0.5)
    tm.tick()
    assert tm.state is S.AWAKE


def test_barge_in_path():
    tm = TurnManager(clock=Clock())
    tm.wake()
    tm.user_started()
    tm.user_ended()
    tm.buddy_started()
    assert tm.barge_in() and tm.state is S.INTERRUPTED
    assert tm.user_started() and tm.state is S.USER_SPEAKING


def test_stop_is_legal_in_every_state_except_muted():
    for state in S:
        tm = TurnManager(clock=Clock())
        tm.state = state
        result = tm.stop()

        if state is S.SLEEPING:
            assert not result and tm.state is S.SLEEPING
        else:
            assert result
            assert tm.state in (S.AWAKE, S.LISTENING_FOR_WAKE)


def test_stop_while_asleep_stays_in_wake_listening():
    tm = TurnManager(clock=Clock())
    assert tm.stop() and tm.state is S.LISTENING_FOR_WAKE


def test_mute_closes_ear_until_unmuted():
    tm = TurnManager(clock=Clock())
    tm.wake()
    assert tm.mute() and not tm.ear_open
    assert not tm.user_started()
    assert tm.unmute() and tm.state is S.LISTENING_FOR_WAKE


def test_ear_closed_while_buddy_speaks():
    tm = TurnManager(clock=Clock())
    tm.wake()
    tm.user_started()
    tm.user_ended()
    tm.buddy_started()
    assert not tm.ear_open


@pytest.mark.parametrize("state,wait,expected", [
    (S.COOLDOWN, 0.5, S.LISTENING_FOR_WAKE),
    (S.PROCESSING, 11.0, S.ERROR_RECOVERY),
    (S.ERROR_RECOVERY, 2.0, S.LISTENING_FOR_WAKE),
    (S.INTERRUPTED, 1.5, S.AWAKE),
    (S.USER_SPEAKING, 30.0, S.PROCESSING),
    (S.AWAKE, 61.0, S.LISTENING_FOR_WAKE),
])
def test_timeouts(state, wait, expected):
    clock = Clock()
    tm = TurnManager(clock=clock)
    tm.state = state
    tm._entered = clock()
    tm._last_activity = clock()
    clock.advance(wait)
    tm.tick()
    assert tm.state is expected


def test_stop_ack_while_asleep_does_not_wake_buddy():
    clock = Clock()
    tm = TurnManager(clock=clock)
    tm.stop()
    assert tm.buddy_started() and not tm.awake
    assert tm.buddy_finished()
    clock.advance(0.5)
    tm.tick()
    assert tm.state is S.LISTENING_FOR_WAKE and not tm.awake
    assert tm.wants_wake_word


def test_awake_cooldown_returns_to_awake():
    clock = Clock()
    tm = TurnManager(clock=clock)
    tm.wake()
    tm.buddy_started()
    tm.buddy_finished()
    clock.advance(0.5)
    tm.tick()
    assert tm.state is S.AWAKE and tm.awake


def test_awake_timeout_not_triggered_early():
    clock = Clock()
    tm = TurnManager(TurnConfig(awake_timeout_s=10), clock=clock)
    tm.wake()
    clock.advance(5)
    tm.tick()
    assert tm.state is S.AWAKE


def test_transition_callback_and_history():
    seen = []
    tm = TurnManager(clock=Clock(),
                     on_transition=lambda o, n, e: seen.append((o, n, e)))
    tm.wake()
    assert seen == [(S.LISTENING_FOR_WAKE, S.AWAKE, "wake")]


# ----- backchannel -----

def ask(bm, now=100.0, **kw):
    base = dict(
        user_speaking_ms=5000, pause_ms=700,
        partial_text="so today I went to the market and bought some things",
    )
    base.update(kw)

    return bm.consider(now, **base)


def test_backchannel_fires_in_valid_pause():
    bm = BackchannelManager(rng=random.Random(1))
    assert ask(bm) is not None


@pytest.mark.parametrize("kw,reason", [
    (dict(user_speaking_ms=1000), "utterance too short"),
    (dict(pause_ms=100), "pause timing"),
    (dict(pause_ms=3000), "pause timing"),
    (dict(buddy_speaking=True), "buddy speaking"),
    (dict(silenced=True), "silenced"),
    (dict(command_active=True), "command"),
    (dict(partial_text="go forward"), "command"),
    (dict(partial_text="stop"), "command"),
    (dict(partial_text="go for"), "command"),
])
def test_backchannel_suppressed(kw, reason):
    bm = BackchannelManager(rng=random.Random(1))
    assert ask(bm, **kw) is None
    assert bm.suppressed_reason == reason


def test_backchannel_can_be_disabled():
    bm = BackchannelManager(BackchannelConfig(enabled=False))
    assert ask(bm) is None
    bm2 = BackchannelManager()
    bm2.set_enabled(False)
    assert ask(bm2) is None


def test_backchannel_gap_rate_limit_and_pause_once():
    bm = BackchannelManager(
        BackchannelConfig(min_gap_s=1.0, max_per_minute=2),
        rng=random.Random(2))
    assert bm.consider(0.0, user_speaking_ms=5000, pause_ms=700) is not None
    assert bm.consider(0.5, user_speaking_ms=5000, pause_ms=700) is None
    assert bm.suppressed_reason == "already fired this pause"
    bm.on_user_resumed()
    assert bm.consider(0.8, user_speaking_ms=5000, pause_ms=700) is None
    assert bm.suppressed_reason == "too soon"
    assert bm.consider(2.0, user_speaking_ms=5000, pause_ms=700) is not None
    bm.on_user_resumed()
    assert bm.consider(4.0, user_speaking_ms=5000, pause_ms=700) is None
    assert bm.suppressed_reason == "rate limit"


def test_backchannel_category_follows_content_and_emotion():
    bm = BackchannelManager()
    assert bm.choose_kind("i feel so tired today", None) \
        is BackchannelKind.EMPATHY
    assert bm.choose_kind("hello there", "sad") is BackchannelKind.EMPATHY
    assert bm.choose_kind("i finally passed my exam", None) \
        is BackchannelKind.SURPRISE
    assert bm.choose_kind("we went out", None) \
        is BackchannelKind.ACKNOWLEDGE


def test_backchannel_wording_not_repeated_back_to_back():
    bm = BackchannelManager(BackchannelConfig(min_gap_s=0, max_per_minute=99),
                            rng=random.Random(3))
    lines = []
    for i in range(12):
        bm.on_user_resumed()
        lines.append(bm.consider(float(i), user_speaking_ms=5000,
                                 pause_ms=700, partial_text="we went out"))
    assert all(a != b for a, b in zip(lines, lines[1:]))


# ----- dialogue -----

NOW = datetime.datetime(2026, 10, 18, 15, 5)


def dm(**kw):
    kw.setdefault("rng", random.Random(1))
    kw.setdefault("now", lambda: NOW)
    return DialogueManager(**kw), ConversationContext()


def test_time_and_date_rules():
    d, ctx = dm()
    assert d.respond("what time is it", ctx).text == "It's 3:05 PM."
    assert "Sunday, October 18, 2026" in d.respond("What's the date?",
                                                   ctx).text


def test_greeting_uses_name_when_known():
    d, ctx = dm()
    ctx.user_name = "Yohan"
    assert "Yohan" in d.respond("hello", ctx).text


def test_repeat_and_empty_repeat():
    d, ctx = dm()
    assert "haven't said" in d.respond("repeat that", ctx).text
    first = d.respond("who are you", ctx).text
    assert d.respond("say that again", ctx).text == first


def test_jokes_do_not_repeat_until_pool_used():
    d, ctx = dm()
    jokes = [d.respond("tell me a joke", ctx).text for _ in range(3)]
    assert len(set(jokes)) == 3


def test_replies_vary_for_repeated_question():
    d, ctx = dm()
    seen = [d.respond("how are you", ctx).text for _ in range(6)]
    assert all(a != b for a, b in zip(seen, seen[1:]))
    assert len(set(seen)) >= 2


def test_unknown_short_and_long_fallbacks():
    d, ctx = dm()
    assert d.respond("blorp", ctx).source == "fallback"
    long = d.respond("so yesterday i went to the shop and it was closed",
                     ctx)
    assert long.intent == "share"


def test_hook_answers_first():
    hook = lambda text, ctx: (  # noqa: E731
        DialogueReply("hooked", "hook") if "tired" in text else None)
    d, ctx = dm(hooks=[hook])
    assert d.respond("i am tired", ctx).text == "hooked"
    assert d.respond("hello", ctx).text != "hooked"


class FakeClassifier:
    def __init__(self, label, conf):
        self.label, self.conf = label, conf

    def predict(self, text):
        return self.label, self.conf


def test_intent_classifier_layer_and_threshold():
    d, ctx = dm(classifier=FakeClassifier("thanks", 0.9))
    r = d.respond("much obliged friend", ctx)
    assert r.source == "intent" and r.intent == "thanks"
    d2, ctx2 = dm(classifier=FakeClassifier("thanks", 0.2))
    assert d2.respond("much obliged friend", ctx2).source == "fallback"


def test_ml_adapter_missing_module_is_safe():
    clf = MlIntentClassifier(module=None)

    if not clf.available:
        assert clf.predict("hello") == ("unknown", 0.0)


def test_ml_adapter_with_module_and_failure():
    class Mod:
        @staticmethod
        def predict_intent(text, threshold):
            return "greeting"

    assert MlIntentClassifier(module=Mod).predict("x") == ("greeting", 1.0)

    class Bad:
        @staticmethod
        def predict_intent(text, threshold):
            raise RuntimeError

    assert MlIntentClassifier(module=Bad).predict("x") == ("unknown", 0.0)


class FakeLlm:
    def __init__(self, reply="Sure, that sounds lovely.", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def complete(self, messages, timeout_s):
        self.calls.append((messages, timeout_s))

        if self.fail:
            raise TimeoutError

        return self.reply


def test_llm_used_only_after_local_layers():
    llm = FakeLlm()
    d, ctx = dm(llm=llm)
    assert d.respond("what time is it", ctx).source == "rule"
    assert llm.calls == []
    r = d.respond("what do you think about rain", ctx)
    assert r.source == "llm" and llm.calls[0][1] == d.llm_timeout_s


def test_llm_failure_and_disallowed_fall_back():
    d, ctx = dm(llm=FakeLlm(fail=True))
    assert d.respond("tell me about whales", ctx).source == "fallback"
    assert d.llm_failures == 1
    llm = FakeLlm()
    d2, ctx2 = dm(llm=llm, llm_allowed=lambda: False)
    assert d2.respond("tell me about whales", ctx2).source == "fallback"
    assert llm.calls == []


def test_llm_action_is_only_a_suggestion():
    text, action = clean_llm_text("Shall I? ACTION: go forward")
    assert action == "go forward" and "ACTION" not in text
    d, ctx = dm(llm=FakeLlm("Sure. ACTION: go forward"))
    r = d.respond("take a walk", ctx)
    assert r.suggested_action == "go forward"


def test_clean_llm_text_strips_markup_and_limits_length():
    text, _ = clean_llm_text("**Hello** `code` ```x = 1``` there")
    assert "*" not in text and "x = 1" not in text
    long, _ = clean_llm_text("This is a sentence. " * 40)
    assert len(long) <= 321 and long.endswith(".")


def test_context_keeps_recent_turns_only():
    d, ctx = dm()
    for i in range(20):
        d.respond(f"blorp {i}", ctx)
    assert len(ctx.turns) == 12
    assert ctx.as_messages()[0]["role"] in ("user", "assistant")


# ----- engine harness -----

class OrderLog:
    def __init__(self):
        self.items = []


class OrderSink(RecordingSink):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def send_movement(self, command, *a):
        super().send_movement(command, *a)
        self.log.items.append(("publish", command))


class Speaker:
    def __init__(self, clock, log):
        self.clock, self.log = clock, log
        self.until = 0.0
        self.starts = []

    def start(self, samples, rate, gain):
        self.until = self.clock() + samples.size / rate
        self.starts.append(samples.size / rate)
        self.log.items.append(("speak",))

    def is_playing(self):
        return self.clock() < self.until

    def stop(self):
        self.until = 0.0
        self.log.items.append(("cancel",))


def tone(n, amp=0.3):
    t = np.arange(n) / RATE
    return (amp * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def hush(seed=0, amp=0.002):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(BLOCK) * amp * 32767).astype(np.int16)


class Rig:
    def __init__(self, router_cfg=None, turn_cfg=None, dialogue=None,
                 awake=False, **engine_kw):
        self.clock = Clock()
        self.log = OrderLog()
        self.sink = OrderSink(self.log)
        self.backend = Speaker(self.clock, self.log)
        self.router = CommandRouter(
            self.sink, config=router_cfg or RouterConfig(),
            clock=self.clock)
        self.speaker = SpeakerManager(
            self.backend, config=SpeakerConfig(), clock=self.clock)
        self.synth = FakeSynthesizer()
        self.guard = EchoGuard(clock=self.clock)
        self.turn = TurnManager(turn_cfg, clock=self.clock)
        self.wake = ScriptedRecognizer()
        self.cmd = ScriptedRecognizer()
        self.gen = ScriptedRecognizer()
        self.dialogue = dialogue or DialogueManager(
            rng=random.Random(1), now=lambda: NOW)
        self.engine = VoiceEngine(
            vad=VoiceActivityDetector(), wake_rec=self.wake,
            command_rec=self.cmd, general_rec=self.gen,
            router=self.router, speaker=self.speaker, synth=self.synth,
            guard=self.guard, turn=self.turn, dialogue=self.dialogue,
            backchannel=BackchannelManager(rng=random.Random(1)),
            config=EngineConfig(), known_locations=("kitchen",),
            clock=self.clock, **engine_kw)
        self.frame_no = 0
        self.run_silence(1.0)

        if awake:
            self.turn.wake()

    def frame(self, speech):
        self.clock.advance(0.02)
        self.frame_no += 1
        data = tone(BLOCK) if speech else hush(self.frame_no)
        self.engine.process_frame(data)

        if self.frame_no % 2 == 0:
            self.engine.tick()

    def run_silence(self, seconds):
        for _ in range(int(seconds / 0.02)):
            self.frame(False)

    def run_speech(self, seconds):
        for _ in range(int(seconds / 0.02)):
            self.frame(True)

    def say_phrase(self, seconds=1.0, tail=1.2):
        self.run_speech(seconds)
        self.run_silence(tail)

    def spoken(self):
        return [t for _, k, t in self.engine.events if k == "speak"]

    def movement_codes(self):
        return [m[0] for m in self.sink.movement]


def T(text, conf=0.9):
    return Transcript(text, conf)


# ----- engine behaviour -----

def test_wake_word_wakes_and_replies():
    rig = Rig()
    rig.wake.finals = [T("hey buddy")]
    rig.say_phrase()
    assert rig.turn.state in (S.BUDDY_SPEAKING, S.COOLDOWN, S.AWAKE)
    assert rig.turn.awake
    assert rig.spoken()[0] == "Yes?"


def test_unrelated_speech_while_sleeping_is_ignored():
    rig = Rig()
    rig.wake.finals = [T("the weather is nice")]
    rig.say_phrase()
    assert rig.turn.state is S.LISTENING_FOR_WAKE
    assert rig.spoken() == [] and rig.sink.movement == []


def test_motion_command_while_asleep_is_not_executed():
    rig = Rig()
    rig.wake.finals = [T("buddy go forward")]
    rig.say_phrase()
    assert rig.sink.movement == []


def test_spoken_stop_while_asleep_works():
    rig = Rig()
    rig.wake.finals = [T("stop")]
    rig.say_phrase()
    assert rig.movement_codes() == [cr.MOVE_STOP]
    assert rig.spoken() == ["Stopping."]


def test_forward_command_moves_and_acknowledges():
    rig = Rig(awake=True)
    rig.cmd.finals = [T("go forward")]
    rig.gen.finals = [T("go forward")]
    rig.say_phrase()
    assert cr.MOVE_FORWARD in rig.movement_codes()
    assert rig.spoken() == ["Going forward."]
    assert rig.router.moving


def test_low_confidence_command_is_not_executed():
    rig = Rig(awake=True)
    rig.cmd.finals = [T("go forward", 0.2)]
    rig.gen.finals = [T("go forward", 0.2)]
    rig.say_phrase()
    assert rig.sink.movement == []
    assert "catch that" in rig.spoken()[0]


def test_chat_goes_to_dialogue_and_is_spoken():
    rig = Rig(awake=True)
    rig.gen.finals = [T("what time is it")]
    rig.say_phrase()
    assert rig.spoken() == ["It's 3:05 PM."]


def test_empty_recognition_returns_to_awake_silently():
    rig = Rig(awake=True)
    rig.say_phrase()
    assert rig.turn.state is S.AWAKE and rig.spoken() == []


def test_go_to_sleep_stops_and_sleeps():
    rig = Rig(awake=True)
    rig.cmd.finals = [T("go forward")]
    rig.say_phrase()
    rig.cmd.finals = [T("go to sleep")]
    rig.say_phrase()
    assert rig.movement_codes()[-1] == cr.MOVE_STOP
    assert not rig.turn.awake and not rig.router.moving


def test_stop_publishes_before_cancel_and_speech():
    rig = Rig(awake=True)
    rig.cmd.finals = [T("go forward")]
    rig.say_phrase()
    rig.engine.say("a long answer that keeps going " * 4, Priority.NORMAL)
    assert rig.speaker.speaking
    rig.log.items.clear()
    rig.engine._do_stop("test")
    kinds = rig.log.items
    assert kinds[0] == ("publish", cr.MOVE_STOP)
    assert kinds.index(("cancel",)) < kinds.index(("speak",))
    assert not rig.router.moving


def test_stop_by_partial_does_not_wait_for_end_of_turn():
    rig = Rig(awake=True)
    rig.cmd.finals = [T("go forward")]
    rig.say_phrase()
    assert rig.router.moving
    rig.run_speech(0.2)
    rig.cmd.set_partial("stop")
    rig.run_speech(0.5)
    assert rig.movement_codes()[-1] == cr.MOVE_STOP
    assert not rig.router.moving


def test_stop_never_calls_dialogue_or_llm():
    class Boom:
        calls = 0

        def respond(self, text, ctx):
            Boom.calls += 1
            raise AssertionError("dialogue must not run for stop")

    rig = Rig(awake=True, dialogue=Boom())
    rig.cmd.finals = [T("buddy stop")]
    rig.gen.finals = [T("buddy stop")]
    rig.say_phrase()
    assert Boom.calls == 0
    assert rig.movement_codes() == [cr.MOVE_STOP]


def test_stop_works_even_when_tts_is_broken():
    rig = Rig(awake=True)
    rig.synth.fail = True
    rig.cmd.finals = [T("stop")]
    rig.say_phrase()
    assert rig.movement_codes() == [cr.MOVE_STOP]
    assert rig.engine.errors >= 1


def test_dialogue_exception_goes_to_error_recovery_then_recovers():
    class Bad:
        def respond(self, text, ctx):
            raise RuntimeError("model crashed")

    rig = Rig(awake=True, dialogue=Bad())
    rig.gen.finals = [T("tell me about whales")]
    rig.say_phrase(tail=0.9)
    assert rig.turn.state is S.ERROR_RECOVERY
    rig.run_silence(2.0)
    assert rig.turn.state is S.LISTENING_FOR_WAKE


def test_does_not_hear_itself_while_speaking():
    rig = Rig(awake=True)
    rig.engine.say("this is a fairly long sentence for buddy to speak "
                   "out loud to the user", Priority.NORMAL)
    assert rig.speaker.speaking
    fed_before = rig.cmd.fed + rig.gen.fed + rig.wake.fed
    leak = tone(BLOCK, amp=0.3 * 10 ** (-12 / 20))

    for _ in range(40):
        rig.clock.advance(0.02)
        rig.engine.process_frame(leak)

    assert rig.cmd.fed + rig.gen.fed + rig.wake.fed == fed_before
    assert rig.guard.barge_in_count == 0
    assert rig.speaker.speaking


def test_user_can_interrupt_buddy():
    rig = Rig(awake=True)
    rig.engine.say("this is a fairly long sentence for buddy to speak "
                   "out loud to the user and it keeps on going",
                   Priority.NORMAL)
    assert rig.turn.state is S.BUDDY_SPEAKING

    for _ in range(25):
        rig.clock.advance(0.02)
        rig.engine.process_frame(tone(BLOCK, amp=0.8))
        rig.engine.tick()

    assert not rig.speaker.speaking
    assert rig.guard.barge_in_count >= 1
    assert rig.turn.state in (S.USER_SPEAKING, S.INTERRUPTED, S.AWAKE)
    assert ("cancel",) in rig.log.items


def test_mute_stops_motion_and_ignores_audio():
    rig = Rig(awake=True)
    rig.cmd.finals = [T("go forward")]
    rig.say_phrase()
    rig.engine.set_muted(True)
    assert rig.movement_codes()[-1] == cr.MOVE_STOP
    fed = rig.wake.fed + rig.cmd.fed
    rig.run_speech(1.0)
    assert rig.wake.fed + rig.cmd.fed == fed
    rig.engine.set_muted(False)
    assert rig.turn.state is S.LISTENING_FOR_WAKE


def test_awake_timeout_stops_a_moving_robot():
    rig = Rig(turn_cfg=TurnConfig(awake_timeout_s=3.0),
              router_cfg=RouterConfig(max_motion_s=60.0), awake=True)
    rig.cmd.finals = [T("go forward")]
    rig.say_phrase(tail=1.0)
    assert rig.router.moving
    rig.run_silence(5.0)
    assert not rig.turn.awake
    assert not rig.router.moving
    assert rig.movement_codes()[-1] == cr.MOVE_STOP


def test_backchannel_in_long_utterance_pause_and_not_for_commands():
    rig = Rig(awake=True)
    rig.run_speech(0.3)
    rig.gen.set_partial("so today i went to the market and bought things")
    rig.run_speech(4.5)
    rig.run_silence(0.7)
    assert any(s.startswith(("Mm", "I see", "Right", "Okay"))
               for s in rig.spoken())

    rig2 = Rig(awake=True)
    rig2.run_speech(0.3)
    rig2.cmd.set_partial("go forward")
    rig2.run_speech(4.5)
    rig2.run_silence(0.7)
    assert rig2.spoken() == []


def test_quiet_mode_suppresses_backchannel():
    rig = Rig(awake=True)
    rig.engine.set_quiet(True)
    rig.run_speech(0.3)
    rig.gen.set_partial("so today i went to the market and bought things")
    rig.run_speech(4.5)
    rig.run_silence(0.7)
    assert rig.spoken() == []


def test_warm_cache_reuses_synthesis():
    rig = Rig()
    rig.engine.warm_cache(["Stopping."])
    n = len(rig.synth.calls)
    rig.engine.say("Stopping.", Priority.CRITICAL)
    assert len(rig.synth.calls) == n
