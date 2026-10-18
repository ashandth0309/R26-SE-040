"""Phase 7 tests: language policy, connectivity, LLM routing, action gate,
and non-blocking replies."""

import random
from concurrent.futures import Future

import pytest

from buddy_core.voice import command_router as cr
from buddy_core.voice.action_gate import ActionGate
from buddy_core.voice.connectivity import (
    ConnectivityConfig,
    ConnectivityManager,
    Link,
)
from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueManager,
)
from buddy_core.voice.language import (
    LanguageGuard,
    LanguagePolicy,
    detect_language,
)
from buddy_core.voice.llm_router import (
    SYSTEM_PROMPT,
    LlmRouter,
    LlmUnavailable,
    OllamaProvider,
    OpenAICompatProvider,
    ProviderEntry,
    RouterMode,
    trim_messages,
)
from buddy_core.voice.turn_manager import TurnState as S


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ----- language -----

@pytest.mark.parametrize("text,lang", [
    ("what time is it", "en"),
    ("வணக்கம் நண்பா", "ta"),
    ("vanakkam epdi irukinga", "ta_latn"),
    ("hello nandri", "ta_latn"),
    ("hello வணக்கம்", "mixed"),
    ("", "unknown"),
    ("the sea was calm", "en"),
])
def test_language_detection(text, lang):
    assert detect_language(text) == lang


def test_policy_modes():
    offline = LanguagePolicy()
    assert offline.understood == {"en"} and offline.spoken == {"en"}
    online = offline.with_online(True)
    assert "ta" in online.understood and "ta" not in online.spoken
    assert "ta" in offline.with_online(True, tamil_voice=True).spoken


def test_guard_is_honest_offline_and_silent_when_fine():
    g = LanguageGuard(LanguagePolicy())
    ctx = ConversationContext()
    reply = g.hook("வணக்கம்", ctx)
    assert "only understand English" in reply.text
    assert g.hook("hello", ctx) is None
    online = LanguageGuard(LanguagePolicy().with_online(True))
    assert online.hook("வணக்கம்", ctx) is None
    assert online.answered_in_english


# ----- connectivity -----

def conn(enabled=True, probe=None, **cfg):
    clock = Clock()
    flag = {"on": enabled}
    mgr = ConnectivityManager(
        enabled=lambda: flag["on"], probe=probe,
        config=ConnectivityConfig(**cfg), clock=clock,
        sleep=lambda s: None)

    return mgr, clock, flag


def test_disabled_means_no_requests_and_no_probing():
    calls = []
    mgr, clock, _ = conn(enabled=False, probe=lambda: calls.append(1))
    assert mgr.link is Link.DISABLED and not mgr.allow_request()
    mgr.tick()
    assert calls == []
    with pytest.raises(ConnectionError):
        mgr.call(lambda: "x")


def test_breaker_opens_backs_off_and_recovers():
    mgr, clock, _ = conn(failure_threshold=2, backoff_base_s=5)
    assert mgr.allow_request()
    mgr.record_failure()
    assert mgr.allow_request()
    mgr.record_failure()
    assert mgr.link is Link.OFFLINE and not mgr.allow_request()
    clock.t += 5.1
    assert mgr.allow_request()
    mgr.record_failure()
    clock.t += 5.1
    assert not mgr.allow_request()
    clock.t += 5
    assert mgr.allow_request()
    mgr.record_success()
    assert mgr.link is Link.ONLINE and mgr.allow_request()


def test_backoff_is_capped():
    mgr, clock, _ = conn(failure_threshold=1, backoff_base_s=10,
                         backoff_max_s=30)

    for _ in range(6):
        mgr.record_failure()

    assert mgr._next_try - clock.t <= 30


def test_call_has_bounded_attempts():
    mgr, _, _ = conn(max_attempts=3, failure_threshold=99)
    attempts = []

    def flaky():
        attempts.append(1)
        raise TimeoutError

    with pytest.raises(TimeoutError):
        mgr.call(flaky)

    assert len(attempts) == 3

    ok = iter([TimeoutError, "fine"])

    def second():
        v = next(ok)

        if v is TimeoutError:
            raise v

        return v

    assert mgr.call(second) == "fine"
    assert mgr.link is Link.ONLINE


def test_call_stops_retrying_when_breaker_opens():
    mgr, _, _ = conn(max_attempts=5, failure_threshold=2)
    attempts = []

    def bad():
        attempts.append(1)
        raise OSError

    with pytest.raises(OSError):
        mgr.call(bad)

    assert len(attempts) == 2 and mgr.link is Link.OFFLINE


def test_probe_runs_when_due_and_survives_errors():
    seq = iter([False, False, True])
    mgr, clock, _ = conn(probe=lambda: next(seq), failure_threshold=2,
                         probe_interval_s=10, backoff_base_s=1)
    mgr.tick()
    clock.t += 11
    mgr.tick()
    assert mgr.link is Link.OFFLINE
    clock.t += 11
    mgr.tick()
    assert mgr.link is Link.ONLINE

    def boom():
        raise RuntimeError

    mgr2, clock2, _ = conn(probe=boom, failure_threshold=1)
    mgr2.tick()
    assert mgr2.link is Link.OFFLINE


# ----- providers -----

class Spy:
    def __init__(self, reply="ok", fail=None, clock=None, cost=0.0):
        self.reply, self.fail, self.clock, self.cost = reply, fail, \
            clock, cost
        self.calls = []

    def complete(self, messages, timeout_s):
        self.calls.append((messages, timeout_s))

        if self.clock:
            self.clock.t += self.cost

        if self.fail:
            raise self.fail

        return self.reply


MSGS = [{"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "tell me about otters"}]


def router(local=None, cloud=None, mode=RouterMode.AUTO, cloud_on=True,
           connectivity=None):
    clock = Clock()
    entries = []

    if local is not None:
        entries.append(ProviderEntry("local", "local", local))

    if cloud is not None:
        entries.append(ProviderEntry("cloud", "cloud", cloud))

    return LlmRouter(entries, mode, connectivity,
                     cloud_enabled=lambda: cloud_on, clock=clock), clock


def test_offline_mode_calls_nothing():
    local, cloud = Spy(), Spy()
    r, _ = router(local, cloud, mode=RouterMode.OFFLINE)
    with pytest.raises(LlmUnavailable):
        r.complete(MSGS, 5)
    assert not local.calls and not cloud.calls


def test_local_mode_never_touches_cloud():
    local, cloud = Spy("local reply"), Spy("cloud reply")
    r, _ = router(local, cloud, mode=RouterMode.LOCAL)
    assert r.complete(MSGS, 5) == "local reply"
    assert not cloud.calls and r.last_provider == "local"


def test_cloud_not_used_unless_user_enabled_it():
    local, cloud = Spy(fail=OSError("down")), Spy("cloud reply")
    r, _ = router(local, cloud, cloud_on=False)
    with pytest.raises(LlmUnavailable):
        r.complete(MSGS, 5)
    assert not cloud.calls


def test_falls_back_to_cloud_with_same_context():
    local, cloud = Spy(fail=OSError("down")), Spy("cloud reply")
    r, _ = router(local, cloud)
    assert r.complete(MSGS, 5) == "cloud reply"
    assert local.calls[0][0] == cloud.calls[0][0]
    assert r.last_provider == "cloud" and "local" in r.last_errors[0]


def test_breaker_blocks_cloud_after_failures():
    mgr, clock, _ = conn(failure_threshold=1, max_attempts=1)
    cloud = Spy(fail=OSError("down"))
    r, _ = router(None, cloud, connectivity=mgr)
    with pytest.raises(LlmUnavailable):
        r.complete(MSGS, 5)
    n = len(cloud.calls)
    with pytest.raises(LlmUnavailable):
        r.complete(MSGS, 5)
    assert len(cloud.calls) == n


def test_time_budget_is_respected():
    clock = Clock()
    slow = Spy(fail=TimeoutError(), clock=clock, cost=5.0)
    fast = Spy("late")
    r = LlmRouter([ProviderEntry("local", "local", slow),
                   ProviderEntry("local2", "local", fast)],
                  RouterMode.LOCAL, clock=clock)
    with pytest.raises(LlmUnavailable):
        r.complete(MSGS, 5.0)
    assert not fast.calls
    assert slow.calls[0][1] <= 5.0


def test_trim_messages_adds_system_and_limits_size():
    many = [{"role": "user", "content": "x" * 400} for _ in range(30)]
    out = trim_messages(many)
    assert out[0]["role"] == "system" and out[0]["content"] == SYSTEM_PROMPT
    assert len(out) <= 9
    assert sum(len(m["content"]) for m in out[1:]) <= 1900


def test_ollama_request_shape_and_parse():
    seen = {}

    def transport(url, payload, headers, timeout):
        seen.update(url=url, payload=payload, timeout=timeout)

        return {"message": {"content": "woof"}}

    p = OllamaProvider("m", "http://127.0.0.1:11434/", transport)
    assert p.complete(MSGS, 3.0) == "woof"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["payload"]["stream"] is False and seen["timeout"] == 3.0


def test_cloud_adapter_needs_key_from_environment_only():
    def transport(url, payload, headers, timeout):
        assert headers["Authorization"] == "Bearer from-env"

        return {"choices": [{"message": {"content": "hi there"}}]}

    p = OpenAICompatProvider("https://example.invalid/v1", "m",
                             transport=transport, environ={})
    with pytest.raises(LlmUnavailable):
        p.complete(MSGS, 3)
    p2 = OpenAICompatProvider("https://example.invalid/v1", "m",
                              transport=transport,
                              environ={"BUDDY_LLM_API_KEY": "from-env"})
    assert p2.complete(MSGS, 3) == "hi there"


def test_no_api_key_literals_in_source():
    from pathlib import Path
    import re

    root = Path(__file__).resolve().parents[1] / "buddy_core" / "voice"
    pattern = re.compile(r"sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{20,}")

    for f in root.glob("*.py"):
        assert not pattern.search(f.read_text()), f.name


def test_dialogue_with_router_falls_back_cleanly_offline():
    r, _ = router(Spy(), Spy(), mode=RouterMode.OFFLINE)
    d = DialogueManager(llm=r, rng=random.Random(1))
    reply = d.respond("tell me about whales", ConversationContext())
    assert reply.source == "fallback" and d.llm_failures == 1


# ----- action gate -----

def gate():
    clock = Clock()

    return ActionGate(clock=clock), clock


def test_gate_turns_suggestion_into_question_and_needs_yes():
    g, _ = gate()
    q = g.propose("go forward")
    assert q == "Do you want me to go forward?"
    cmd = g.confirm("yes please")
    assert cmd.kind.value == "forward"
    assert g.confirm("yes") is None


@pytest.mark.parametrize("suggestion", [
    None, "", "stop", "start patrol", "dance wildly", "go to the moon",
    "fly",
])
def test_gate_drops_unsafe_or_unknown_suggestions(suggestion):
    g, _ = gate()
    assert g.propose(suggestion) is None
    assert g.confirm("yes") is None


def test_gate_cancels_on_no_other_talk_and_timeout():
    g, clock = gate()
    g.propose("turn left")
    assert g.confirm("no") is None and g.confirm("yes") is None
    g.propose("turn left")
    assert g.confirm("tell me a joke") is None and g.confirm("yes") is None
    g.propose("turn left")
    clock.t += 16
    assert g.confirm("yes") is None


# ----- engine: replies never block the voice loop -----

class ManualExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))

        return future

    def run_all(self):
        jobs, self.jobs = self.jobs, []

        for future, fn, args in jobs:
            if future.cancelled():
                continue

            try:
                future.set_result(fn(*args))
            except Exception as exc:
                future.set_exception(exc)


def test_stop_is_instant_while_a_reply_is_still_being_prepared():
    from test_voice_engine import Rig, T

    ex = ManualExecutor()
    rig = Rig(awake=True, executor=ex)
    rig.cmd.finals = [T("go forward")]
    rig.say_phrase()
    rig.gen.finals = [T("tell me about whales")]
    rig.say_phrase(tail=0.9)
    assert rig.turn.state is S.PROCESSING and len(ex.jobs) == 1
    rig.cmd.finals = [T("stop")]
    rig.cmd.set_partial("")
    rig.say_phrase()
    assert rig.movement_codes()[-1] == cr.MOVE_STOP
    assert not rig.router.moving
    ex.run_all()
    rig.run_silence(0.5)
    assert "I'm not sure" not in " ".join(rig.spoken())
    assert rig.spoken()[-1] == "Stopping."


def test_late_reply_after_processing_timeout_is_dropped():
    from test_voice_engine import Rig, T

    ex = ManualExecutor()
    rig = Rig(awake=True, executor=ex)
    rig.gen.finals = [T("tell me about whales")]
    rig.say_phrase(tail=0.9)
    rig.run_silence(11.0)
    assert rig.turn.state in (S.ERROR_RECOVERY, S.LISTENING_FOR_WAKE)
    before = rig.spoken()
    ex.run_all()
    rig.run_silence(0.5)
    assert rig.spoken() == before
    assert any(k == "stale_reply_dropped"
               for _, k, _ in rig.engine.events)


def test_user_speaking_again_cancels_pending_reply():
    from test_voice_engine import Rig, T

    ex = ManualExecutor()
    rig = Rig(awake=True, executor=ex)
    rig.gen.finals = [T("tell me about whales")]
    rig.say_phrase(tail=0.9)
    assert rig.turn.state is S.PROCESSING
    rig.gen.finals = [T("what time is it")]
    rig.say_phrase(tail=0.9)
    ex.run_all()
    rig.run_silence(0.3)
    spoken = rig.spoken()
    assert spoken.count("It's 3:05 PM.") == 1
    assert not any("whales" in s for s in spoken)


def test_reply_delivered_when_executor_finishes_in_time():
    from test_voice_engine import Rig, T

    ex = ManualExecutor()
    rig = Rig(awake=True, executor=ex)
    rig.gen.finals = [T("what time is it")]
    rig.say_phrase(tail=0.9)
    assert rig.spoken() == []
    ex.run_all()
    rig.run_silence(0.2)
    assert rig.spoken() == ["It's 3:05 PM."]


class SuggestingLlm:
    def complete(self, messages, timeout_s):
        return "A walk sounds nice. ACTION: go forward"


def llm_rig(llm):
    from test_voice_engine import Rig

    d = DialogueManager(llm=llm, rng=random.Random(1))

    return Rig(awake=True, dialogue=d)


def test_llm_suggestion_never_moves_robot_by_itself():
    from test_voice_engine import T

    rig = llm_rig(SuggestingLlm())
    rig.gen.finals = [T("I feel like exploring the garden today")]
    rig.say_phrase()
    assert rig.sink.movement == []
    assert any("Do you want me to go forward?" in s for s in rig.spoken())


def test_confirming_suggestion_moves_through_the_router():
    from test_voice_engine import T

    rig = llm_rig(SuggestingLlm())
    rig.gen.finals = [T("I feel like exploring the garden today")]
    rig.say_phrase()
    rig.run_silence(1.0)
    rig.cmd.finals = [T("yes")]
    rig.gen.finals = [T("yes")]
    rig.say_phrase()
    assert cr.MOVE_FORWARD in rig.movement_codes()


def test_unconfirmed_or_changed_subject_suggestion_is_dropped():
    from test_voice_engine import T

    rig = llm_rig(SuggestingLlm())
    rig.gen.finals = [T("I feel like exploring the garden today")]
    rig.say_phrase()
    rig.run_silence(1.0)
    rig.gen.finals = [T("what time is it")]
    rig.say_phrase()
    rig.run_silence(1.0)
    rig.cmd.finals = [T("yes")]
    rig.gen.finals = [T("yes")]
    rig.say_phrase()
    assert rig.sink.movement == []


def test_stop_clears_a_pending_suggestion():
    from test_voice_engine import T

    rig = llm_rig(SuggestingLlm())
    rig.gen.finals = [T("I feel like exploring the garden today")]
    rig.say_phrase()
    rig.cmd.finals = [T("stop")]
    rig.say_phrase()
    rig.cmd.finals = [T("yes")]
    rig.gen.finals = [T("yes")]
    rig.say_phrase()
    assert cr.MOVE_FORWARD not in rig.movement_codes()
