"""The one door to the speaker."""

from buddy_orchestrator.contracts import SpeechPriority
from buddy_orchestrator.speech import Kind, SpeechArbiter


def make():
    out, t = [], [0.0]
    arb = SpeechArbiter(out.append, clock=lambda: t[0])

    return arb, out, t


def test_priorities_follow_the_kind():
    arb, out, t = make()

    for kind in Kind:
        t[0] += 100
        arb.say(kind, f"text {kind.value}", "test")

    assert [u.priority for u in out] == [
        SpeechPriority.CRITICAL, SpeechPriority.HIGH,
        SpeechPriority.NORMAL, SpeechPriority.LOW]


def test_only_emergency_speech_interrupts():
    arb, out, t = make()

    for kind in Kind:
        t[0] += 100
        arb.say(kind, f"x {kind.value}", "test")

    assert [u.interrupt for u in out] == [True, False, False, False]


def test_empty_and_whitespace_text_is_dropped():
    arb, out, _ = make()
    assert not arb.say(Kind.STATUS, "   ", "t")
    assert out == [] and arb.dropped["empty"] == 1


def test_shutdown_silences_everything_even_emergency():
    arb, out, _ = make()
    arb.context.shutting_down = True
    assert not arb.say(Kind.EMERGENCY, "help", "t")


def test_quiet_mode_keeps_emergency_and_status_only():
    arb, out, t = make()
    arb.context.quiet_mode = True
    results = []

    for kind in Kind:
        t[0] += 100
        results.append(arb.say(kind, f"x {kind.value}", "t"))

    assert results == [True, True, False, False]


def test_proactive_waits_for_the_conversation():
    arb, out, t = make()

    for flag in ("user_speaking", "buddy_speaking",
                 "conversation_active"):
        setattr(arb.context, flag, True)
        t[0] += 100
        assert not arb.say(Kind.PROACTIVE, "hi", "t")
        setattr(arb.context, flag, False)

    assert arb.say(Kind.PROACTIVE, "hi", "t")


def test_proactive_gap_and_dedupe():
    arb, out, t = make()
    assert arb.say(Kind.PROACTIVE, "one", "t")
    t[0] += 5
    assert not arb.say(Kind.PROACTIVE, "two", "t")
    assert arb.dropped["proactive_gap"] == 1
    t[0] += 30
    assert arb.say(Kind.PROACTIVE, "two", "t")
    t[0] += 1
    assert not arb.say(Kind.STATUS, "two", "t")        # repeated words
    assert arb.dropped["duplicate"] == 1


def test_emergency_stop_blocks_proactive_not_status():
    arb, out, _ = make()
    arb.context.emergency_stop = True
    assert not arb.say(Kind.PROACTIVE, "hi", "t")
    assert arb.say(Kind.STATUS, "stopped", "t")


def test_long_text_is_trimmed():
    arb, out, _ = make()
    arb.say(Kind.STATUS, "word " * 200, "t")
    assert len(out[0].text) <= arb.policy.max_text_chars


def test_request_ids_are_unique():
    arb, out, t = make()

    for i in range(5):
        t[0] += 100
        arb.say(Kind.STATUS, f"msg {i}", "t")

    assert len({u.request_id for u in out}) == 5
