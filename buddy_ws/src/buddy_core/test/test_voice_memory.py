"""Phase 6 tests: memory, consent, controls, personality, proactive."""

import json
import random
import threading

import pytest

from buddy_core.voice.companion_commands import (
    CompanionControls,
    SilenceMode,
    bind_engine,
)
from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueManager,
)
from buddy_core.voice.memory_store import (
    CompanionMemoryManager,
    MemoryConfig,
    looks_sensitive,
)
from buddy_core.voice.personality import (
    BuddyPersonalityEngine,
    PersonalityConfig,
    load_personality,
    similarity,
)
from buddy_core.voice.proactive import (
    CATALOG,
    ProactiveConfig,
    ProactiveSuggester,
    SuggestContext,
)
from buddy_core.voice.speaker_manager import Priority


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ----- memory -----

def test_nothing_saved_without_consent_but_session_works(tmp_path):
    db = tmp_path / "m.db"
    m = CompanionMemoryManager(db)
    res = m.remember("name", "Yohan", "name")
    assert not res.ok and res.reason == "no consent"
    assert m.recall("name") == "Yohan"
    assert m.all_facts() == []
    m.close()
    assert CompanionMemoryManager(db).recall("name") is None


def test_consented_memory_persists(tmp_path):
    db = tmp_path / "m.db"
    m = CompanionMemoryManager(db)
    m.grant_consent()
    assert m.remember("name", "Yohan", "name").ok
    m.close()
    again = CompanionMemoryManager(db)
    assert again.consent and again.recall("name") == "Yohan"


@pytest.mark.parametrize("value", [
    "my password is hunter2", "card 4111 1111 1111 1111",
    "4111-1111-1111-1111", "the pin number is 1234", "api key abc",
])
def test_credentials_and_payment_refused_even_with_consent(value):
    m = CompanionMemoryManager()
    m.grant_consent()
    res = m.remember("secret", value)
    assert not res.ok and res.reason == "sensitive"
    assert m.all_facts() == []
    assert looks_sensitive(value)


def test_ordinary_numbers_are_fine():
    m = CompanionMemoryManager()
    m.grant_consent()
    assert m.remember("favourite_number", "my lucky number is 42").ok


def test_limits():
    m = CompanionMemoryManager(config=MemoryConfig(
        max_facts=2, max_value_chars=10, max_notes=1))
    m.grant_consent()
    assert m.remember("a", "x").ok
    assert not m.remember("b", "this is far too long").ok
    assert m.remember("b", "short").ok
    assert m.remember("c", "third").reason == "memory full"
    assert m.remember("a", "updated").ok

    n = CompanionMemoryManager(config=MemoryConfig(max_notes=1))
    n.grant_consent()
    assert n.add_note("first").ok
    assert n.add_note("second").reason == "too many notes"


def test_correct_forget_and_forget_all():
    m = CompanionMemoryManager()
    m.grant_consent()
    m.remember("name", "Yohan")
    m.correct("name", "Yohan Perera")
    assert m.recall("name") == "Yohan Perera"
    m.remember("pet", "dog")
    assert m.forget("pet") and m.recall("pet") is None
    assert not m.forget("pet")
    m.remember("a", "1")
    assert m.forget_all() == 2
    assert m.all_facts() == [] and m.recall("name") is None


def test_deleted_data_is_gone_from_the_file(tmp_path):
    db = tmp_path / "m.db"
    m = CompanionMemoryManager(db)
    m.grant_consent()
    marker = "zebra-marker-7781"
    m.remember("note_x", marker, "note")
    m.close()
    assert marker.encode() in db.read_bytes()
    m2 = CompanionMemoryManager(db)
    m2.forget("note_x")
    m2.close()
    assert marker.encode() not in db.read_bytes()


def test_revoking_consent_keeps_data_but_blocks_new_saves():
    m = CompanionMemoryManager()
    m.grant_consent()
    m.remember("name", "Yohan")
    m.revoke_consent()
    assert not m.remember("pet", "dog").ok
    assert m.recall("name") == "Yohan"


def test_disabled_memory_stores_nothing():
    m = CompanionMemoryManager(config=MemoryConfig(enabled=False))
    m.grant_consent()
    res = m.remember("name", "Yohan")
    assert res.reason == "memory disabled" and m.all_facts() == []
    assert not m.consent


def test_export_and_describe(tmp_path):
    m = CompanionMemoryManager()
    assert "not remembering" in m.describe()
    m.remember("name", "Yohan")
    assert "won't keep" in m.describe()
    m.grant_consent()
    m.remember("name", "Yohan")
    assert "Yohan" in m.describe()
    path = m.export_to(tmp_path / "out" / "x.json")
    data = json.loads(path.read_text())
    assert data["consent"] and data["facts"][0]["value"] == "Yohan"


def test_corrupt_database_is_moved_aside(tmp_path):
    db = tmp_path / "m.db"
    db.write_bytes(b"this is not a sqlite file" * 50)
    m = CompanionMemoryManager(db)
    assert m.degraded and m.recovered_from
    m.grant_consent()
    assert m.remember("name", "Yohan").ok
    assert list(tmp_path.glob("m.db.corrupt-*"))


def test_old_notes_expire_but_facts_stay():
    clock = Clock()
    m = CompanionMemoryManager(
        config=MemoryConfig(note_retention_days=10), clock=clock)
    m.grant_consent()
    m.remember("name", "Yohan")
    m.add_note("old note")
    clock.t += 11 * 86400
    m.add_note("new note")
    assert m.purge_expired() == 1
    keys = {f.category for f in m.all_facts()}
    assert "name" not in keys and len(m.all_facts()) == 2


def test_concurrent_writes_do_not_crash(tmp_path):
    m = CompanionMemoryManager(tmp_path / "m.db")
    m.grant_consent()
    errors = []

    def work(i):
        try:
            for j in range(10):
                m.remember(f"k{i}_{j}", "v")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(m.all_facts()) == 40


# ----- controls -----

class Probe:
    def __init__(self):
        self.muted = []
        self.silence = []
        self.cloud = []


def controls(consent=False, cloud_available=False, tmp_path=None):
    clock = Clock()
    mem = CompanionMemoryManager()

    if consent:
        mem.grant_consent()

    probe = Probe()
    c = CompanionControls(
        mem, set_muted=probe.muted.append, set_silence=probe.silence.append,
        set_cloud=probe.cloud.append,
        cloud_available=lambda: cloud_available,
        export_dir=(tmp_path or __import__("pathlib").Path("/tmp/bx")),
        clock=clock)

    return c, mem, probe, clock, ConversationContext()


def say(c, ctx, text):
    return c.hook(text, ctx)


def test_name_without_consent_asks_then_saves_on_yes():
    c, mem, _, _, ctx = controls()
    reply = say(c, ctx, "my name is Yohan")
    assert "remember that for next time" in reply.text
    assert ctx.user_name == "Yohan" and mem.all_facts() == []
    reply = say(c, ctx, "yes")
    assert mem.consent and mem.recall("name") == "Yohan"
    assert "remember" in reply.text.lower()


def test_name_declined_is_not_saved():
    c, mem, _, _, ctx = controls()
    say(c, ctx, "call me Yohan")
    say(c, ctx, "no")
    assert not mem.consent and mem.all_facts() == []


def test_confirmation_expires_and_unrelated_talk_cancels():
    c, mem, _, clock, ctx = controls()
    say(c, ctx, "my name is Yohan")
    clock.t += 31
    assert say(c, ctx, "yes") is None and not mem.consent
    say(c, ctx, "my name is Yohan")
    assert say(c, ctx, "what time is it") is None
    assert say(c, ctx, "yes") is None and not mem.consent


def test_consented_name_saved_corrected_and_recalled():
    c, mem, _, _, ctx = controls(consent=True)
    assert "I'll remember" in say(c, ctx, "My name is Yohan.").text
    say(c, ctx, "no my name is Yohan Perera")
    assert "Yohan Perera" in say(c, ctx, "what's my name").text
    assert ctx.user_name == "Yohan Perera"


def test_unclear_names_rejected_and_unknown_name_answer():
    c, _, _, _, ctx = controls(consent=True)
    assert "didn't catch" in say(c, ctx, "call me later").text
    assert "haven't told me" in say(c, ctx, "what's my name").text


def test_notes_sensitive_and_full():
    c, mem, _, _, ctx = controls(consent=True)
    assert "remember" in say(c, ctx, "remember that I like green tea").text
    assert "can't keep passwords" in say(
        c, ctx, "remember that my password is hunter2").text
    assert any("green tea" in f.value for f in mem.all_facts())


def test_note_without_consent_asks_first():
    c, mem, _, _, ctx = controls()
    assert "save that" in say(c, ctx, "remember that I like tea").text
    assert mem.all_facts() == []
    say(c, ctx, "yes")
    assert len(mem.all_facts()) == 1


def test_forget_name_and_forget_everything_needs_confirmation():
    c, mem, _, _, ctx = controls(consent=True)
    say(c, ctx, "my name is Yohan")
    say(c, ctx, "remember that I like tea")
    assert "forgotten" in say(c, ctx, "forget my name").text
    assert ctx.user_name is None and mem.recall("name") is None
    ask = say(c, ctx, "forget everything")
    assert "Are you sure" in ask.text and len(mem.all_facts()) == 1
    assert "won't" in say(c, ctx, "no").text and len(mem.all_facts()) == 1
    say(c, ctx, "delete my data")
    assert "deleted everything" in say(c, ctx, "yes").text
    assert mem.all_facts() == []


def test_stop_remembering_and_grant_consent():
    c, mem, _, _, ctx = controls(consent=True)
    assert "won't save anything new" in say(c, ctx,
                                            "stop remembering").text
    assert not mem.consent
    say(c, ctx, "you can remember things about me")
    assert mem.consent


def test_describe_and_export(tmp_path):
    c, mem, _, _, ctx = controls(consent=True, tmp_path=tmp_path)
    say(c, ctx, "my name is Yohan")
    assert "Yohan" in say(c, ctx, "what do you remember about me").text
    reply = say(c, ctx, "export my data")
    assert "buddy_memory.json" in reply.text
    assert (tmp_path / "buddy_memory.json").exists()


def test_mute_confirms_first_then_mutes_after_delay():
    c, _, probe, clock, ctx = controls()
    assert "muting" in say(c, ctx, "mute yourself").text
    c.tick()
    assert probe.muted == []
    clock.t += 3
    c.tick()
    assert probe.muted == [True]


def test_quiet_silent_normal_and_timer():
    c, _, probe, clock, ctx = controls()
    say(c, ctx, "be quiet")
    assert probe.silence == [SilenceMode.QUIET]
    say(c, ctx, "be silent for 2 minutes")
    assert probe.silence == [SilenceMode.QUIET]
    clock.t += 3
    c.tick()
    assert probe.silence[-1] is SilenceMode.SILENT
    clock.t += 120
    c.tick()
    assert probe.silence[-1] is SilenceMode.NORMAL
    say(c, ctx, "be quiet")
    say(c, ctx, "you can talk now")
    assert probe.silence[-1] is SilenceMode.NORMAL


def test_cloud_defaults_off_and_needs_confirmation():
    c, _, probe, _, ctx = controls(cloud_available=False)
    assert "isn't set up" in say(c, ctx, "go online").text
    assert probe.cloud == []
    c, _, probe, _, ctx = controls(cloud_available=True)
    assert "Say yes" in say(c, ctx, "go online").text
    assert probe.cloud == []
    say(c, ctx, "yes")
    assert probe.cloud == [True]
    say(c, ctx, "go offline")
    assert probe.cloud == [True, False]


def test_recording_question_is_honest():
    c, _, _, _, ctx = controls()
    assert "don't save recordings" in say(c, ctx, "are you recording").text
    c.cloud_enabled = lambda: True
    assert "sent to an online service" in say(
        c, ctx, "are you recording").text


def test_ordinary_talk_is_not_intercepted():
    c, _, _, _, ctx = controls()

    for text in ("hello", "what time is it", "my day was long",
                 "I'm tired", "remember to breathe"):
        assert say(c, ctx, text) is None


def test_hooks_work_inside_dialogue_manager():
    c, mem, _, _, ctx = controls(consent=True)
    d = DialogueManager(hooks=[c.hook], rng=random.Random(1))
    assert "Nice to meet you" in d.respond("my name is Asha", ctx).text
    assert "Hello Asha" in d.respond("hello", ctx).text or \
        "Asha" in d.respond("hello", ctx).text


# ----- engine binding -----

def test_silent_mode_blocks_chat_but_not_command_acks_or_stop():
    from test_voice_engine import Rig, T

    rig = Rig(awake=True)
    c, _, _, _, _ = controls()
    bind_engine(c, rig.engine)
    rig.dialogue.hooks.append(c.hook)
    rig.gen.finals = [T("be silent")]
    rig.say_phrase()
    assert any("going silent" in s for s in rig.spoken())
    rig.run_silence(3.0)
    assert rig.speaker.floor is Priority.HIGH
    before = len(rig.spoken())
    rig.gen.finals = [T("what time is it")]
    rig.say_phrase()
    assert len(rig.spoken()) == before
    rig.cmd.finals = [T("go forward")]
    rig.gen.finals = [T("go forward")]
    rig.say_phrase()
    assert "Going forward." in rig.spoken()
    rig.cmd.finals = [T("stop")]
    rig.say_phrase()
    assert rig.spoken()[-1] == "Stopping."


def test_mute_command_mutes_engine_and_stops_motion():
    from test_voice_engine import Rig, T
    from buddy_core.voice import command_router as cr

    rig = Rig(awake=True)
    c, _, _, _, _ = controls()
    bind_engine(c, rig.engine)
    rig.dialogue.hooks.append(c.hook)
    rig.cmd.finals = [T("go forward")]
    rig.say_phrase()
    rig.gen.finals = [T("mute yourself")]
    rig.say_phrase()
    rig.run_silence(4.0)
    assert rig.engine.muted
    assert rig.movement_codes()[-1] == cr.MOVE_STOP


# ----- speaker floor -----

def test_speaker_floor_filters_priorities():
    from test_voice_audio import make_speaker, pcm, RATE

    sm, be, clock = make_speaker()
    sm.set_floor(Priority.HIGH)
    assert sm.say("chat", pcm(0.2), RATE, priority=Priority.NORMAL) is None
    assert sm.say("ack", pcm(0.2), RATE, priority=Priority.HIGH) is not None
    assert sm.say("stop", pcm(0.2), RATE, priority=Priority.CRITICAL)
    sm.set_floor(Priority.LOW)
    assert sm.say("chat", pcm(0.2), RATE) is not None


# ----- personality -----

def test_personality_yaml_loading(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("warmth: 5\nverbosity: 0.2\nname: Rex\nmax_sentences: x\n"
                 "unknown_key: 1\n")
    cfg = load_personality(f)
    assert cfg.warmth == 1.0 and cfg.verbosity == 0.2
    assert cfg.name == "Rex" and cfg.max_sentences == 3
    assert load_personality(None) == PersonalityConfig()
    assert load_personality(tmp_path / "missing.yaml") == PersonalityConfig()


def test_shipped_personality_loads():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "config" / \
        "personality.yaml"
    assert load_personality(path).name == "Buddy"


def test_similarity_and_repetition_detection():
    assert similarity("I am fine today", "i am fine today") == 1.0
    assert similarity("a b c", "x y z") == 0.0
    p = BuddyPersonalityEngine(rng=random.Random(1))
    first = p.finish("I'm doing well, thanks for asking.",
                     allow_opener=False)
    assert p.is_repetitive(first)
    assert not p.is_repetitive("The weather looks lovely outside today.")


def test_repeat_replaced_by_alternative():
    p = BuddyPersonalityEngine(rng=random.Random(1))
    p.finish("I'm doing well, thanks for asking.", allow_opener=False)
    out = p.finish("I'm doing well, thanks for asking.",
                   allow_opener=False,
                   alternatives=["All good here, how about you?"])
    assert out == "All good here, how about you?"


def test_verbosity_limits_sentences():
    long = "One. Two. Three. Four. Five. Six."
    short = BuddyPersonalityEngine(PersonalityConfig(verbosity=0.0))
    chatty = BuddyPersonalityEngine(PersonalityConfig(verbosity=1.0))
    assert short.shorten(long).count(".") < chatty.shorten(long).count(".")
    assert short.shorten(long) != long


def test_openers_light_and_never_double():
    p = BuddyPersonalityEngine(
        PersonalityConfig(opener_chance=1.0), rng=random.Random(1))
    out = p.finish("The capital of that country is quite far away.")
    assert out.split()[0] in ("Sure.", "Okay.", "Alright.", "Right.",
                              "Of course.")
    assert out.endswith("far away.")
    assert p.finish("Okay. That is already started properly now.") \
        .startswith("Okay.")
    assert p.finish("Hi.", allow_opener=True) == "Hi."


# ----- proactive -----

def ctx(**kw):
    base = dict(hour=15, awake_minutes=100, emotion=None)
    base.update(kw)

    return SuggestContext(**base)


def suggester(**kw):
    clock = Clock()
    s = ProactiveSuggester(rng=random.Random(1), clock=clock, **kw)

    return s, clock


def test_suggests_when_conditions_met_only():
    s, _ = suggester()
    assert s.maybe_suggest(ctx(awake_minutes=5)) is None
    assert s.maybe_suggest(ctx()) is not None


@pytest.mark.parametrize("kw", [
    dict(silenced=True), dict(muted=True), dict(busy=True),
    dict(seconds_since_user_spoke=5),
])
def test_never_interrupts(kw):
    s, _ = suggester()
    assert s.maybe_suggest(ctx(**kw)) is None


def test_gap_and_daily_cap():
    s, clock = suggester(config=ProactiveConfig(min_gap_s=100,
                                                max_per_day=2))
    assert s.maybe_suggest(ctx()) is not None
    clock.t += 50
    assert s.maybe_suggest(ctx()) is None
    clock.t += 60
    assert s.maybe_suggest(ctx()) is not None
    clock.t += 200
    assert s.maybe_suggest(ctx()) is None
    assert s.maybe_suggest(ctx(), day=1) is not None


def test_refusal_snoozes_and_never_blocks():
    s, clock = suggester(config=ProactiveConfig(min_gap_s=1))
    first = s.maybe_suggest(ctx(awake_minutes=70))
    assert s.hook("not now", ConversationContext()).intent == "snoozed"
    clock.t += 5
    assert s.maybe_suggest(ctx(awake_minutes=70)) is None
    clock.t += 86400
    again = s.maybe_suggest(ctx(awake_minutes=70))
    assert again.sid == first.sid
    reply = s.hook("don't suggest that again", ConversationContext())
    assert reply.intent == "blocked"
    clock.t += 90000
    assert s.maybe_suggest(ctx(awake_minutes=70)) is None


def test_acceptance_recorded_and_window_expires():
    s, clock = suggester()
    sug = s.maybe_suggest(ctx())
    assert s.hook("yes", ConversationContext()).intent == "accepted"
    assert s.accepted == [sug.sid]
    s.maybe_suggest(ctx(), day=1)
    clock.t += 120
    assert s.hook("yes", ConversationContext()) is None


def test_permanent_block_persists_only_with_consent():
    mem = CompanionMemoryManager()
    mem.grant_consent()
    s, _ = suggester(memory=mem)
    s.maybe_suggest(ctx(awake_minutes=70))
    s.hook("don't suggest that again", ConversationContext())
    s2, _ = suggester(memory=mem)
    assert s2.maybe_suggest(ctx(awake_minutes=70)) is None

    mem2 = CompanionMemoryManager()
    s3, _ = suggester(memory=mem2)
    s3.maybe_suggest(ctx(awake_minutes=70))
    s3.hook("don't suggest that again", ConversationContext())
    s4, _ = suggester(memory=CompanionMemoryManager())
    assert s4.maybe_suggest(ctx(awake_minutes=70)) is not None


def test_suggestions_never_ask_for_robot_motion():
    banned = ("forward", "patrol", "turn", "move", "go to")

    for s in CATALOG:
        assert not any(w in s.text.lower() for w in banned)
