"""Phase 8: proactive interaction policy (no robot, no audio, no camera)."""

import pytest

from buddy_core.vision.multimodal import Agreement, FusedObservation
from buddy_core.vision.proactive import (
    MemoryState, PState, ProactiveConfig, ProactiveContext,
    ProactiveInteractionManager, Reply, StoreState, parse_reply)


def low(agreement=Agreement.CONSISTENT, conf=0.6, group="negative"):
    return FusedObservation(agreement, group, conf, group, None,
                            ("vision",), "t", False)


def ctx(now, **kw):
    base = dict(now=now, day=1, person_key="p1", recognized=True,
                person_present=True, observation=low(),
                observation_stamp=now, person_settings={"proactive": True})
    base.update(kw)

    return ProactiveContext(**base)


def run_until_offer(m, t0=1000.0, **kw):
    t = t0
    s = m.tick(ctx(t, **kw))

    while t - t0 < 200 and s.kind != "offer":
        t += 5
        s = m.tick(ctx(t, **kw))

    return s, t


def test_offer_only_after_sustained_signal():
    m = ProactiveInteractionManager()
    first = m.tick(ctx(1000.0))
    assert first.kind == "none" and m.reason == "not_sustained"
    s, t = run_until_offer(m)
    assert s.kind == "offer" and t - 1000.0 >= 60
    assert m.state is PState.INTERACTION_OFFERED
    assert s.text.endswith("?")
    assert not s.needs_approach


@pytest.mark.parametrize("kw,reason", [
    (dict(recognized=False), "stranger_no_invitation"),
    (dict(person_settings={}), "not_opted_in"),
    (dict(person_key=None), "no_persistent_identity"),
    (dict(quiet_mode=True), "quiet_mode"),
    (dict(someone_speaking=True), "someone_speaking"),
    (dict(robot_busy=True), "robot_busy"),
    (dict(person_busy_hint=True), "person_busy"),
    (dict(camera_fresh=False), "camera_not_fresh"),
])
def test_never_offers_when_blocked(kw, reason):
    m = ProactiveInteractionManager()
    s, _ = run_until_offer(m, **kw)
    assert s.kind != "offer" and m.offers_made == 0
    assert m.reason == reason


def test_per_person_quiet_setting_blocks():
    m = ProactiveInteractionManager()
    s, _ = run_until_offer(
        m, person_settings={"proactive": True, "quiet": True})
    assert s.kind != "offer"


@pytest.mark.parametrize("agreement", [
    Agreement.CONFLICT, Agreement.SUPPRESSED, Agreement.NONE])
def test_unreliable_agreement_never_triggers(agreement):
    m = ProactiveInteractionManager()
    s, _ = run_until_offer(m, observation=low(agreement))
    assert s.kind != "offer"


def test_non_negative_or_weak_or_stale_signal_never_triggers():
    for obs, stamp_back in ((low(group="positive"), 0),
                            (low(conf=0.1), 0),
                            (low(), 60)):
        m = ProactiveInteractionManager()
        t0 = 1000.0
        offered = False

        for i in range(60):
            t = t0 + i * 5
            s = m.tick(ctx(t, observation=obs,
                           observation_stamp=t - stamp_back))
            offered |= s.kind == "offer"

        assert not offered


def test_interrupted_signal_restarts_sustain_timer():
    m = ProactiveInteractionManager()
    m.tick(ctx(1000.0))
    m.tick(ctx(1050.0))
    m.tick(ctx(1055.0, observation=None))      # signal lost
    s = m.tick(ctx(1062.0))                    # would be 62 s since start
    assert s.kind == "none"


def test_invitation_overrides_signal_but_not_safety_blocks():
    m = ProactiveInteractionManager()
    s = m.tick(ctx(10.0, observation=None, invited=True,
                   person_settings={}))
    assert s.kind == "offer"
    m2 = ProactiveInteractionManager()
    s2 = m2.tick(ctx(10.0, invited=True, quiet_mode=True))
    assert s2.kind != "offer"


def test_emergency_suspends_everything():
    m = ProactiveInteractionManager()
    s = m.tick(ctx(10.0, emergency=True, invited=True))
    assert s.kind == "none" and m.state is PState.SUSPENDED
    s = m.tick(ctx(11.0, observation=None))
    assert m.state is not PState.SUSPENDED


def test_accept_starts_conversation_and_ends_with_cooldown():
    m = ProactiveInteractionManager()
    s, t = run_until_offer(m)
    r = m.on_reply("yes please", t + 3, 1)
    assert r.kind == "ack" and m.state is PState.CONVERSATION_ACTIVE
    m.tick(ctx(t + 10, observation=None))
    assert m.state is PState.CONVERSATION_ACTIVE
    m.conversation_finished(t + 50)
    assert m.state is PState.COOLDOWN
    s2 = m.tick(ctx(t + 60))
    assert s2.kind == "none"


def test_conversation_ends_when_person_leaves():
    m = ProactiveInteractionManager()
    s, t = run_until_offer(m)
    m.on_reply("yes", t + 1, 1)
    m.tick(ctx(t + 5, person_present=False))
    assert m.state is PState.COOLDOWN


def test_decline_sets_long_cooldown_and_no_repeat():
    m = ProactiveInteractionManager()
    s, t = run_until_offer(m)
    r = m.on_reply("no thanks", t + 2, 1)
    assert r.kind == "ack" and m.state is PState.USER_DECLINED
    offered = 0

    for i in range(1, 600):          # 50 minutes of continued low signal
        if m.tick(ctx(t + i * 5)).kind == "offer":
            offered += 1

    assert offered == 0


def test_cooldown_survives_a_new_manager_via_backend():
    be = MemoryState()
    m = ProactiveInteractionManager(backend=be)
    s, t = run_until_offer(m)
    m.on_reply("no", t + 1, 1)
    m2 = ProactiveInteractionManager(backend=be)
    s2, _ = run_until_offer(m2, t0=t + 120)
    assert s2.kind != "offer"


def test_second_decline_same_day_waits_longer():
    cfg = ProactiveConfig(sustain_s=1.0, min_gap_s=0.0,
                          decline_cooldown_s=100.0,
                          repeat_decline_cooldown_s=1000.0,
                          default_max_per_day=9)
    m = ProactiveInteractionManager(cfg)
    s, t = run_until_offer(m, t0=0.0)
    m.on_reply("no", t, 1)
    s, t2 = run_until_offer(m, t0=t + 150.0)
    assert s.kind == "offer"
    m.on_reply("no", t2, 1)
    s, _ = run_until_offer(m, t0=t2 + 150.0)
    assert s.kind != "offer"


def test_stop_asking_calls_back_and_blocks_for_long():
    called = []
    m = ProactiveInteractionManager(on_stop_asking=called.append)
    s, t = run_until_offer(m)
    r = m.on_reply("please stop asking me that", t + 1, 1)
    assert called == ["p1"] and "won't" in r.text
    assert m._get("p1", "cooldown_until") > t + 86400 * 20


def test_space_request_uses_space_cooldown():
    m = ProactiveInteractionManager()
    s, t = run_until_offer(m)
    r = m.on_reply("I need some space", t + 1, 1)
    assert "space" in r.text.lower()
    assert m._get("p1", "cooldown_until") >= t + 4 * 3600


def test_silence_counts_as_no():
    m = ProactiveInteractionManager()
    s, t = run_until_offer(m)
    m.tick(ctx(t + 30))
    assert m.state is PState.COOLDOWN and m.reason == "no_response"
    assert m.tick(ctx(t + 40)).kind == "none"


def test_unclear_answer_is_not_acceptance():
    m = ProactiveInteractionManager()
    s, t = run_until_offer(m)
    m.on_reply("hmm what is for dinner", t + 1, 1)
    assert m.state is PState.COOLDOWN


def test_daily_limit_and_gap():
    cfg = ProactiveConfig(sustain_s=1.0, min_gap_s=100.0,
                          silence_cooldown_s=1.0, default_max_per_day=2)
    m = ProactiveInteractionManager(cfg)
    offers = 0

    for i in range(400):
        t = i * 5.0
        s = m.tick(ctx(t))
        offers += s.kind == "offer"

    assert offers == 2
    m2 = ProactiveInteractionManager(cfg)
    offers = 0

    for i in range(400):
        t = i * 5.0
        offers += m2.tick(ctx(t, day=1 + i // 100)).kind == "offer"

    assert offers > 2           # a new day resets the count


def test_per_person_limit_setting():
    cfg = ProactiveConfig(sustain_s=1.0, min_gap_s=0.0,
                          silence_cooldown_s=1.0)
    m = ProactiveInteractionManager(cfg)
    offers = 0

    for i in range(200):
        offers += m.tick(ctx(i * 5.0, person_settings={
            "proactive": True, "max_checkins_per_day": 1})
        ).kind == "offer"

    assert offers == 1


def test_people_have_independent_cooldowns():
    be = MemoryState()
    m = ProactiveInteractionManager(backend=be)
    s, t = run_until_offer(m)
    m.on_reply("no", t + 1, 1)
    s2, _ = run_until_offer(m, t0=t + 200, person_key="p2")
    assert s2.kind == "offer" and s2.person_key == "p2"


def test_offer_wording_never_states_an_emotion():
    m = ProactiveInteractionManager()
    seen = set()

    for k in range(6):
        s, t = run_until_offer(m, t0=k * 100000.0, person_key=f"q{k}")
        seen.add(s.text)
        low_words = ("you are sad", "you're sad", "you are angry",
                     "you look sad", "you're upset")
        assert not any(w in s.text.lower() for w in low_words)

    assert len(seen) > 1


@pytest.mark.parametrize("text,expected", [
    ("yes please", Reply.ACCEPT), ("sure", Reply.ACCEPT),
    ("no", Reply.DECLINE), ("I'm fine thanks", Reply.DECLINE),
    ("no thanks", Reply.DECLINE),
    ("not now", Reply.SPACE), ("leave me alone", Reply.SPACE),
    ("stop asking", Reply.STOP_ASKING),
    ("don't ask me that again", Reply.STOP_ASKING),
    ("no, I'm fine, yes sure", Reply.DECLINE),
    ("", Reply.UNCLEAR), ("banana", Reply.UNCLEAR),
])
def test_parse_reply(text, expected):
    assert parse_reply(text) is expected


def test_reset_person_forgets_open_offer():
    m = ProactiveInteractionManager()
    s, t = run_until_offer(m)
    m.reset_person("p1")
    assert m.state is PState.IDLE
    assert m.on_reply("yes", t + 1, 1).kind == "none"


def test_store_backed_state_and_cascade_delete(tmp_path):
    import time

    import numpy as np

    from buddy_core.vision.face_store import (
        Consent, FaceProfileStore, StoreError)
    store = FaceProfileStore(str(tmp_path / "f.db"))
    pid = store.add_person("Ann", Consent(time.time(), "spoken_yes", True))
    store.add_embedding(pid, np.ones(128, np.float32) / 11.3)
    st = StoreState(store)
    st.set(pid, "cooldown_until", 123.0)
    assert st.get(pid, "cooldown_until") == 123.0
    store.delete_person(pid)
    assert store.kv_get(pid, "cooldown_until", None) is None

    with pytest.raises(StoreError):
        st.set(pid, "x", 1)
