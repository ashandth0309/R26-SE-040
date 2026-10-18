"""Face store, consent-based enrollment, corrections, greeting policy."""

import os
import stat

import numpy as np
import pytest

from buddy_core.vision.embedding import l2_normalize
from buddy_core.vision.enrollment import (
    Consent3,
    EnrollmentConfig,
    EnrollmentManager,
    EnrollState,
    FaceSample,
    parse_consent,
)
from buddy_core.vision.face_store import (
    Consent,
    ConsentError,
    FaceProfileStore,
    StoreError,
    clean_name,
)
from buddy_core.vision.greeting import GreetingConfig, GreetingPolicy
from buddy_core.vision.identity import IdentityState, MatchConfig, match

DIM = 128


def vec(seed, noise=0.0, noise_seed=None):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM)
    if noise:
        ns = seed + 1000 if noise_seed is None else 5000 + noise_seed
        v = v + noise * np.random.default_rng(ns).standard_normal(DIM)

    return l2_normalize(v)


OK = Consent(1000.0, "spoken_yes", True, "remember your face?")


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# ---------- consent parsing ----------

@pytest.mark.parametrize("text,expected", [
    ("yes", Consent3.YES), ("Yes, please.", Consent3.YES),
    ("yeah sure", Consent3.YES), ("okay", Consent3.YES),
    ("yes I do", Consent3.YES), ("I agree", Consent3.YES),
    ("yes Buddy", Consent3.YES), ("go ahead", Consent3.YES),
    ("no", Consent3.NO), ("No thanks", Consent3.NO),
    ("not really", Consent3.NO), ("don't", Consent3.NO),
    ("yes but not now", Consent3.NO), ("cancel", Consent3.NO),
    ("yes they do", Consent3.THIRD_PARTY),
    ("she says yes", Consent3.THIRD_PARTY),
    ("my friend is fine with it", Consent3.THIRD_PARTY),
    ("he wants that", Consent3.THIRD_PARTY),
    ("maybe", Consent3.UNCLEAR), ("", Consent3.UNCLEAR),
    ("what is the weather", Consent3.UNCLEAR),
])
def test_consent_parser(text, expected):
    assert parse_consent(text) == expected


# ---------- store ----------

def make_store(tmp_path=None, clock=None, **kw):
    path = ":memory:" if tmp_path is None else str(tmp_path / "f" / "x.db")

    return FaceProfileStore(path, clock=clock or (lambda: 5000.0), **kw)


def test_no_person_without_consent_from_the_subject():
    store = make_store()
    with pytest.raises(ConsentError):
        store.add_person("Ann", Consent(1.0, "spoken_yes", False))
    with pytest.raises(ConsentError):
        store.add_person("Ann", None)
    assert store.list_people() == []


def test_add_person_embeddings_and_gallery():
    store = make_store()
    pid = store.add_person("ann lee", OK)
    for i in range(3):
        store.add_embedding(pid, vec(1, 0.05 * (i + 1)))
    g = store.gallery()
    assert list(g) == [pid] and g[pid].shape == (3, DIM)
    assert store.get_person(pid)["display_name"] == "Ann Lee"
    assert store.get_person(pid)["consent_method"] == "spoken_yes"


def test_names_are_cleaned_unique_and_limits_enforced():
    assert clean_name("  o'brien;;  ") == "O'Brien"
    store = make_store(max_people=2)
    store.add_person("Ann", OK)
    with pytest.raises(StoreError, match="already"):
        store.add_person("ann", OK)
    with pytest.raises(StoreError, match="name"):
        store.add_person("!!!", OK)
    store.add_person("Bo", OK)
    with pytest.raises(StoreError, match="limit"):
        store.add_person("Cy", OK)


def test_wrong_size_embedding_is_rejected():
    store = make_store()
    pid = store.add_person("Ann", OK)
    with pytest.raises(StoreError):
        store.add_embedding(pid, np.ones(5))
    with pytest.raises(StoreError):
        store.add_embedding("nobody", vec(1))


def test_template_cap_drops_oldest():
    store = make_store(max_templates=3)
    pid = store.add_person("Ann", OK)
    for i in range(5):
        store.add_embedding(pid, vec(i))
    assert store.template_count(pid) == 3


def test_recognition_switch_removes_person_from_gallery():
    store = make_store()
    pid = store.add_person("Ann", OK)
    store.add_embedding(pid, vec(1))
    assert pid in store.gallery()
    store.set_setting(pid, "recognition", False)
    assert store.gallery() == {}
    with pytest.raises(StoreError):
        store.set_setting(pid, "nonsense", 1)


def test_delete_person_removes_biometrics_from_the_file(tmp_path):
    store = make_store(tmp_path)
    pid = store.add_person("Ann", OK)
    secret = vec(42)
    store.add_embedding(pid, secret)
    store.add_negative(pid, vec(43))
    blob = secret.astype(np.float32).tobytes()
    store._db.commit()
    assert blob in open(store.path, "rb").read()
    assert store.delete_person(pid)
    assert blob not in open(store.path, "rb").read()
    assert store.gallery() == {} and store.negatives() == {}
    assert store.get_person(pid) is None and not store.delete_person(pid)
    assert all(row["person_id"] is None for row in store.audit_log())


def test_delete_all_wipes_everything(tmp_path):
    store = make_store(tmp_path)
    for n in ("Ann", "Bo"):
        pid = store.add_person(n, OK)
        store.add_embedding(pid, vec(hash(n) % 1000))
    assert store.delete_all() == 2
    assert store.list_people() == [] and store.gallery() == {}
    assert [r["event"] for r in store.audit_log()] == ["deleted_all"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_database_file_is_owner_only(tmp_path):
    store = make_store(tmp_path)
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600
    assert stat.S_IMODE(os.stat(os.path.dirname(store.path)).st_mode) \
        & 0o077 == 0


def test_export_has_no_vectors_unless_asked():
    store = make_store()
    pid = store.add_person("Ann", OK)
    store.add_embedding(pid, vec(1))
    assert "vectors" not in store.export_person(pid)
    assert len(store.export_person(pid, include_vectors=True)["vectors"]
               ) == 1
    assert store.export_person("nope") is None


def test_retention_purges_people_not_seen_for_a_long_time():
    clock = Clock(1000.0)
    store = make_store(clock=clock)
    old = store.add_person("Old", OK)
    new = store.add_person("New", OK)
    store.add_embedding(old, vec(1))
    store.mark_seen(old, 1010.0)
    store.mark_seen(new, 9000.0)
    assert store.purge_not_seen_since(5000.0) == 1
    assert store.get_person(old) is None and store.get_person(new)


def test_rename_cannot_collide():
    store = make_store()
    a, b = store.add_person("Ann", OK), store.add_person("Bo", OK)
    with pytest.raises(StoreError):
        store.rename(b, "ann")
    store.rename(a, "Anna")
    assert store.get_person(a)["display_name"] == "Anna"


# ---------- corrections ----------

def test_user_correction_blocks_that_face_from_matching_the_person():
    store = make_store()
    pid = store.add_person("Ann", OK)
    ann = vec(1)
    store.add_embedding(pid, ann)
    lookalike = l2_normalize(ann + 1.2 * vec(2))
    cfg = MatchConfig(accept=0.5, floor=0.3, margin=0.0)
    assert match(lookalike, store.gallery(), cfg).person_id == pid
    store.add_negative(pid, lookalike)
    result = match(lookalike, store.gallery(), cfg, store.negatives())
    assert result.state is IdentityState.UNKNOWN and result.person_id is None
    assert match(ann, store.gallery(), cfg,
                 store.negatives()).person_id == pid


def test_correction_that_would_lock_out_the_real_person_is_refused():
    store = make_store()
    pid = store.add_person("Ann", OK)
    ann = vec(1)
    store.add_embedding(pid, ann)
    with pytest.raises(StoreError, match="almost the same"):
        store.add_negative(pid, ann)
    assert store.negatives() == {}


# ---------- enrollment ----------

def make_mgr(store=None, **cfg):
    clock = Clock()
    store = store or make_store()
    mgr = EnrollmentManager(store, EnrollmentConfig(**cfg), clock=clock,
                            wall=lambda: 7000.0)

    return mgr, store, clock


def sample(v, tid=1, faces=1, usable=True, yaw=0.0):
    return FaceSample(tid, v, usable, yaw, faces, 0.9)


def feed(mgr, clock, base_seed, yaws, tid=1):
    reply = None
    for i, yaw in enumerate(yaws):
        clock.advance(0.5)
        r = mgr.on_face(sample(vec(base_seed, 0.6, i), tid, yaw=yaw))
        reply = r or reply

    return reply


def test_full_happy_path_saves_only_after_final_yes():
    mgr, store, clock = make_mgr()
    r = mgr.begin("sam", 1, 1)
    assert r.state is EnrollState.AWAITING_CONSENT and "yourself" in r.say
    assert store.list_people() == []
    assert mgr.on_utterance("yes").state is EnrollState.COLLECTING
    assert store.list_people() == []
    r = feed(mgr, clock, 7, [-0.4, -0.3, 0.0, 0.0, 0.3, 0.4])
    assert r.state is EnrollState.CONFIRMING and store.list_people() == []
    done = mgr.on_utterance("yes")
    assert done.done and done.state is EnrollState.DONE
    person = store.get_person(done.person_id)
    assert person["display_name"] == "Sam"
    assert person["templates"] >= 5 and person["consent_at"] == 7000.0


def test_refusal_at_consent_saves_nothing():
    mgr, store, _ = make_mgr()
    mgr.begin("Sam", 1, 1)
    assert mgr.on_utterance("no thanks").state is EnrollState.CANCELLED
    assert store.list_people() == []


def test_consent_given_on_behalf_of_the_subject_is_refused():
    mgr, store, _ = make_mgr()
    mgr.begin("Sam", 1, 1)
    r = mgr.on_utterance("yes they do")
    assert r.state is EnrollState.AWAITING_CONSENT
    assert "yourself" in r.say or "person" in r.say
    assert mgr.on_face(sample(vec(1))) is None  # still not collecting
    assert mgr.progress["samples"] == 0 and store.list_people() == []


def test_unclear_answers_cancel_after_one_reprompt():
    mgr, store, _ = make_mgr()
    mgr.begin("Sam", 1, 1)
    assert mgr.on_utterance("hmm").state is EnrollState.AWAITING_CONSENT
    assert mgr.on_utterance("the weather").state is EnrollState.CANCELLED


def test_two_people_in_view_blocks_start_and_collection():
    mgr, store, clock = make_mgr()
    assert mgr.begin("Sam", 1, 2).state is EnrollState.REFUSED
    assert mgr.begin("Sam", None, 0).state is EnrollState.REFUSED
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    clock.advance(1)
    r = mgr.on_face(sample(vec(1), faces=2))
    assert r.state is EnrollState.CANCELLED and store.list_people() == []


def test_a_different_person_stepping_in_aborts_enrollment():
    mgr, store, clock = make_mgr()
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    clock.advance(1)
    mgr.on_face(sample(vec(1, 0.1)))
    clock.advance(1)
    r = mgr.on_face(sample(vec(2), tid=2))
    assert r.state is EnrollState.CANCELLED and mgr.progress["samples"] == 0


def test_same_track_but_different_face_aborts():
    mgr, store, clock = make_mgr()
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    clock.advance(1)
    mgr.on_face(sample(vec(1)))
    clock.advance(1)
    r = mgr.on_face(sample(vec(99)))
    assert r.state is EnrollState.CANCELLED and store.list_people() == []


def test_unusable_and_near_duplicate_samples_are_not_counted():
    mgr, _, clock = make_mgr()
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    clock.advance(1)
    mgr.on_face(sample(vec(1), usable=False))
    assert mgr.progress["samples"] == 0
    mgr.on_face(sample(vec(1, 0.0)))
    clock.advance(1)
    mgr.on_face(sample(vec(1, 0.0)))
    assert mgr.progress["samples"] == 1      # same look again: skipped
    clock.advance(1)
    mgr.on_face(sample(vec(1, 0.6, 1)))
    assert mgr.progress["samples"] == 2
    clock.advance(0.1)
    mgr.on_face(sample(vec(1, 0.6, 2)))
    assert mgr.progress["samples"] == 2      # too soon after the last one


def test_needs_variety_of_poses_unless_target_reached():
    mgr, _, clock = make_mgr(min_samples=3, target_samples=6,
                             min_pose_bins=2)
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    r = feed(mgr, clock, 3, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert r is None or r.state is not EnrollState.CONFIRMING
    r = feed(mgr, clock, 3, [0.5])
    assert mgr.state in (EnrollState.CONFIRMING, EnrollState.COLLECTING)


def test_decline_at_final_confirmation_discards_samples():
    mgr, store, clock = make_mgr()
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    feed(mgr, clock, 7, [-0.4, -0.3, 0.0, 0.3, 0.4, 0.5])
    assert mgr.state is EnrollState.CONFIRMING
    r = mgr.on_utterance("no")
    assert r.state is EnrollState.CANCELLED and store.list_people() == []
    assert mgr.progress["samples"] == 0


def test_timeouts_end_enrollment_without_saving():
    mgr, store, clock = make_mgr(consent_timeout_s=10)
    mgr.begin("Sam", 1, 1)
    clock.advance(11)
    assert mgr.tick().state is EnrollState.CANCELLED
    assert mgr.state is EnrollState.CANCELLED and store.list_people() == []
    assert mgr.tick() is None


def test_duplicate_registration_under_a_new_name_is_blocked():
    store = make_store()
    mgr, _, clock = make_mgr(store)
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    feed(mgr, clock, 7, [-0.4, -0.3, 0.0, 0.3, 0.4, 0.5])
    assert mgr.on_utterance("yes").done
    mgr.begin("Sammy", 1, 1)
    mgr.on_utterance("yes")
    feed(mgr, clock, 7, [-0.4, -0.3, 0.0, 0.3, 0.4, 0.5])
    r = mgr.on_utterance("yes")
    assert r.state is EnrollState.REFUSED and "already know you as Sam" \
        in r.say
    assert [p["display_name"] for p in store.list_people()] == ["Sam"]


def test_same_name_for_a_different_face_does_not_overwrite():
    store = make_store()
    mgr, _, clock = make_mgr(store)
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    feed(mgr, clock, 7, [-0.4, -0.3, 0.0, 0.3, 0.4, 0.5])
    first = mgr.on_utterance("yes").person_id
    before = store.template_count(first)
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    feed(mgr, clock, 55, [-0.4, -0.3, 0.0, 0.3, 0.4, 0.5])
    r = mgr.on_utterance("yes")
    assert r.state is EnrollState.REFUSED
    assert store.template_count(first) == before


def test_same_person_can_add_more_looks_under_their_own_name():
    store = make_store()
    mgr, _, clock = make_mgr(store)
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    feed(mgr, clock, 7, [-0.4, -0.3, 0.0, 0.3, 0.4, 0.5])
    pid = mgr.on_utterance("yes").person_id
    before = store.template_count(pid)
    mgr.begin("Sam", 1, 1)
    mgr.on_utterance("yes")
    feed(mgr, clock, 7, [-0.5, -0.2, 0.0, 0.2, 0.5, 0.6])
    r = mgr.on_utterance("yes")
    assert r.done and r.person_id == pid
    assert store.template_count(pid) > before
    assert len(store.list_people()) == 1


def test_cancel_and_busy_handling():
    mgr, store, _ = make_mgr()
    assert mgr.cancel().say == "Nothing to cancel."
    mgr.begin("Sam", 1, 1)
    assert "already" in mgr.begin("Bo", 1, 1).say
    mgr.on_utterance("yes")
    assert mgr.on_utterance("no") .state is EnrollState.CANCELLED
    assert store.list_people() == []


def test_a_stranger_is_never_enrolled_by_looking_at_them():
    mgr, store, clock = make_mgr()
    for _ in range(50):
        clock.advance(1)
        assert mgr.on_face(sample(vec(5))) is None
    assert store.list_people() == [] and store.gallery() == {}


# ---------- greeting ----------

def test_greets_once_after_absence_and_not_every_frame():
    clock = Clock(1000.0)
    pol = GreetingPolicy(GreetingConfig(min_absence_s=60, cooldown_s=3600,
                                        confirm_s=1.0), clock)
    s = {"greet": True, "recognition": True}
    assert not pol.observe("p1", s)           # just appeared
    clock.advance(1.5)
    assert pol.observe("p1", s)               # settled in view
    pol.mark_greeted("p1")
    for _ in range(100):
        clock.advance(0.5)
        assert not pol.observe("p1", s)       # same visit: never again
    clock.advance(120)                        # left and came back
    pol.observe("p1", s)
    clock.advance(1.5)
    assert not pol.observe("p1", s)           # inside the cooldown
    clock.advance(4000)
    pol.observe("p1", s)
    clock.advance(1.5)
    assert pol.observe("p1", s)


def test_greeting_respects_person_settings():
    clock = Clock()
    pol = GreetingPolicy(clock=clock)
    clock.advance(1)
    pol.observe("p", {"greet": False})
    clock.advance(5)
    assert not pol.observe("p", {"greet": False})
    assert not pol.observe("p", {"greet": True, "recognition": False})
    pol.forget("p")
    assert "p" not in pol._last_seen
