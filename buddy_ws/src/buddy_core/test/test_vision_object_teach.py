"""Phase 11: teaching, recognising, correcting and forgetting objects."""

import os
import stat

import cv2
import numpy as np
import pytest

from buddy_core.vision import models
from buddy_core.vision.object_embedding import FakeObjectEmbedder
from buddy_core.vision.object_teach import (
    MatchConfig, ObjectRecognizer, ObjectStore, ObjectStoreError, Phase,
    TeachConfig, TeachSession, correct, crop_quality, forget,
    forget_everything, identify, norm_name, parse_yes_no, select_target)
from buddy_core.vision.objects import ObjectDetection

RNG = np.random.default_rng(3)


def textured(color, size=240, seed=0):
    """A sharp, distinct blocky image so quality gates pass."""

    r = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), np.uint8)
    base[:] = color
    blocks = r.integers(-50, 50, (12, 12, 1)).astype(np.int16)
    noise = cv2.resize(blocks.astype(np.float32), (size, size),
                       interpolation=cv2.INTER_NEAREST)[..., None]
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def scene(color, seed=0, w=320, h=240):
    s = np.full((h, w, 3), 90, np.uint8)
    s[20:220, 60:260] = cv2.resize(textured(color, 200, seed), (200, 200))

    return s


def det(label="cup", box=(60, 20, 260, 220)):
    return [ObjectDetection(label, 0.9, box, "drink")]


def teach(store, name, color, n_frames=40, label="cup"):
    emb = FakeObjectEmbedder()
    t = TeachSession(store, emb, TeachConfig(samples_needed=4,
                                             min_gap_s=0.0))
    t.start(name, 0.0)
    reply = None

    for i in range(n_frames):
        # vary each frame: a different pattern seed and slight rotation
        img = scene(color, seed=i)
        r = t.feed(img, det(label), float(i))
        reply = r or reply

        if t.phase is Phase.CONFIRM_SAVE:
            break

    return t, reply


def test_norm_name_strips_articles_and_punctuation():
    assert norm_name("My Blue Mug!") == "blue mug"
    assert norm_name("  the   red  cup ") == "red cup"
    assert norm_name("") == ""


def test_yes_no_parser():
    assert parse_yes_no("yes please") is True
    assert parse_yes_no("no") is False
    assert parse_yes_no("no, don't") is False
    assert parse_yes_no("banana") is None


def test_full_teach_flow_stores_only_after_yes(tmp_path):
    store = ObjectStore(str(tmp_path / "o.db"))
    t, reply = teach(store, "blue mug", (200, 60, 30))
    assert t.phase is Phase.CONFIRM_SAVE
    assert store.list() == []               # nothing stored yet
    r = t.answer("yes")
    assert r.saved and "blue mug" in r.text
    assert [o["name"] for o in store.list()] == ["blue mug"]
    assert store.sample_count(r.object_id) == 4
    assert stat.S_IMODE(os.stat(tmp_path / "o.db").st_mode) == 0o600


def test_saying_no_discards_everything():
    store = ObjectStore()
    t, _ = teach(store, "blue mug", (200, 60, 30))
    r = t.answer("no")
    assert not r.saved and t.phase is Phase.IDLE and store.list() == []
    assert "Nothing was saved" in r.text


def test_unclear_answer_keeps_waiting():
    store = ObjectStore()
    t, _ = teach(store, "blue mug", (200, 60, 30))
    r = t.answer("hmm")
    assert t.phase is Phase.CONFIRM_SAVE and not r.saved


def test_cancel_and_timeout_save_nothing():
    store = ObjectStore()
    t = TeachSession(store, FakeObjectEmbedder(),
                     TeachConfig(timeout_s=5.0, min_gap_s=0.0))
    t.start("x", 0.0)
    assert t.feed(scene((10, 200, 10)), det(), 6.0).text.startswith("That")
    assert t.phase is Phase.IDLE and store.list() == []
    t.start("y", 10.0)
    t.cancel()
    assert t.phase is Phase.IDLE


def test_duplicate_samples_are_not_collected_twice():
    store = ObjectStore()
    t = TeachSession(store, FakeObjectEmbedder(),
                     TeachConfig(samples_needed=4, min_gap_s=0.0))
    t.start("x", 0.0)
    img = scene((50, 50, 200), seed=1)

    for i in range(10):
        t.feed(img, det(), float(i))

    assert len(t.vecs) == 1


def test_bad_quality_gives_hints_then_gives_up():
    store = ObjectStore()
    t = TeachSession(store, FakeObjectEmbedder(),
                     TeachConfig(min_gap_s=0.0, max_bad_frames=3))
    t.start("x", 0.0)
    dark = np.full((240, 320, 3), 5, np.uint8)
    first = t.feed(dark, det(), 1.0)
    assert "dark" in first.text
    assert t.feed(dark, det(), 2.0) is None          # no nagging
    t.feed(dark, det(), 3.0)
    last = t.feed(dark, det(), 4.0)
    assert "Nothing was saved" in last.text and t.phase is Phase.IDLE


def test_crop_quality_reasons():
    sharp = scene((120, 120, 120))
    assert crop_quality(sharp, (60, 20, 260, 220)) == (True, "ok")
    assert crop_quality(sharp, (0, 0, 20, 20))[1] == "too_small"
    assert crop_quality(np.full((240, 320, 3), 255, np.uint8),
                        (0, 0, 200, 200))[1] == "too_bright"
    blur = cv2.GaussianBlur(sharp, (51, 51), 20)
    assert crop_quality(blur, (60, 20, 260, 220))[1] == "blurry"


def test_face_in_crop_refuses_to_learn():
    store = ObjectStore()
    t = TeachSession(store, FakeObjectEmbedder(), TeachConfig(
        min_gap_s=0.0))
    t.start("dad", 0.0)
    r = t.feed(scene((10, 10, 10)), det(), 1.0, faces=[(100, 50, 160, 120)])
    assert "person" in r.text and store.list() == [] and not t.active


def test_duplicate_object_is_detected_and_can_add_samples():
    store = ObjectStore()
    t, _ = teach(store, "blue mug", (200, 60, 30))
    t.answer("yes")
    t2, reply = teach(store, "my cup", (200, 60, 30))
    assert t2.phase is Phase.CONFIRM_DUPLICATE
    assert "blue mug" in reply.text
    r = t2.answer("yes")
    assert "blue mug" in r.text and t2.phase is Phase.CONFIRM_SAVE
    r = t2.answer("yes")
    assert r.saved and len(store.list()) == 1
    assert store.sample_count(r.object_id) == 8


def test_duplicate_but_user_says_different_creates_new_object():
    store = ObjectStore()
    t, _ = teach(store, "blue mug", (200, 60, 30))
    t.answer("yes")
    t2, _ = teach(store, "second mug", (200, 60, 30))
    assert t2.phase is Phase.CONFIRM_DUPLICATE
    t2.answer("no")
    r = t2.answer("yes")
    assert r.saved and len(store.list()) == 2


def test_store_rules(tmp_path):
    s = ObjectStore(max_objects=2, max_samples=3)
    v = [np.ones(16, np.float32) / 4] * 5
    a = s.add("a", v)
    assert s.sample_count(a) == 3            # capped
    s.add("b", v)

    with pytest.raises(ObjectStoreError):
        s.add("c", v)                         # object limit

    with pytest.raises(ObjectStoreError):
        s.add("A", v)                         # name used (normalised)

    with pytest.raises(ObjectStoreError):
        s.add("   ", v)

    s.rename(a, "alpha")
    assert s.find("my alpha") == a

    with pytest.raises(ObjectStoreError):
        s.rename(a, "b")

    assert s.delete(a) and s.find("alpha") is None
    assert s.gallery().keys() == {s.find("b")}


def test_delete_all_and_forget_helpers():
    s = ObjectStore()
    s.add("a", [np.ones(4, np.float32)])
    s.add("b", [np.ones(4, np.float32)])
    assert "forgot a" in forget(s, "my a")
    assert "don't have" in forget(s, "zzz")
    assert "all 1" in forget_everything(s)
    assert "don't have" in forget_everything(s)


def test_recognizer_match_uncertain_unknown_and_class_hint():
    s = ObjectStore()
    emb = FakeObjectEmbedder()
    mug = emb.embed(scene((200, 60, 30)))
    s.add("blue mug", [mug], "cup")
    r = ObjectRecognizer(s, MatchConfig(accept=0.8, margin=0.05))
    assert r.match(mug, "cup").state == "match"
    assert r.match(mug, "bottle").state == "unknown"      # wrong class
    other = emb.embed(scene((20, 200, 40)))
    assert r.match(other, "cup").state in ("unknown", "uncertain")
    assert r.match(other, "cup").name != "blue mug" or \
        r.match(other, "cup").state != "match"


def test_two_similar_objects_do_not_match_without_margin():
    s = ObjectStore()
    v = np.ones(16, np.float32)
    v /= np.linalg.norm(v)
    w = v.copy()
    w[0] += 0.02
    w /= np.linalg.norm(w)
    s.add("one", [v])
    s.add("two", [w])
    m = ObjectRecognizer(s).match(v)
    assert m.state != "match"


def test_correction_blocks_that_look_for_that_object():
    s = ObjectStore()
    emb = FakeObjectEmbedder()
    mug = emb.embed(scene((200, 60, 30)))
    look_alike = emb.embed(scene((198, 62, 32), seed=1))
    s.add("blue mug", [mug])
    r = ObjectRecognizer(s, MatchConfig(accept=0.5, margin=0.0,
                                        negative_sim=0.9))
    assert r.match(look_alike).state == "match"
    msg = correct(s, "blue mug", look_alike)
    assert "not blue mug" in msg
    assert r.match(look_alike).state != "match"
    assert "don't have" in correct(s, "ghost", look_alike)


def test_correction_with_right_name_adds_positive_sample():
    s = ObjectStore()
    emb = FakeObjectEmbedder()
    a = s.add("mug", [emb.embed(scene((200, 60, 30)))])
    b = s.add("bowl", [emb.embed(scene((20, 60, 230)))])
    v = emb.embed(scene((100, 100, 100), seed=5))
    correct(s, "mug", v, "bowl")
    assert s.sample_count(b) == 2 and s.sample_count(a) == 1


def test_select_target_prefers_large_central_and_falls_back():
    big = ObjectDetection("cup", 0.8, (100, 80, 220, 200), "drink")
    small = ObjectDetection("cup", 0.9, (0, 0, 20, 20), "drink")
    t = select_target([small, big], (240, 320, 3))
    assert t.box == (100, 80, 220, 200) and t.source == "detector"
    f = select_target([], (240, 320, 3))
    assert f.source == "center_crop" and f.label is None
    assert select_target([], (240, 320, 3), allow_center_crop=False) is None


def test_identify_phrases_are_hedged_and_offer_teaching():
    s = ObjectStore()
    emb = FakeObjectEmbedder()
    img = scene((200, 60, 30))
    s.add("blue mug", [emb.embed(img[20:220, 60:260])], "cup")
    rec = ObjectRecognizer(s)
    a = identify(img, det(), emb, rec)
    assert a.kind == "taught" and a.text.startswith("I think")
    b = identify(scene((20, 200, 40)), det("bottle"), emb, rec)
    assert b.kind == "generic" and "bottle" in b.text and b.offer_teach
    c = identify(scene((20, 200, 40)), [], emb, rec)
    assert c.kind == "unknown" and "teach" in c.text
    d = identify(np.full((240, 320, 3), 5, np.uint8), det(), emb, rec)
    assert d.kind == "unknown" and "clearly" in d.text
    e = identify(img, det(), emb, rec, faces=[(100, 50, 160, 120)])
    assert "not people" in e.text


needs = pytest.mark.skipif(
    not models.available("object_embedder_source"),
    reason="model not fetched")


@needs
def test_real_embedder_separates_same_and_different_objects():
    from skimage import data

    from buddy_core.vision.object_embedding import ObjectEmbedder
    e = ObjectEmbedder()
    a = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)
    b = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    a2 = a[h // 8:-h // 8, w // 8:-w // 8]
    va, va2, vb = e.embed(a), e.embed(a2), e.embed(b)
    assert va.shape == (1280,) and abs(np.linalg.norm(va) - 1) < 1e-4
    assert float(va @ va2) > 0.85 > 0.65 > float(va @ vb)
