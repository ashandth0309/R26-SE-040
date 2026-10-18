"""Phase 12: object and scene memory (opt-in, expiring, forgettable)."""

import json
import os
import stat

import numpy as np
import pytest

from buddy_core.vision.object_memory import ObjectMemory, ago, region_of

SHAPE = (480, 640, 3)
BOX = (300, 200, 340, 240)


def mem(**kw):
    m = ObjectMemory(**kw)
    m.enable()

    return m


def see(m, now, name="blue mug", place="kitchen table", subject="o1",
        kind="taught", label="cup"):
    return m.observe(subject, kind, label, name, place, BOX, SHAPE, now)


def test_off_by_default_records_and_answers_nothing():
    m = ObjectMemory()
    assert not m.enabled
    assert not see(m, 0.0)
    assert "not keeping track" in m.where_is("blue mug", 1.0).text
    assert m.export() == "[]"


def test_where_is_reports_place_and_age_without_promising():
    m = mem()
    see(m, 1000.0)
    a = m.where_is("my blue mug".replace("my ", ""), 1000.0 + 1200)
    assert a.found and "kitchen table" in a.text
    assert "20 minutes ago" in a.text
    assert "can't tell whether it is still there" in a.text
    assert not a.stale


def test_old_sighting_is_marked_possibly_moved():
    m = mem()
    see(m, 0.0)
    a = m.where_is("blue mug", 8 * 3600)
    assert a.stale and "may have been moved" in a.text
    assert "8 hours ago" in a.text


def test_never_seen_and_unknown_names():
    m = mem()
    assert m.where_is("keys", 0.0).found is False
    assert "haven't seen" in m.where_is("keys", 0.0).text


def test_update_rate_limit_and_move_detection():
    m = mem(min_update_s=5.0)
    assert see(m, 0.0)
    assert not see(m, 2.0)                       # too soon, same place
    assert see(m, 3.0, place="sofa")             # moved: always recorded
    assert m.moved_recently("blue mug", 10.0) == ("kitchen table", "sofa")
    assert m.moved_recently("blue mug", 5000.0) is None
    assert "sofa" in m.where_is("blue mug", 10.0).text


def test_people_and_unknown_kinds_are_never_recorded():
    m = mem()
    assert not m.observe("p", "generic", "person", None, "sofa", BOX,
                         SHAPE, 0.0)
    assert not m.observe("x", "face", "cup", None, "sofa", BOX, SHAPE, 0.0)
    assert json.loads(m.export()) == []


def test_generic_objects_answer_by_label():
    m = mem()
    m.observe("g1", "generic", "remote", None, None, BOX, SHAPE, 0.0)
    a = m.where_is("remote", 60.0)
    assert a.found and "centre" in a.text


def test_seen_at_lists_recent_only():
    m = mem()
    see(m, 0.0, subject="a", name="mug")
    see(m, 1700.0, subject="b", name="bowl")
    assert m.seen_at("kitchen table", 1800.0, 600.0) == ["bowl"]
    assert set(m.seen_at("kitchen table", 1800.0)) == {"mug", "bowl"}


def test_forget_one_all_purge_and_disable():
    m = mem(retention_s=100.0)
    see(m, 0.0, subject="a", name="mug")
    see(m, 90.0, subject="b", name="bowl")
    assert m.purge(150.0) == 1                   # mug expired
    assert m.where_is("mug", 150.0).found is False
    assert m.forget("bowl") == 1
    see(m, 200.0, subject="c", name="pan")
    m.disable(forget=True)
    assert not m.enabled and json.loads(m.export()) == []


def test_row_cap_drops_oldest():
    m = mem(max_rows=3)

    for i in range(5):
        see(m, float(i * 10), subject=f"s{i}", name=f"n{i}")

    names = {r["name"] for r in json.loads(m.export())}
    assert names == {"n2", "n3", "n4"}


def test_file_is_owner_only_and_stores_no_pixels(tmp_path):
    p = tmp_path / "m.db"
    m = mem(path=str(p))
    see(m, 0.0)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    cols = [r[1] for r in m._db.execute("PRAGMA table_info(sightings)")]
    assert not any(c in cols for c in ("image", "frame", "thumbnail"))


def test_scene_memory_teach_and_recognise():
    m = mem()
    rng = np.random.default_rng(0)

    def vec(centre):
        v = centre + rng.normal(0, 0.05, centre.shape)

        return (v / np.linalg.norm(v)).astype(np.float32)

    kitchen = rng.normal(size=64)
    lounge = rng.normal(size=64)
    m.teach_place("Kitchen", [vec(kitchen) for _ in range(4)])
    m.teach_place("living room", [vec(lounge) for _ in range(4)])
    assert m.places() == ["kitchen", "living room"]
    assert m.recognise_place(vec(kitchen))[0] == "kitchen"
    assert m.recognise_place(vec(lounge))[0] == "living room"
    other = rng.normal(size=64)
    assert m.recognise_place(vec(other))[0] is None
    assert m.forget_place("kitchen") and m.places() == ["living room"]


def test_teaching_a_place_needs_enough_views_and_a_name():
    m = mem()
    v = np.ones(8, np.float32) / 8 ** .5

    with pytest.raises(ValueError):
        m.teach_place("kitchen", [v, v])

    with pytest.raises(ValueError):
        m.teach_place("  ", [v, v, v])


def test_helpers():
    assert region_of((0, 0, 10, 10), SHAPE) == "upper left"
    assert region_of((310, 230, 330, 250), SHAPE) == "centre"
    assert region_of((600, 440, 630, 470), SHAPE) == "lower right"
    assert ago(30) == "a moment ago"
    assert ago(7200) == "about 2 hours ago"
    assert ago(86400 * 3) == "about 3 days ago"
