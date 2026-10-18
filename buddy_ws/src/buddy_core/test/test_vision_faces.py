"""Face detection quality, tracking, identity decisions, scheduling."""

import cv2
import numpy as np
import pytest

from buddy_core.vision import models
from buddy_core.vision.embedding import HashEmbedder, l2_normalize
from buddy_core.vision.face_pipeline import (
    FaceTrackingPipeline,
    PipelineConfig,
)
from buddy_core.vision.faces import (
    FaceDetection,
    QualityConfig,
    ScriptedDetector,
    assess_quality,
    yaw_and_roll,
)
from buddy_core.vision.identity import (
    IdentityState,
    MatchConfig,
    TrackIdentity,
    match,
)
from buddy_core.vision.tracker import (
    MultiTracker,
    TrackerConfig,
    TrackState,
    iou,
    nms,
)


def box(cx, cy, s=80):
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


# ---------- tracker ----------

def test_iou_and_nms_remove_duplicates():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    boxes = [box(100, 100), box(104, 102), box(400, 100)]
    assert nms(boxes, [0.9, 0.8, 0.7], 0.6) == [0, 2]


def test_track_confirms_after_n_init_and_keeps_its_id():
    tr = MultiTracker(TrackerConfig(n_init=2))
    tr.update([box(100, 100)], 0.0)
    assert tr.tracks[0].state is TrackState.TENTATIVE and not tr.visible()
    tr.update([box(104, 100)], 0.1)
    assert tr.visible() and tr.visible()[0].track_id == 1
    for i in range(20):
        tr.update([box(108 + 4 * i, 100)], 0.2 + 0.1 * i)
    assert [t.track_id for t in tr.visible()] == [1]
    assert tr.created_total == 1


def test_duplicate_detections_make_one_track():
    tr = MultiTracker()
    for t in (0.0, 0.1, 0.2):
        tr.update([box(100, 100), box(103, 101)], t)
    assert len(tr.visible()) == 1


def test_two_people_crossing_do_not_swap_ids():
    tr = MultiTracker(TrackerConfig(n_init=1))
    ids = {}
    for i in range(30):
        t = i * 0.1
        a, b = box(50 + 20 * i, 100), box(650 - 20 * i, 160)
        tr.update([a, b], t)
        if i == 0:
            ids = {round(x.box[0]): x.track_id for x in tr.tracks}
    left = min(tr.tracks, key=lambda x: x.box[1])
    right = max(tr.tracks, key=lambda x: x.box[1])
    assert left.track_id == ids[10] and right.track_id == ids[610]


def test_brief_occlusion_keeps_id_and_marks_lost():
    tr = MultiTracker(TrackerConfig(n_init=2, max_age_s=1.5))
    for i in range(4):
        tr.update([box(100 + 10 * i, 100)], i * 0.1)
    tid = tr.visible()[0].track_id
    tr.coast(0.8)
    assert tr.tracks[0].state is TrackState.LOST and not tr.visible()
    tr.update([box(150, 100)], 1.2)
    assert [t.track_id for t in tr.visible()] == [tid]


def test_person_who_leaves_for_long_gets_a_new_session_id():
    tr = MultiTracker(TrackerConfig(n_init=2, max_age_s=1.0))
    for i in range(3):
        tr.update([box(100, 100)], i * 0.1)
    first = tr.visible()[0].track_id
    tr.coast(5.0)
    assert tr.tracks == []
    for i in range(3):
        tr.update([box(100, 100)], 6.0 + i * 0.1)
    second = tr.visible()[0].track_id
    assert second != first and tr.visible()[0].session_id.startswith(
        "session_person_")


def test_one_frame_false_positive_never_becomes_a_track():
    tr = MultiTracker(TrackerConfig(n_init=2, tentative_age_s=0.5))
    tr.update([box(300, 300)], 0.0)
    tr.coast(1.0)
    assert tr.tracks == [] and tr.visible() == []


def test_predicted_box_moves_with_velocity_but_is_capped():
    tr = MultiTracker(TrackerConfig(n_init=1))
    for i in range(5):
        tr.update([box(100 + 50 * i, 100)], i * 0.1)
    t = tr.tracks[0]
    near = t.predicted(t.last_seen + 0.2)
    far = t.predicted(t.last_seen + 10.0)
    assert near[0] > t.box[0]
    assert far[0] - t.box[0] <= 600 * 0.5 + 1


# ---------- identity ----------

def unit(seed):
    return l2_normalize(np.random.default_rng(seed).standard_normal(128))


def noisy(v, amount, seed):
    rng = np.random.default_rng(seed)

    return l2_normalize(v + amount * rng.standard_normal(v.size))


def test_empty_gallery_is_unknown_not_a_guess():
    r = match(unit(1), {})
    assert r.state is IdentityState.UNKNOWN and r.person_id is None


def test_enrolled_person_is_recognised_and_stranger_is_unknown():
    gallery = {"alice": np.stack([noisy(unit(1), 0.1, s)
                                  for s in range(4)])}
    assert match(noisy(unit(1), 0.1, 99), gallery).person_id == "alice"
    stranger = match(unit(2), gallery)
    assert stranger.state is IdentityState.UNKNOWN
    assert stranger.person_id is None


def test_close_to_two_people_is_uncertain_and_not_named():
    a, b = unit(1), unit(2)
    mixed = l2_normalize(a + b)
    gallery = {"alice": a[None], "bob": b[None]}
    r = match(mixed, gallery, MatchConfig(accept=0.5, floor=0.3,
                                          margin=0.1))
    assert r.state is IdentityState.UNCERTAIN
    assert r.person_id is None and r.candidate in ("alice", "bob")


def test_score_between_floor_and_accept_is_uncertain():
    a = unit(1)
    probe = l2_normalize(a + 1.6 * unit(7))
    r = match(probe, {"alice": a[None]},
              MatchConfig(accept=0.9, floor=0.3, margin=0.0))
    assert r.state is IdentityState.UNCERTAIN and r.person_id is None


def test_one_noisy_frame_cannot_name_or_unname_someone():
    from buddy_core.vision.identity import MatchResult as MR
    good = MR(IdentityState.RECOGNIZED, "alice", "alice", 0.8, 0.3)
    bad = MR(IdentityState.UNKNOWN, None, None, 0.1, None)
    ti = TrackIdentity()
    ti.add(good, 0.0)
    ti.add(good, 0.5)
    assert ti.state is not IdentityState.RECOGNIZED
    ti.add(good, 1.0)
    assert ti.state is IdentityState.RECOGNIZED and ti.person_id == "alice"
    ti.add(bad, 6.0)
    assert ti.person_id == "alice"  # one bad frame does not un-name
    for i in range(4):
        ti.add(bad, 7.0 + i)
    assert ti.person_id is None and ti.state is not IdentityState.RECOGNIZED


def test_two_different_people_across_frames_are_never_merged():
    from buddy_core.vision.identity import MatchResult as MR
    ti = TrackIdentity()
    for i, who in enumerate(["alice", "bob", "alice"]):
        ti.add(MR(IdentityState.RECOGNIZED, who, who, 0.8, 0.3), i * 0.5)
    assert ti.person_id is None


def test_retry_schedule_slows_for_unknown_and_is_rare_for_named():
    from buddy_core.vision.identity import MatchResult as MR
    ti = TrackIdentity()
    assert ti.due(0.0)
    unk = MR(IdentityState.UNKNOWN, None, None, 0.1, None)
    for i in range(3):
        ti.add(unk, i * 0.5)
    assert not ti.due(1.5) and ti.due(4.5)


# ---------- quality ----------

def face_img(brightness=140, size=120, noise=40):
    """Blocky texture: keeps contrast and edges after any resize."""

    rng = np.random.default_rng(0)
    cells = rng.normal(0, noise, (size // 6, size // 6, 1))
    block = np.repeat(np.repeat(cells, 12, axis=0), 12, axis=1)
    img = np.clip(brightness + np.repeat(block, 3, axis=2), 0, 255)

    return img.astype(np.uint8)


def det(x, y, w, h, lm=None):
    return FaceDetection(x, y, w, h, 0.9, lm or ())


def test_quality_flags_dark_bright_blurry_small_and_cut_faces():
    ok = assess_quality(face_img(), det(40, 40, 120, 120))
    assert ok.usable, ok.reasons
    assert "too_dark" in assess_quality(
        face_img(10, noise=2), det(40, 40, 120, 120)).reasons
    assert "too_bright" in assess_quality(
        face_img(250, noise=2), det(40, 40, 120, 120)).reasons
    blurred = cv2.GaussianBlur(face_img(), (0, 0), 8)
    assert "blurry" in assess_quality(blurred, det(40, 40, 120, 120)).reasons
    assert "too_small" in assess_quality(face_img(), det(40, 40, 30, 30))\
        .reasons
    assert "partly_out_of_frame" in assess_quality(
        face_img(), det(180, 40, 120, 120)).reasons


def test_head_turn_is_flagged_from_landmarks():
    frontal = ((90, 100), (150, 100), (120, 130), (95, 160), (145, 160))
    turned = ((90, 100), (150, 100), (150, 130), (95, 160), (145, 160))
    yaw0, roll0 = yaw_and_roll(frontal)
    yaw1, _ = yaw_and_roll(turned)
    assert abs(yaw0) < 0.05 and abs(roll0) < 1
    assert abs(yaw1) > 0.4
    q = assess_quality(face_img(), det(40, 40, 120, 120, turned),
                       QualityConfig(max_yaw_hint=0.3))
    assert "head_turned" in q.reasons
    assert yaw_and_roll(()) == (None, None)


# ---------- pipeline ----------

def frame(w=640, h=480):
    rng = np.random.default_rng(1)
    cells = rng.normal(0, 40, (h // 12, w // 12, 1))
    block = np.repeat(np.repeat(cells, 12, axis=0), 12, axis=1)

    return np.clip(140 + np.repeat(block, 3, axis=2), 0, 255)\
        .astype(np.uint8)


def make_pipeline(script, gallery=None, labels=None, **cfg):
    det_ = ScriptedDetector(script)
    emb = HashEmbedder(labels or {})
    config = PipelineConfig(detect_min_interval_s=0.0, detect_width=640,
                            **cfg)
    pipe = FaceTrackingPipeline(det_, emb, lambda: gallery or {}, config)

    return pipe, det_, emb


def run(pipe, n, dt=0.1, start=0.0):
    out = None
    img = frame()
    for i in range(n):
        out = pipe.process(img, start + i * dt)

    return out


def one_face(x=200, y=150, s=100):
    return [FaceDetection(x, y, s, s, 0.9)]


def test_no_face_is_different_from_unknown_person():
    pipe, _, _ = make_pipeline([[]])
    out = run(pipe, 3)
    assert out.no_face and out.faces_in_view == 0 and out.tracks == ()
    pipe2, _, _ = make_pipeline([one_face()])
    out2 = run(pipe2, 6)
    assert not out2.no_face and out2.faces_in_view == 1
    assert out2.tracks[0].identity_state in (
        IdentityState.UNKNOWN, IdentityState.UNCERTAIN)
    assert out2.tracks[0].person_id is None


def test_enrolled_person_gets_named_and_stranger_does_not():
    embedder = HashEmbedder({200: "alice", 400: "stranger"})
    gallery = {"alice": embedder.embed(None, FaceDetection(
        200, 0, 1, 1, 1.0))[None]}
    pipe, _, _ = make_pipeline(
        [one_face(200) + one_face(400)], gallery,
        {200: "alice", 400: "stranger"})
    out = run(pipe, 12, dt=0.6)
    named = {t.person_id for t in out.tracks}
    assert named == {"alice", None}
    stranger = [t for t in out.tracks if t.person_id is None][0]
    assert stranger.identity_state is IdentityState.UNKNOWN


def test_recognition_does_not_run_every_frame():
    embedder = HashEmbedder({200: "alice"})
    gallery = {"alice": embedder.embed(None, FaceDetection(
        200, 0, 1, 1, 1.0))[None]}
    pipe, _, emb = make_pipeline([one_face(200)], gallery, {200: "alice"})
    run(pipe, 100, dt=0.1)
    assert pipe.frames == 100
    assert emb.calls < 100 * 0.2


def test_embeddings_are_capped_per_frame():
    script = [[FaceDetection(60 + 120 * i, 150, 100, 100, 0.9)
               for i in range(5)]]
    pipe, _, emb = make_pipeline(script, max_embeds_per_frame=1)
    run(pipe, 1)
    run(pipe, 2, start=0.1)
    assert emb.calls <= 3


def test_detector_rate_is_limited_and_tracker_coasts_between_passes():
    det_ = ScriptedDetector([one_face()])
    pipe = FaceTrackingPipeline(det_, None, None, PipelineConfig(
        detect_min_interval_s=0.375, detect_width=640))
    img = frame()
    passes = []
    for i in range(10):
        passes.append(pipe.process(img, i * 0.125).detected)
    assert sum(passes) == 4 and det_.calls == 4
    assert pipe.process(img, 1.25).faces_in_view == 1


def test_poor_quality_faces_are_not_embedded():
    dark = np.full((480, 640, 3), 5, np.uint8)
    pipe, _, emb = make_pipeline([one_face()])
    for i in range(10):
        pipe.process(dark, i * 0.1)
    assert emb.calls == 0


def test_recognition_can_be_switched_off():
    pipe, _, emb = make_pipeline([one_face()], recognition_enabled=False)
    out = run(pipe, 10)
    assert emb.calls == 0 and out.faces_in_view == 1


def test_large_frames_are_detected_at_reduced_width():
    seen = []

    class Spy:
        def detect(self, image):
            seen.append(image.shape[1])

            return []

    pipe = FaceTrackingPipeline(Spy(), None, None, PipelineConfig(
        detect_width=320, detect_min_interval_s=0.0))
    pipe.process(np.zeros((960, 1280, 3), np.uint8), 0.0)
    assert seen == [320]


def test_boxes_from_small_detection_map_back_to_full_frame():
    class Fixed:
        def detect(self, image):
            return [FaceDetection(80, 60, 40, 40, 0.9)]

    pipe = FaceTrackingPipeline(Fixed(), None, None, PipelineConfig(
        detect_width=320, detect_min_interval_s=0.0))
    img = np.full((960, 1280, 3), 120, np.uint8)
    out = None
    for i in range(3):
        out = pipe.process(img, i * 0.1)
    x1, y1, x2, y2 = out.tracks[0].box
    assert (x1, y1, x2, y2) == pytest.approx((320, 240, 480, 400))


def test_timings_are_measured_not_invented():
    pipe, _, _ = make_pipeline([one_face()])
    out = run(pipe, 2)
    assert set(out.timings_ms) <= {"detect", "recognise"}
    assert all(v >= 0 for v in out.timings_ms.values())


# ---------- real models (skipped when the files are not installed) ----------

need_models = pytest.mark.skipif(
    not (models.available("face_detector") and
         models.available("face_embedder")),
    reason="vision models not installed (scripts/fetch_vision_models.py)")


@pytest.fixture(scope="module")
def portrait():
    skimage = pytest.importorskip("skimage")
    from skimage import data

    assert skimage

    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


@need_models
def test_yunet_finds_the_face_in_a_public_domain_portrait(portrait):
    from buddy_core.vision.faces import YuNetDetector
    d = YuNetDetector(str(models.model_path("face_detector")))
    faces = d.detect(portrait)
    assert len(faces) == 1
    f = faces[0]
    assert 150 < f.x < 210 and 40 < f.y < 100 and len(f.landmarks) == 5
    assert assess_quality(portrait, f).usable
    assert d.detect(np.zeros((240, 320, 3), np.uint8)) == []


@need_models
def test_sface_same_person_scores_above_accept_under_changes(portrait):
    from buddy_core.vision.embedding import SFaceEmbedder
    from buddy_core.vision.faces import YuNetDetector
    d = YuNetDetector(str(models.model_path("face_detector")))
    e = SFaceEmbedder(str(models.model_path("face_embedder")))
    base = e.embed(portrait, d.detect(portrait)[0])
    assert base.shape == (128,)
    assert float(np.linalg.norm(base)) == pytest.approx(1.0, abs=1e-4)
    variants = {
        "dark": np.clip(portrait * 0.35, 0, 255).astype(np.uint8),
        "blur": cv2.GaussianBlur(portrait, (0, 0), 2.5),
        "half": cv2.resize(portrait, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA),
        "mirror": portrait[:, ::-1].copy(),
    }
    cfg = MatchConfig()
    for name, img in variants.items():
        found = d.detect(img)
        assert found, name
        sim = float(base @ e.embed(img, found[0]))
        assert sim > cfg.accept, (name, sim)


@need_models
def test_sface_two_different_public_domain_people_score_below_accept(
        portrait):
    from skimage import data
    from buddy_core.vision.embedding import SFaceEmbedder
    from buddy_core.vision.faces import YuNetDetector
    other = cv2.cvtColor(data.camera(), cv2.COLOR_GRAY2BGR)
    d = YuNetDetector(str(models.model_path("face_detector")))
    e = SFaceEmbedder(str(models.model_path("face_embedder")))
    a = e.embed(portrait, d.detect(portrait)[0])
    faces = d.detect(other)
    assert faces
    b = e.embed(other, faces[0])
    assert float(a @ b) < MatchConfig().accept
    assert match(b, {"astronaut": a[None]}).person_id is None


@need_models
def test_end_to_end_pipeline_on_real_models(portrait):
    from buddy_core.vision.embedding import SFaceEmbedder
    from buddy_core.vision.faces import YuNetDetector
    d = YuNetDetector(str(models.model_path("face_detector")))
    e = SFaceEmbedder(str(models.model_path("face_embedder")))
    ref = e.embed(portrait, d.detect(portrait)[0])
    pipe = FaceTrackingPipeline(d, e, lambda: {"astro": ref[None]},
                                PipelineConfig(detect_width=512,
                                               detect_min_interval_s=0.0))
    out = None
    for i in range(8):
        out = pipe.process(portrait, i * 0.6)
    assert out.faces_in_view == 1
    assert out.tracks[0].person_id == "astro"
    assert out.tracks[0].identity_state is IdentityState.RECOGNIZED


def test_face_stays_visible_between_slow_detection_passes():
    det_ = ScriptedDetector([one_face()])
    pipe = FaceTrackingPipeline(det_, None, None, PipelineConfig(
        detect_min_interval_s=0.5, detect_width=640))
    img = frame()
    counts = [pipe.process(img, i * 0.125).faces_in_view
              for i in range(16)]
    assert counts[:4] == [0, 0, 0, 0]       # needs two passes to confirm
    assert counts[4:] == [1] * 12           # then stays between passes

    gone = ScriptedDetector([one_face(), one_face(), []])
    pipe2 = FaceTrackingPipeline(gone, None, None, PipelineConfig(
        detect_min_interval_s=0.5, detect_width=640))
    outs = [pipe2.process(img, i * 0.5) for i in range(8)]
    assert outs[1].faces_in_view == 1
    assert outs[-1].tracks == () and outs[-1].no_face
