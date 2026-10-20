from flask import Flask, jsonify, Response, request
import cv2
import threading
import time
import atexit
import sys
from pathlib import Path
from collections import deque

import numpy as np
from flask_cors import CORS

# --------------------------------------------------
# BUDDY — FLUTTER CAMERA + ADVANCED VISION API
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT / "buddy_ws" / "src" / "buddy_core")
)

from buddy_core.vision.vision_factory import build_vision
from buddy_core.vision.face_store import Consent, StoreError, clean_name

from buddy_emotion_report import register_emotion_report, record_emotion_people

app = Flask(__name__)
# Development only: allow the local Flutter Web app to call this API.
# Do not expose this unauthenticated API to untrusted networks.
CORS(app, resources={r"/api/*": {"origins": "*"}})

VISION_DATA = ROOT / "buddy_api_vision_data"

camera = None
vision = None

camera_lock = threading.Lock()
state_lock = threading.Lock()
stop_event = threading.Event()

worker_thread = None
latest_jpeg = None

latest_people = []
latest_objects = []
latest_events = []

camera_connected = False
vision_ready = False
last_frame_time = 0.0
last_vision_error = None

# The worker alone opens and reads the camera. These are short-lived raw
# frames kept in RAM only; the face database stores embeddings, not photos.
raw_frames = deque(maxlen=60)  # tuples of (monotonic timestamp, BGR frame)
vision_lock = threading.RLock()


def initialize_vision():
    global vision, vision_ready, last_vision_error

    print("\nLoading Sobiya's advanced vision system...")

    try:
        vision = build_vision(data_dir=str(VISION_DATA))

        vision_ready = vision.manager.c.faces is not None

        if not vision_ready:
            raise RuntimeError(
                "Face detector model is unavailable"
            )

        names = [
            p["display_name"]
            for p in vision.store.list_people()
        ]

        print("Vision loaded successfully.")
        print("Registered people:", names)
        print("Missing models:", vision.unavailable)

    except Exception as exc:
        vision = None
        vision_ready = False
        last_vision_error = str(exc)
        print("Vision initialization error:", exc)
        print("Basic camera streaming remains available.")


def open_camera():
    global camera, camera_connected

    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(
                0, cv2.CAP_DSHOW
            )

            if camera.isOpened():
                camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH, 640
                )
                camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT, 480
                )
                camera.set(
                    cv2.CAP_PROP_FPS, 15
                )

        camera_connected = (
            camera is not None and camera.isOpened()
        )

        return camera_connected


def draw_label(frame, text, x, y, color):
    height, width = frame.shape[:2]

    x = max(0, min(int(x), width - 1))
    y = max(24, min(int(y), height - 1))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        text, font, scale, thickness
    )

    right = min(width - 1, x + tw + 12)
    top = max(0, y - th - 12)

    cv2.rectangle(
        frame,
        (x, top),
        (right, y + 5),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y - 3),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_people(frame, people):
    height, width = frame.shape[:2]

    for person in people:
        box = person.box

        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        known = (
            person.person_id is not None
            and bool(person.name)
        )

        color = (
            (50, 185, 70)
            if known
            else (0, 160, 255)
        )

        name = person.name if known else "Unknown"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        draw_label(
            frame,
            name,
            x1,
            max(25, y1 - 7),
            color
        )


def serialize_person(person):
    return {
        "person_id": person.person_id,
        "name": person.name or "Unknown",
        "recognized": person.person_id is not None,
        "identity_state": person.identity_state,
        "box": list(person.box) if person.box else None,
        "expression": person.expression_group,
        "expression_confidence":
            person.expression_confidence,
    }


def camera_worker():
    global latest_jpeg
    global latest_people, latest_objects, latest_events
    global camera_connected, last_frame_time
    global last_vision_error

    print("Starting shared camera worker...")

    last_analysis = 0.0
    cached_people = []
    cached_objects = []
    cached_events = []

    while not stop_event.is_set():

        if not open_camera():
            with state_lock:
                camera_connected = False

            time.sleep(1)
            continue

        with camera_lock:
            success, frame = camera.read()

        if not success:
            with state_lock:
                camera_connected = False

            time.sleep(0.3)
            continue

        now = time.monotonic()
        # Capture about one raw frame every 0.3s (no photos written to disk).
        with state_lock:
            if not raw_frames or now - raw_frames[-1][0] >= 0.30:
                raw_frames.append((now, frame.copy()))

        # AI processing is deliberately slower than streaming.
        # This keeps the camera responsive on a laptop.
        if vision_ready and (
            now - last_analysis >= 0.35
        ):
            last_analysis = now

            try:
                # Do not run the detector/embedding pipeline concurrently
                # with a face enrollment request.
                with vision_lock:
                    result = vision.manager.process(
                        frame.copy(),
                        now=now
                    )

                # Diagnostics only: does not change any model or camera processing.
                with state_lock:
                    vision_stage_debug.clear()
                    vision_stage_debug.update({
                        "stages_run": list(result.stages_run),
                        "stages_skipped": dict(result.stages_skipped),
                        "timings_ms": dict(result.timings_ms),
                        "expression_service_frames": getattr(
                            vision.manager.c.expr_service, "frames", None),
                        "expression_track_details": [
                            {
                                "track_id": track_id,
                                "frames_in_window": len(getattr(track, "_win", ())),
                                "confirmed": getattr(track, "_confirmed", None),
                                "last_usable": getattr(track, "_last_usable", None),
                            }
                            for track_id, track in getattr(
                                vision.manager.c.expr_service, "_tracks", {}).items()
                        ],
                    })
                cached_people = list(result.people)
                # Log published facial-expression estimates for opted-in profiles.
                record_emotion_people(
                    [serialize_person(p) for p in cached_people], vision.store
                )
                cached_objects = list(result.objects)
                # Feed the existing teaching session with this shared-camera frame.
                _buddy_feed_object_teaching(frame.copy(), cached_objects, cached_people, now)
                cached_events = [
                    {
                        "kind": event.kind,
                        "data": event.data,
                    }
                    for event in result.events
                ]

                with state_lock:
                    latest_people = [
                        serialize_person(p)
                        for p in cached_people
                    ]

                    latest_objects = cached_objects
                    latest_events = cached_events
                    last_vision_error = None

            except Exception as exc:
                with state_lock:
                    last_vision_error = str(exc)

                print("Vision processing error:", exc)

        # Draw on the streaming copy, not the original frame.
        output = frame.copy()

        draw_people(output, cached_people)

        cv2.putText(
            output,
            "BUDDY AI VISION",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            output,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if ok:
            with state_lock:
                latest_jpeg = encoded.tobytes()
                last_frame_time = time.time()
                camera_connected = True

        time.sleep(1 / 15)

    print("Camera worker stopped.")


def generate_frames():
    previous = None

    while not stop_event.is_set():
        with state_lock:
            frame_bytes = latest_jpeg

        if frame_bytes is None:
            time.sleep(0.1)
            continue

        if frame_bytes == previous:
            time.sleep(0.04)
            continue

        previous = frame_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-store\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.get("/")
def home():
    return jsonify({
        "name": "BUDDY Robot API",
        "status": "online",
        "mode": "development",
        "vision_enabled": vision_ready
    })


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "buddy-api"
    })


@app.get("/api/robot_status")
def robot_status():
    with state_lock:
        cam = camera_connected

    return jsonify({
        "status": "online",
        "connected": True,
        "robot_name": "BUDDY",
        "robot_online": True,
        "battery": None,
        "mode": "development",
        "hardware_connected": False,
        "camera_connected": cam,
        "navigation_connected": False,
        "vision_connected": vision_ready
    })


vision_stage_debug = {}  # Last analysis only; no images stored


register_emotion_report(app, lambda: vision)


@app.get("/api/vision/status")
def vision_status():
    with state_lock:
        people = list(latest_people)
        objects = list(latest_objects)
        events = list(latest_events)
        error = last_vision_error
        stamp = last_frame_time
        cam = camera_connected

    return jsonify({
        "vision_ready": vision_ready,
        "camera_connected": cam,
        "last_frame_time": stamp,
        "people": people,
        "objects": objects,
        "events": events,
        "vision_diagnostics": dict(vision_stage_debug),
        "last_error": error
    })


@app.get("/api/household_members")
def household_members():
    if vision is None:
        return jsonify({"error": "Vision is unavailable"}), 503

    with vision_lock:
        members = vision.store.list_people()

    # Matches Sobiya's existing Flutter Household screen data shape.
    return jsonify([
        {
            "id": p["person_id"],
            "name": p["display_name"],
            "role": "Registered Companion",
            "status": "RECOGNIZED",
            "templates": p["templates"],
            "image": None,
        }
        for p in members
    ])


@app.post("/api/enroll_unknown_person")
def enroll_unknown_person():
    """Register one consenting subject from recent frames of the shared camera.

    JSON: {"name": "Person", "subject_consent": true}
    The app must first confirm consent from the *person being enrolled*.
    An asserted checkbox is not independent verification of their consent.
    """
    if vision is None or not vision_ready or vision.embedder is None:
        return jsonify(success=False, error="Face recognition unavailable"), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(success=False, error="JSON body required"), 400

    name = clean_name(data.get("name", ""))
    if not name or len(name) < 2:
        return jsonify(success=False, error="Enter a valid name"), 400

    if data.get("subject_consent") is not True:
        return jsonify(success=False, error=(
            "Ask the person for permission and confirm their consent "
            "before saving their face templates"
        )), 400

    with state_lock:
        now = time.monotonic()
        # Sample a few seconds of recent webcam frames. Never open camera again.
        frames = [frame.copy() for stamp, frame in raw_frames
                  if now - stamp <= 12.0]
        connected = camera_connected

    if not connected or len(frames) < 6:
        return jsonify(success=False, error=(
            "Camera needs a few more seconds of clear video. Please retry."
        )), 400

    # Avoid enrolling from a single near-identical moment.
    frames = frames[::max(1, len(frames) // 10)][:10]
    vectors = []
    try:
        with vision_lock:
            store = vision.store
            if store.find_by_name(name):
                return jsonify(success=False, error=(
                    "This name is already registered."
                )), 409

            detector = vision.manager.c.faces.detector
            gallery = store.gallery()
            saw_multiple = False
            for frame in frames:
                detections = detector.detect(frame)
                if len(detections) > 1:
                    saw_multiple = True
                    break
                if len(detections) != 1:
                    continue

                vec = vision.embedder.embed(frame, detections[0])
                if vec is None:
                    continue

                # Do not create duplicate identities for a known face.
                if any(np.max(templates @ vec) >= 0.48
                       for templates in gallery.values()):
                    return jsonify(success=False, error=(
                        "This person appears to be registered already."
                    )), 409
                vectors.append(vec)

            if saw_multiple:
                return jsonify(success=False, error=(
                    "More than one face detected. Only the consenting person "
                    "should be in view."
                )), 400

            if len(vectors) < 5:
                return jsonify(success=False, error=(
                    "Need at least five clear face samples. Face the camera "
                    "and move your head slightly, then retry."
                )), 400

            consent = Consent(
                given_at=time.time(),
                method="flutter_subject_confirmation",
                subject_confirmed=True,
                prompt="I agree to BUDDY storing my facial templates"
            )
            person_id = store.add_person(name, consent)
            try:
                for vec in vectors:
                    store.add_embedding(person_id, vec)
            except Exception:
                store.delete_person(person_id)
                raise

        return jsonify(
            success=True,
            name=name,
            person_id=person_id,
            templates=len(vectors),
            message=f"{name} registered successfully"
        ), 201
    except StoreError as exc:
        return jsonify(success=False, error=str(exc)), 400
    except Exception:
        app.logger.exception("Face enrollment failed")
        return jsonify(success=False, error="Face enrollment failed"), 500


@app.get("/video_call")
def video_call():
    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        ),
        headers={
            "Cache-Control": "no-store"
        }
    )


# BUDDY local notifications: uses existing shared vision state only.
from buddy_alert_routes import install_alert_routes

def _read_alert_vision_state():
    with state_lock:
        return {
            'people': list(latest_people),
            'camera_connected': camera_connected,
            'vision_ready': vision_ready,
            'jpeg': latest_jpeg,
        }

install_alert_routes(app, _read_alert_vision_state)



# BUDDY object teaching: shares the existing camera and the same object store.
from buddy_object_routes import install_object_routes

def _buddy_latest_object_frame():
    with state_lock:
        return raw_frames[-1][1].copy() if raw_frames else None

_buddy_feed_object_teaching = install_object_routes(
    app, lambda: vision, vision_lock, _buddy_latest_object_frame
)

@atexit.register
def cleanup():
    global camera

    stop_event.set()

    if worker_thread is not None:
        worker_thread.join(timeout=2)

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


if __name__ == "__main__":
    print("\nBUDDY FLUTTER + ADVANCED VISION API")
    print("Using vision data:", VISION_DATA)
    print("Motor commands remain disabled.\n")

    initialize_vision()

    worker_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    worker_thread.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
