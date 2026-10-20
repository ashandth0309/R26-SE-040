"""BUDDY shared-camera object teaching API, reusing the existing vision stack.

No new camera is opened; no existing database is deleted or overwritten.
Only user-confirmed feature vectors are persisted by TeachSession.
"""
import threading
import time
from dataclasses import dataclass

from flask import jsonify, request

from buddy_core.vision.object_teach import (
    TeachSession, Phase, ObjectRecognizer, identify,
)


@dataclass(frozen=True)
class _Detection:
    label: str
    score: float
    box: tuple
    category: str = ""

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _detections(items):
    result = []
    for item in items:
        try:
            # Reject face/person targets rather than teaching a person as an object.
            if isinstance(item, dict):
                label = str(item.get('label') or '')
                box = item.get('box')
                score = float(item.get('score') or 0)
                category = str(item.get('category') or '')
            else:
                label = str(item.label or '')
                box = item.box
                score = float(item.score)
                category = str(getattr(item, 'category', '') or '')
            if label.lower() == 'person' or box is None or len(box) != 4:
                continue
            result.append(_Detection(label, score, tuple(float(v) for v in box), category))
        except (TypeError, ValueError, AttributeError):
            continue
    return result


class ObjectLearningService:
    def __init__(self, get_vision, vision_lock):
        self.get_vision = get_vision
        self.vision_lock = vision_lock
        self.lock = threading.RLock()
        self.teacher = None
        self.vision_id = None
        self.message = 'Ready to learn an object.'
        self.updated = 0.0

    def _ensure(self):
        vision = self.get_vision()
        if vision is None or vision.object_embedder is None or vision.manager.c.objects is None:
            raise RuntimeError('Object recognition model is unavailable')
        if self.vision_id != id(vision):
            self.teacher = TeachSession(vision.objects, vision.object_embedder)
            self.vision_id = id(vision)
        return vision

    def _state(self):
        t = self.teacher
        return {
            'ok': True, 'phase': t.phase.value if t else 'idle',
            'active': bool(t and t.active),
            'samples': len(t.vecs) if t else 0,
            'samples_needed': t.cfg.samples_needed if t else 6,
            'name': t.name if t else '', 'reply': self.message,
            'updated': self.updated,
        }

    def status(self):
        with self.vision_lock, self.lock:
            self._ensure()
            return self._state()

    def command(self, action, name=''):
        with self.vision_lock, self.lock:
            self._ensure()
            t = self.teacher
            now = time.monotonic()
            if action == 'start':
                if t.active:
                    return {'ok': False, 'error': 'Finish or cancel the current lesson first.', **self._state()}
                r = t.start(name, now)
            elif action == 'confirm':
                if t.phase not in (Phase.CONFIRM_DUPLICATE, Phase.CONFIRM_SAVE):
                    return {**self._state(), 'ok': False, 'error': 'Samples are not ready for confirmation.'}
                r = t.answer('yes')
            elif action == 'reject':
                if t.phase not in (Phase.CONFIRM_DUPLICATE, Phase.CONFIRM_SAVE):
                    return {**self._state(), 'ok': False, 'error': 'There is no question to reject.'}
                r = t.answer('no')
            elif action == 'cancel':
                r = t.cancel()
            else:
                return {**self._state(), 'ok': False, 'error': 'Unknown teaching action.'}
            self.message = r.text
            self.updated = time.time()
            return {**self._state(), 'saved': r.saved, 'object_id': r.object_id}

    def feed(self, frame, objects, people, now):
        # Called only by existing Flask camera worker; NEVER open another camera.
        with self.vision_lock, self.lock:
            if self.teacher is None or self.teacher.phase is not Phase.CAPTURING:
                return
            boxes = [p.box for p in people if getattr(p, 'box', None) is not None]
            try:
                result = self.teacher.feed(frame, _detections(objects), now, faces=boxes)
                if result is not None:
                    self.message = result.text
                    self.updated = time.time()
            except Exception as exc:
                self.teacher.cancel('Object teaching stopped after a processing error.')
                self.message = 'Object teaching error: ' + str(exc)
                self.updated = time.time()

    def list_items(self):
        with self.vision_lock, self.lock:
            vision = self._ensure()
            return [{**obj, 'samples': vision.objects.sample_count(obj['object_id'])}
                    for obj in vision.objects.list()]

    def identify_latest(self, frame):
        with self.vision_lock, self.lock:
            vision = self._ensure()
            if self.teacher and self.teacher.active:
                raise ValueError('Finish or cancel teaching before identifying.')
            detector = vision.manager.c.objects
            faces = vision.manager.c.faces
            objects = _detections(detector.detect(frame))
            boxes = [p.box for p in faces.detector.detect(frame)] if faces else []
            answer = identify(frame, objects, vision.object_embedder,
                              ObjectRecognizer(vision.objects), faces=boxes)
            return {'success': True, 'kind': answer.kind, 'text': answer.text,
                    'name': answer.name, 'label': answer.label,
                    'offer_teach': answer.offer_teach,
                    'confidence': float(answer.confidence)}


def install_object_routes(app, get_vision, vision_lock, get_frame):
    service = ObjectLearningService(get_vision, vision_lock)

    @app.get('/api/objects')
    def buddy_objects_list():
        try:
            return jsonify(service.list_items())
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503

    @app.get('/api/objects/teach/status')
    def buddy_objects_teach_status():
        try:
            return jsonify(service.status())
        except RuntimeError as exc:
            return jsonify(ok=False, error=str(exc)), 503

    @app.post('/api/objects/teach')
    def buddy_objects_teach():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(ok=False, error='JSON object required'), 400
        action = str(data.get('action') or '').strip().lower()
        name = str(data.get('name') or '').strip()
        try:
            answer = service.command(action, name)
            return jsonify(answer), 200 if answer['ok'] else 400
        except RuntimeError as exc:
            return jsonify(ok=False, error=str(exc)), 503

    @app.post('/api/objects/identify')
    def buddy_objects_identify():
        frame = get_frame()
        if frame is None:
            return jsonify(success=False, error='No live camera frame available'), 503
        try:
            return jsonify(service.identify_latest(frame))
        except ValueError as exc:
            return jsonify(success=False, error=str(exc)), 409
        except RuntimeError as exc:
            return jsonify(success=False, error=str(exc)), 503

    return service.feed
