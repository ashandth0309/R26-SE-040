"""BUDDY local-development unknown-person notification endpoints.

Temporary RAM-only snapshots (120 sec), no disk storage, no additional camera.
Call install_alert_routes(app, read_vision_state) from buddy_api.py.
"""
import base64
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from threading import RLock
import time
from flask import jsonify


def install_alert_routes(app, read_vision_state):
    alerts = deque(maxlen=30)
    lock = RLock()
    sequence = 0
    last_seen = 0.0
    last_alert = 0.0
    acknowledged_while_present = False
    snapshot_ttl = 120.0

    def sample():
        nonlocal sequence, last_seen, last_alert, acknowledged_while_present
        state = read_vision_state()
        people = state.get('people') or []
        live = bool(state.get('camera_connected') and state.get('vision_ready'))
        unknown = live and any(
            isinstance(p, dict) and not p.get('recognized', False)
            and p.get('box') is not None for p in people
        )
        now = time.monotonic()
        # Temporary thumbnail: from existing encoded camera output, never opens camera.
        preview = None
        if unknown and isinstance(state.get('jpeg'), bytes):
            try:
                target = next(p for p in people if isinstance(p, dict)
                    and not p.get('recognized', False) and p.get('box'))
                raw = np.frombuffer(state['jpeg'], dtype=np.uint8)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is not None:
                    height, width = frame.shape[:2]
                    x1, y1, x2, y2 = [int(v) for v in target['box']]
                    margin = int(max(x2-x1, y2-y1) * 0.22)
                    x1, y1 = max(0, x1-margin), max(0, y1-margin)
                    x2, y2 = min(width, x2+margin), min(height, y2+margin)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (160, 160))
                        ok, compressed = cv2.imencode('.jpg', crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 65])
                        if ok:
                            preview = base64.b64encode(compressed.tobytes()).decode('ascii')
            except (StopIteration, ValueError, cv2.error):
                pass
        with lock:
            for alert in alerts:
                if alert.get('image') is not None and now - alert.get('_created', 0) > snapshot_ttl:
                    alert['image'] = None
            if not unknown:
                if now - last_seen >= 5.0:
                    acknowledged_while_present = False
                return
            last_seen = now
            if acknowledged_while_present or now - last_alert < 30.0:
                return
            sequence += 1
            last_alert = now
            alerts.appendleft({
                'id': f'unknown_{sequence}',
                'title': 'Unknown Person Detected',
                'desc': 'An unrecognized face is visible in BUDDY live camera.',
                'time': datetime.now().strftime('%I:%M %p'),
                'isUnread': True,
                'category': 'Security',
                'hasAction': True,
                'actionText': 'Verify & Enroll',
                'actionType': 'enroll',
                'image': preview,  # RAM-only thumbnail, expires after 2 minutes.
                '_created': now,
            })

    @app.get('/api/notifications')
    def buddy_notifications():
        sample()
        with lock:
            return jsonify([{k: v for k, v in a.items() if not k.startswith('_')} for a in alerts])

    @app.post('/api/clear_unknown_alert')
    def buddy_clear_unknown_alert():
        nonlocal acknowledged_while_present
        with lock:
            for alert in alerts:
                alert['isUnread'] = False
            acknowledged_while_present = True
        return jsonify(success=True, message='Unknown alerts acknowledged')

    @app.get('/api/unknown_alert_status')
    def buddy_unknown_alert_status():
        sample()
        with lock:
            return jsonify(count=len(alerts), unread=sum(
                1 for x in alerts if x['isUnread']
            ))
