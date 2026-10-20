"""Opt-in live expression timeline for the BUDDY Flask demo.

In-memory data only; intentionally lost on Flask restart. Estimates of visible
facial expressions are not evidence of a person's actual emotional state.
"""
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Lock

from flask import jsonify

_lock = Lock()
_records = {}  # person_id -> deque
_last = {}  # person_id -> last logging timestamp
_lookups = {}  # lower(display name) -> person id


def register_emotion_report(app, get_vision):
    @app.get('/api/emotion_report/<path:name>')
    def buddy_emotion_report(name):
        target = name.strip().casefold()
        v = get_vision()
        if v is None:
            return jsonify({'error': 'Vision unavailable'}), 503
        person = next((p for p in v.store.list_people()
                       if p['display_name'].casefold() == target), None)
        if person is None:
            return jsonify({'error': 'Person not found'}), 404
        if not person['settings'].get('expression'):
            return jsonify({'error': 'Expression tracking not enabled for this person'}), 403
        pid = person['person_id']
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        with _lock:
            reports = [r.copy() for r in _records.get(pid, ()) if
                       datetime.fromisoformat(r['timestamp']) >= cutoff]
        # These signals are NOT wired to live fall/music subsystems here.
        return jsonify({'person': person['display_name'],
                        'reports': reports, 'fall_detected': None,
                        'music_playing': None, 'last_played_time': None,
                        'music_target': person['display_name'],
                        'fall_status_available': False,
                        'music_status_available': False,
                        'source': 'live_confirmed_facial_expression_estimates',
                        'retention': 'memory_only_24_hours'})


def record_emotion_people(serialized_people, store):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    for item in serialized_people:
        pid = item.get('person_id')
        emotion = item.get('expression')
        conf = item.get('expression_confidence')
        if not pid or not emotion or not isinstance(conf, (int, float)) or conf <= 0:
            continue
        person = store.get_person(pid)
        if not person or not person['settings'].get('expression'):
            continue
        # Confirmed only: caller supplies expressions from the context's
        # published person snapshot, not raw network/model class guesses.
        with _lock:
            q = _records.setdefault(pid, deque(maxlen=1000))
            while q and datetime.fromisoformat(q[0]['timestamp']) < cutoff:
                q.popleft()
            prev = q[-1] if q else None
            same = prev is not None and prev['emotion'] == str(emotion).lower()
            if same and (now - datetime.fromisoformat(prev['timestamp'])).total_seconds() < 30:
                continue
            q.append({'emotion': str(emotion).lower(),
                      'confidence': float(max(0, min(1, conf))),
                      'timestamp': now.astimezone().isoformat(timespec='seconds')})
