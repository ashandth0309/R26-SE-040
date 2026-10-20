"""Windows-only live BUDDY voice/vision demonstration (no ROS, no motors).
Uses existing Vosk/Piper models, webcam owned by running Flask server.
"""
import json
import os
import queue
import re
import time
import wave
import io
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel
from piper import PiperVoice

from buddy_core.voice.buddy_object_voice_bridge import ObjectConversation
from buddy_core.voice.dialogue_manager import ConversationContext

ROOT = Path(__file__).resolve().parent
VOSK_PATH = ROOT / 'models' / 'vosk-model-small-en-us-0.15'
PIPER_PATH = ROOT / 'models' / 'piper' / 'en_US-lessac-medium.onnx'
MIC = int(os.getenv('BUDDY_MIC_DEVICE', '1'))
SPEAKER = int(os.getenv('BUDDY_SPEAKER_DEVICE', '3'))
RATE = 16000

if not VOSK_PATH.is_dir() or not PIPER_PATH.is_file():
    raise SystemExit('Vosk/Piper model not found in the project models directory.')

SetLogLevel(-1)
print('Loading offline Vosk model...', flush=True)
model = Model(str(VOSK_PATH))
rec = KaldiRecognizer(model, RATE)
print('Loading offline Piper voice...', flush=True)
voice = PiperVoice.load(str(PIPER_PATH))
bridge = ObjectConversation(None)
context = ConversationContext()
frames = queue.Queue(maxsize=80)
listening = True
awake_until = 0.0


def callback(indata, count, time_info, status):
    if not listening:
        return
    try:
        frames.put_nowait(bytes(indata))
    except queue.Full:
        pass


def speak(text):
    global listening, rec
    if not text:
        return
    print('BUDDY:', text, flush=True)
    listening = False
    try:
        with io.BytesIO() as buffer:
            with wave.open(buffer, 'wb') as wav:
                voice.synthesize_wav(text, wav)
            buffer.seek(0)
            with wave.open(buffer, 'rb') as wav:
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                rate = wav.getframerate()
        sd.play(samples, rate, device=SPEAKER, blocking=True)
    except Exception as exc:
        print('AUDIO ERROR:', type(exc).__name__, str(exc), flush=True)
    finally:
        sd.stop()
        while True:
            try:
                frames.get_nowait()
            except queue.Empty:
                break
        rec = KaldiRecognizer(model, RATE)
        listening = True


def normalized(text):
    """Strip conversational lead-ins without altering the object name."""
    t = ' '.join(re.sub(r"[^a-z0-9' ]", ' ', text.lower()).split())
    t = re.sub(r'^(?:hey |hello )?buddy\b\s*', '', t)
    for _ in range(3):
        newer = re.sub(r'^(?:okay |ok |well |so |good |great |now |please |then |alright |and )+', '', t)
        newer = re.sub(r'^(?:tell me |can you tell me |could you tell me |please tell me |show me )+', '', newer)
        if newer == t:
            break
        t = newer
    return t.strip()


def handle(raw):
    global awake_until
    raw = ' '.join(raw.lower().split())
    if not raw:
        return
    print('HEARD:', raw, flush=True)
    # Highest priority: no movement is possible in this demo.
    if re.search(r'\b(stop|halt|freeze)\b', raw):
        awake_until = 0
        speak('Okay, stopping. No motors are connected.')
        return
    if raw in ('buddy', 'hey buddy', 'hello buddy'):
        awake_until = time.monotonic() + 45
        speak('Yes?')
        return

    state = bridge.api('GET', '/api/objects/teach/status')
    teaching = bool(isinstance(state, dict) and state.get('active'))
    contains_wake = bool(re.search(r'\bbuddy\b', raw))
    if contains_wake:
        awake_until = time.monotonic() + 45
    elif time.monotonic() > awake_until and not teaching:
        return

    phrase = normalized(raw)
    identification = bool(re.search(r'\b(?:what is this|what is that|what\'s this|what\'s that|identify this|recognize this|recognise this)\b', phrase))
    teaching_phrase = bool(re.fullmatch(r"(?:this is|it is|it's|that is|that's) (?:my |a |an |the )?.+", phrase))
    short_reply = phrase in ('yes', 'yes please', 'yes remember it', 'remember it', 'save it',
                             'no', 'no thanks', 'cancel', 'cancel teaching', 'never mind')

    if identification or teaching_phrase or (short_reply and (teaching or bridge.teaching_offered)):
        if identification:
            clean = 'what is this'
        elif teaching_phrase:
            clean = phrase
        else:
            clean = phrase
        reply = bridge.respond(clean, context)
        speak(reply.text)
        awake_until = time.monotonic() + 45
    elif phrase in ('thank you', 'thanks', 'good job', 'well done'):
        speak('You are welcome!')
    elif phrase in ('hello', 'hi', 'how are you'):
        speak('Hi! I am here, and ready to look at an object.')
    else:
        # Do not invent object recognition or activate unspecified motor commands.
        print('INFO: Speech heard but not a supported demo command.', flush=True)


print('\nREADY. Say "Buddy" then "What is this?"')
print('Try "This is my bottle" to teach, and "Yes" to confirm.')
print('Existing Flask server must remain running. Ctrl+C to stop.\n')
try:
    with sd.RawInputStream(samplerate=RATE, blocksize=4000, device=MIC,
                           dtype='int16', channels=1, callback=callback):
        while True:
            raw = frames.get()
            if rec.AcceptWaveform(raw):
                result = json.loads(rec.Result())
                handle(result.get('text', ''))
except KeyboardInterrupt:
    print('\nBUDDY voice demo stopped (Ctrl+C).')
