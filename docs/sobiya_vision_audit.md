# Vision audit (Sobiya_AdvancedVision, Phases 1 and 2)

Branch: `Sobiya_AdvancedVision`, created from `origin/Ashandth_Voice` at
`7ccf320f`. Nothing was merged from any other branch. Existing code from
other branches was read through `git show` only.

## Base check

- `origin/Ashandth_Voice` exists and matches the reviewed voice commit.
- Working tree was clean. Voice tests on this base: 483 pass, 1 fails
  (`test_logging_utils.py::test_console_handler_created`, fails the same way
  before the voice work).
- Not run: anything under real ROS 2 or on the Pi.

## Where the vision code lives

| Branch | State |
|---|---|
| `buddy-ros2` and this base | `faceRecAndEmotion/` source files, but `models/face_model.keras` and `models/class_indices.json` are **0-byte placeholders**. No trained model, no database, no fall or object module. |
| `integration/sobiya-vision-app-v2` | Most complete: adds `fall_detection_module.py`, `object_learning_module.py`, `face_model.h5` (11 MB), `emotion_model.h5` (4.5 MB), `known_faces.json` (+ `.backup`), `object_database.json`. |
| `sobiya-object-fall-detect` | Same fall and object modules (byte-identical to v2), but inside a nested copy of the whole repo under `R26-SE-040-buddy-ros2/` and with two 12.8 MB mp3 files. Not to be copied wholesale. |
| `Sobiya` | Training data and old experiments: 28,709 train + 6,950 test jpgs (`data/`, looks like an expression dataset), 120 `faces_dataset` jpgs, 409 tracked files under `unknown_frames/`, a full Vosk model, 13 `.pyc` files. |
| `voice-+-face` | Older copy of the same modules, no fall or object code. |

## What each existing module really does

**Face recognition** (`face_recognition_module.py`)
- Detector: OpenCV Haar cascade (frontal faces only; weak for head turns,
  glasses glare, low light).
- Trained path: a Keras classifier over a fixed set of known people, accepted
  at confidence 0.95 and margin 0.40. A closed-set classifier cannot
  represent a new person without retraining, and it gives no distance
  to "someone I have never seen".
- Enrollment path: `get_embedding()` is hand-built intensity and edge
  statistics (grid means, histogram, Canny). It is **not** an identity
  embedding. It will confuse people in similar lighting and fail under
  lighting change. Threshold 0.88 was not calibrated on any data we hold.
- Verdict: reuse the detector idea and quality gates (`validate_face`);
  replace the identity embedding.

**Expression** (`emotion_detection.py`)
- Calls DeepFace (TensorFlow) if present, otherwise returns
  `neutral` with confidence **100** and keeps going. That makes a missing
  model look like a confident calm face. Must become "unknown".
- Labels are used directly as "emotion" and mapped to robot reactions
  (`ROBOT_REACTIONS`), including `fear -> backward` and `angry -> cautious`.
  Per the brief, expression must not drive motion.
- The repo also holds `emotion_model.h5`; its training provenance and
  accuracy are not documented anywhere I found.

**Fall detection** (`fall_detection_module.py`, 40 lines)
- Thresholds the grayscale frame at 75, takes the largest dark blob, and
  calls it a fall if the blob is wide, low in the frame and > 4000 px.
- It does not detect a person, estimate pose, use time, suppress false
  alarms, detect recovery or raise alerts. A dark rug, a sofa, a shadow or a
  person lying on a bed will trigger it. It is a placeholder, not a fall
  detector. Phase 14 needs a rebuild on pose plus temporal logic.

**Object learning** (`object_learning_module.py`)
- Reasonable core idea: MobileNetV2 (ImageNet) features, L2-normalised,
  averaged over 30 frames, matched by dot product at 0.70.
- Gaps: fixed centre crop instead of a detected object; no class-versus-
  instance separation; no confirmation, duplicate check, correction or
  delete; threshold 0.70 uncalibrated; best-match logic can return a name
  from a lower score than the best; needs TensorFlow and downloads ImageNet
  weights at first run (fails offline).

**Pipeline / glue**
- `main.py` opens the camera itself with `cv2.VideoCapture(0, cv2.CAP_DSHOW)`.
  `CAP_DSHOW` is Windows only, so this cannot open the Pi camera, and it
  breaks the one-camera-owner rule.
- No ROS 2 code in `faceRecAndEmotion/` at all (no publishers or
  subscribers). `robot_bridge.py` is a chat and command helper.
- `config.py` hardcodes Windows backend, creates directories on import, and
  mixes UI colours with logic.
- Runtime printing with emoji, a Tamil-transliterated comment, and
  `__pycache__` committed on `Sobiya`.

## ROS 2 interfaces already in `buddy_interfaces`

`RecognizedPerson` (person_id, display_name, known, confidence, bbox),
`EmotionObservation` (person_id, expression, confidence, source),
`SecurityEvent`, and actions `FollowPerson`, `ApproachCaller`. These are
enough to reuse for person and expression output. Gaps that need new
messages: object detections, activity or fall events, gestures,
interaction suggestions, and typed services for "what is this?" and
enrollment.

**Camera topic.** `/camera/image_raw` exists only on `Yohan_Mapping`
(`camera_node.py`, bgr8). This base does not provide it. I will build a
shared camera node here that publishes the same topic and format so the two
can be reconciled at integration, and only one of them may run.

## Privacy concerns found (nothing deleted, per brief)

1. `Sobiya` tracks 409 files under `unknown_frames/`: automatic captures of
   people who never enrolled. This breaks the consent rule. The base
   `.gitignore` already ignores that folder, but the history on `Sobiya`
   keeps them.
2. `known_faces.json` and `.backup` (about 0.7 MB) hold a stored identity
   record for one person in git history.
3. `frame_manager.py` still writes unknown faces to disk automatically
   whenever it runs. Must be disabled by default.
4. 36k dataset jpgs on `Sobiya` need their source and licence recorded;
   if they come from a public set they should not live in git.
5. Model binaries (`.h5`, `.keras`) are tracked on some branches.

Suggested safe migration (needs owner approval): move datasets and models
out of git to a release or shared drive, purge `unknown_frames/` and the
face database from history only after backing up and getting Sobiya's
agreement. I have not touched any of these.

## Sandbox facts that shape the build

- Available: OpenCV 5, NumPy, SciPy, scikit-learn, onnxruntime,
  mediapipe 0.10.32 (tasks API only; needs downloaded `.task` model files),
  pytest. Not available: ROS 2, TensorFlow (installable by pip), PyTorch,
  Picamera2.
- Model downloads: GitHub works from the shell, Google storage and Hugging
  Face do not. So pose and detector models must come from GitHub or pip
  wheels, and every model's source and licence will be recorded.
- No Raspberry Pi and no camera here. All camera, latency, FPS, thermal and
  power numbers must come from running the benchmark on the Pi. Expression
  accuracy can only be measured on the datasets I have, and says nothing
  about the Pi camera in a living room.

## Reuse decision

| Keep | Replace | Drop |
|---|---|---|
| Haar cascade as a fallback detector, face quality gates, async frame writer pattern, MobileNetV2 feature idea, existing msg types | identity embedding, fall detector, expression-to-reaction mapping, camera ownership, neutral-at-100% fallback | `ROBOT_REACTIONS` motion mapping, Windows camera backend, auto-save of unknown faces |
