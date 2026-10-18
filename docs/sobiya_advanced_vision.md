# Advanced Visual Intelligence and Proactive Companionship (Sobiya_AdvancedVision)

Component owner: Sobiya Anton Suresh. Branch: `Sobiya_AdvancedVision`
(created from `origin/Ashandth_Voice` at `7ccf320f`; nothing was merged from
any other branch).

**Status: software complete and tested with mocks, scripted data and public
datasets. NOT verified on the Raspberry Pi, with the Pi Camera Module 2, or
under real ROS 2.** Read "Known limitations" before relying on anything.

Three things this document never claims, and the code does not do:

- It does not claim Buddy can tell how a person feels from their face. A
  facial expression is a visible pattern, and the models here were trained
  to name patterns, not feelings. Buddy only ever treats it as a weak hint,
  and the person's own words always win.
- It does not claim a feature was physically tested when only mock tests
  ran. Section 20 lists what each kind of evidence is worth.
- It does not claim Buddy can approach or follow a person safely. Approach
  and follow exist only as policy code against a mock navigation adapter.

## 1. Purpose

Give Buddy a careful sense of sight on a Raspberry Pi 4 with one Pi Camera
Module 2 (no LiDAR, no depth camera, no time-of-flight sensor):

- recognise consenting familiar people, and never guess about strangers;
- notice a possibly concerning expression and, only for people who opted
  in, offer company as a question;
- learn objects the user teaches, and say honestly how sure it is;
- notice a possible fall and ask "Are you okay?" before any alert;
- recognise a few whole-arm gestures;
- hand all of it to the voice engine as advice. Vision never drives motors
  and never bypasses the safety supervisor.

## 2. Architecture

```
              Pi Camera Module 2
                     |
            camera_node (vision_camera)        the ONLY camera owner
                     |  /camera/image_raw (bgr8)
                     v
              vision_node (VisionNodeCore)
   +------------------------------------------------------+
   |  VisualContextManager  (priority + frame budget)     |
   |   1 faces   YuNet -> tracker -> SFace -> identity    |
   |   2 objects NanoDet -> stabilizer -> taught names    |
   |   3 pose    BlazePose -> posture/motion/gesture/fall |
   |   4 expression MobileFaceNet -> smoothing -> group   |
   |   5 scene   taught places (object memory, opt-in)    |
   |  Policies: greeting, proactive, fall check,          |
   |            spoken settings, consent, cooldowns       |
   +------------------------------------------------------+
        |            |              |             |
   /vision/people /vision/objects /vision/events /vision/status
   /person/recognized  /emotion/observation  /security/event (falls)
        |
        v                         /voice/status  (in: speaking, quiet, moving)
   voice_companion (Ashandth) <----------------------------+
        |
        v
   /movement/request -> safety_supervisor -> motors     (vision is not on this path)
```

Principles that shaped the design:

1. Perception only. The vision package imports no motor or audio-output
   code (checked by a test), and `VisualContextManager` has no actuation
   methods.
2. Consent before storing. Nothing about a person is stored before they
   confirm out loud. Nothing about an object is stored before a spoken yes.
3. Unknown stays unknown. Strangers are not named, not enrolled, and their
   expressions are not read.
4. Every number is measured or labelled as not measured.
5. Fail closed: no camera, no model, an unreadable settings file, or a
   stale frame means less vision, never a guess.

## 3. Reused existing modules

| Reused | From | How |
|---|---|---|
| `RecognizedPerson`, `EmotionObservation`, `SecurityEvent`, `SafetyState` messages | `buddy_interfaces` | Published or consumed unchanged |
| `FollowPerson`, `ApproachCaller` actions | `buddy_interfaces` | Mirrored by the mock navigation adapter only |
| Voice `EmotionFusion` | `buddy_core.voice` | Used read-only in tests to prove face hints only support stated feelings |
| `/voice/status` JSON | voice node | Read to avoid interrupting speech |
| MobileNetV2 feature idea, quality-gate idea | `faceRecAndEmotion` (old branches) | Re-implemented; no old code copied |

Not reused, with reasons, in `docs/sobiya_vision_audit.md`: the Haar-cascade
plus hand-built-feature face path (not an identity embedding), the
DeepFace "neutral at 100%" fallback, the expression-to-motion map, the
Windows-only camera backend and the automatic saving of unknown faces.

## 4. New modules

All under `buddy_ws/src/buddy_core/buddy_core/vision/`.

| Area | Modules |
|---|---|
| Camera | `frames`, `camera_sources`, `camera_service`, `camera_node`, `resources` |
| Faces | `faces`, `tracker`, `embedding`, `identity`, `face_pipeline`, `face_store`, `enrollment`, `greeting` |
| Expression | `expression`, `expression_tracker`, `multimodal` |
| Objects | `objects`, `object_stabilizer`, `object_embedding`, `object_teach`, `object_memory` |
| Body | `pose`, `activity`, `fall`, `gestures` |
| Policy | `proactive`, `approach`, `follow`, `vision_settings`, `vision_commands`, `privacy` |
| Wiring | `context`, `vision_factory`, `vision_node`, `demo` |
| Evaluation | `evaluation`, `expression_eval`, `object_eval`, `benchmark`, `eval_report` |
| Models | `models`, `onnx_runtime`, `vision_models.json` |

Interface additions to the shared `buddy_interfaces` package (additive, need
team approval): messages `VisionObject`, `VisionObjects`, `VisionPerson`,
`VisionPeople`, `VisionEvent`; services `VisionCommand`, `IdentifyObject`,
`TeachObject`, `WhereIs`.

## 5. Raspberry Pi Camera Module 2 setup (UNVERIFIED)

None of this has run on the robot.

1. Connect the camera ribbon cable to the Pi 4 CSI port with the Pi off.
2. Raspberry Pi OS: `sudo apt install -y python3-picamera2 python3-opencv`
   and check the camera with `rpicam-hello --list-cameras`. On Ubuntu 24.04
   the package names and libcamera setup differ; check the Raspberry Pi
   documentation for that OS.
3. Start the camera node (the only process allowed to open the camera):

   ```
   ros2 run buddy_core vision_camera --ros-args \
       -p backend:=picamera2 -p width:=640 -p height:=480 -p fps:=10.0
   ```

4. Check `ros2 topic hz /camera/image_raw` and that colours are right. If
   red and blue are swapped, change `pixel_format` (the code assumes
   Picamera2 `RGB888` is already BGR byte order, which is untested here).
5. Backends: `mock` (default, no hardware), `video` (replay a file),
   `opencv` (USB camera), `picamera2`.

The camera node publishes `/camera/image_raw` as `bgr8`, node name
`buddy_camera`, the same as Yohan's `camera_node` on `Yohan_Mapping`. Only
one of the two may run. The vision camera node refuses to start if a
`buddy_camera` node already exists and holds a lock file. Which camera node
the team keeps is an open decision.

The camera's horizontal field of view (62.2 degrees) is used only to turn a
box position into a bearing for the mock follow code.

## 6. Model details

Models are not in git. `scripts/fetch_vision_models.py` downloads them to
`~/.buddy/models/vision` (or `$BUDDY_VISION_MODELS`) from a pinned
`opencv_zoo` commit and checks SHA-256. A tampered or truncated file is
rejected.

| Key | Model | Licence | Used for |
|---|---|---|---|
| face_detector | YuNet 2023mar | MIT | face boxes and 5 landmarks |
| face_embedder | SFace 2021dec (128-d) | Apache-2.0 | identity |
| expression | MobileFaceNet FER 2022july | Apache-2.0 | 7 expression scores |
| object_detector | NanoDet-Plus 2022nov (COCO 80) | Apache-2.0 | objects and people boxes |
| pose | MediaPipe BlazePose 2023mar | Apache-2.0 | body landmarks |
| object_embedder_source | MobileNetV2 2022apr | Apache-2.0 | feature model for taught objects, built once by cutting the graph at the pooled features (needs the `onnx` package once) |
| person_detector, palm_detector, hand_pose | MediaPipe 2023 | Apache-2.0 | downloaded but NOT integrated |

A missing model leaves its stage out and the reason is reported in
`/vision/status` (`unavailable`), never silently ignored.

## 7. Dataset descriptions

- **FER2013 test split** (6,950 images, 48x48 grayscale, 7 classes): the only
  labelled dataset used. It was used only to evaluate expression models.
  It is a different camera and population from the robot. It has no labels
  for skin tone, age, glasses, occlusion, lighting or head angle.
- **Public-domain portraits** (skimage sample images and similar): smoke
  tests of the real face models. Two people only. Not an accuracy dataset.
- **Team `emotion_model.h5` and the `Sobiya` branch training images**: the h5
  was evaluated for comparison; its training data provenance is unknown and
  it is not used by Buddy.
- No multi-person face set, no home-object set and no fall recordings
  exist in the development sandbox, so identity, object and fall accuracy
  are **not measured**. Evaluation tools are provided to run on consented
  captures (section 21).

## 8. Face recognition

- YuNet finds faces; a tracker keeps a session id per visible person.
- SFace makes a 128-number embedding for good-quality faces, a few times per
  second, not every frame.
- Matching is open-set: below the floor is "unknown"; between the floor and
  the accept threshold, or too close to two people, is "uncertain" and the
  person is not named. A single frame can never name or un-name someone.
- Identity is decided only for people who consented. A person can turn
  recognition off, which removes them from the gallery without deleting
  them.
- The accept/margin thresholds are **uncalibrated**. Real-model smoke tests
  show a same-person pair above and a different-person pair below the
  accept threshold; that is not a false-accept rate.
- Enrollment (`enrollment.py`): the subject themselves must say yes to
  enrol, only one person may be in view, a different person stepping in
  aborts it, near-duplicate or poor frames are not counted, pose variety is
  required, and a final yes is needed before anything is saved. Nobody can
  consent on another person's behalf. A duplicate under a new name is
  blocked, and a user correction ("that is not me") blocks that face from
  matching the person again.

## 9. Facial-expression classification

- MobileFaceNet gives 7 scores. They are never published raw. A smoother
  needs several good frames over time, ignores one odd frame, treats two
  strong competing expressions as unknown, and reports `unknown` when
  conditions are poor (dark, blurred, turned away, possibly occluded).
- Only coarse groups leave the module: positive, neutral, negative,
  surprise. The words the voice engine understands are happy, neutral and
  surprised; "negative" is passed as-is and ignored by the voice side.
- Expression is read only for recognised people who opted in. Strangers are
  never analysed (a test checks the model is not even called).
- **Measured on FER2013 (cross-dataset, not the Pi camera):**
  accuracy 55.5%, macro-F1 0.445 for the zoo model Buddy uses. Per-class
  recall: happy 0.93, surprise 0.79, neutral 0.63, disgust 0.51, sad 0.32,
  angry 0.28, fear 0.07. For the coarse "negative" group, precision is 0.85
  and recall 0.47. About 12% of neutral faces were called negative.
  The full confusion matrices and per-class precision/recall are in
  `docs/evaluation/test_matrix.md`.
- **The model is badly overconfident:** expected calibration error 0.32,
  and when it says 90% or more it is right only about 69% of the time. After
  temperature scaling (T=3.85, fitted on half the data, evaluated on the
  other half) the error falls to 0.027. The shipped temperature is 1.0, so
  confidence values are NOT trustworthy as probabilities until calibrated on
  target-camera data. Gates (for example "0.30 or more") are therefore weak
  filters.
- The team's `emotion_model.h5` scored 52.9% accuracy (macro-F1 0.485) on the
  same set, with disgust recall 0; it was not adopted because its
  provenance is unknown and it needs TensorFlow.
- Results for different skin tones, ages, glasses, occlusion and lighting
  are **not available**: the dataset does not label them. This is a
  coverage gap, not a pass.

## 10. Object recognition and enrollment

- NanoDet-Plus finds COCO objects on the Pi at a modest rate. People are
  removed from the object list by default. A stabilizer needs three
  sightings to confirm an object and drops it after 2 s unseen.
- Detection quality on a real home is **unmeasured**; the one false positive
  seen in testing was a "sheep" on a microscope photograph.
- Teaching ("Buddy, this is my pen"): a hand-held object is chosen, several
  sharp, varied views are collected (blurry or faceless-crop rules apply),
  Buddy asks "Should I remember it as pen?", and only after a spoken yes
  is the object stored as feature vectors. No images are stored. A crop
  containing a face is refused. A near-duplicate asks "Is it the same
  thing?". Corrections ("no, that's not my pen") are kept as negative
  examples. Leading "my" or "the" is dropped from the name.
- Recognition ("what is this?") answers in hedged language with a number:
  "I think this is your pen. It looks about 91 percent similar to what you
  showed me." That number is a cosine similarity, **not a calibrated
  probability**. The accept (0.80), margin (0.05) and duplicate (0.88)
  thresholds are uncalibrated. Rough sandbox data: same object with
  augmentation 0.86 to 0.98, different objects 0.44 to 0.57.

## 11. Object memory

Off until the user turns it on by voice. When on, Buddy records the place,
a coarse region and the time an object was last seen, no images, at most one
update per 5 s per object, a 7-day retention and a 500-row cap. It never
records people. Answers: "I last saw your keys in the kitchen about 20
minutes ago. I can't tell whether they are still there." Taught places (at
least three views) allow rough scene recognition. Everything can be
forgotten or purged by voice.

## 12. Fall detection (logic only, thresholds unvalidated)

Not a medical or safety device. The sequence is: upright, fast drop to
lying, stays lying (6 s, or 12 s on furniture), Buddy asks "Are you okay?",
and if nobody answers within 20 s a `fall_alert` goes out as a
`SecurityEvent` (no image). "I'm okay" ends the check; "help" alerts
immediately. Slow lowering is recorded as resting, not a fall. There is a
10-minute alert cooldown, a 10 s grace for the person leaving view, and the
labels never include "sleeping" or any medical word.

Pose is estimated for each person box and, when the box is wide or
confidence is low, also from 90 and 270 degree crops so a person lying down
can be read. The lying confirmation window adapts to the pose sampling gap
(a bug found in Phase 21 where 1 Hz sampling never confirmed lying).
Fall detection needs the pose stage, so the `lightweight` profile has none.

If the camera fails during a fall check, the check is dropped, not
escalated. All thresholds were set from reasoning and tested on scripted
skeletons only. No real or staged fall has been recorded.

## 13. Human activity and gesture recognition

Posture labels: standing, sitting, lying, bending, upright_upper_body,
unknown, with dwell-time smoothing, motion and hand-raised tracking. Gestures
come from whole-arm geometry only: wave, raised hand, both hands up and
pointing (with a ray test for the object pointed at). A gesture never moves
the robot. Hand-shape gestures (thumbs up, open palm) are **not
implemented**: the palm and hand models are downloaded but not integrated
or validated.

## 14. Multimodal integration

The voice engine already weighs face hints. Vision publishes
`/emotion/observation` only for opted-in people, only when an expression is
confirmed, and publishes `unknown` with confidence 0 when it is not or the
person leaves or the camera stops (the voice side keeps the last hint
forever otherwise). What the person says always overrides the face; face
and voice that agree are capped; a disagreement gives no label; a stale or
unconfirmed face is ignored. Vision also reads `/voice/status` so it does
not offer anything while Buddy is speaking, in quiet mode, muted or moving.
The raw smoothed confidence is published (not the capped fusion value) so
the voice side's 0.6 threshold can apply; because that raw value is
overconfident, treat it as support only.

## 15. Proactive interaction policy

States: idle, person observed, context evaluation, waiting for invitation,
offered, conversation active, declined, cooldown, suspended.

An offer is always a question ("Hey, you seem a little quiet. Would you like
some company?"), never a statement about feelings. Requirements: a
recognised person who turned `proactive` on, a negative-group signal
that agrees with voice or stands alone, confidence at least 0.30, sustained
60 s, no quiet mode, not speaking, robot not busy, person not busy, no
emergency, fresh camera. Strangers never get an offer.

Cooldowns (per person, stored with the profile and deleted with it): a "no"
waits 2 h, a second "no" the same day 24 h, "I want space" 4 h, no answer
1 h, after a conversation 15 min, "stop asking" 30 days. At most two offers
a day with 30 min between. Silence counts as no. The user's answer, such
as "Yes, today was difficult", is handed on to the normal dialogue
(`pass_to_dialogue`) while a bare "yes" gets a short acknowledgement.

## 16. Navigation safety gating

`approach.py` and `follow.py` can only send a goal that contains a bearing,
never velocities. They refuse unless navigation reports available,
reliable distance, localisation OK, obstacle avoidance OK, movement allowed,
no emergency stop and (for follow) `follow_validated`. Further rules: the
person must be recognised and have invited it, bearing confidence at least
0.6, target seen within 1 s, stop distance at least 1.0 m, one request per
60 s, cancel on voice stop, decline, emergency, quiet mode, timeout, a
person lost for 2 s or degraded navigation. Follow never switches to
another person.

Only `MockNavigationAdapter` exists and neither module is wired to ROS.
With the real hardware (no distance sensor) every request is refused.
**Buddy cannot be described as approaching or following people safely until
real navigation and distance safety are validated.**

## 17. ROS 2 messages, topics and services

Topic names other than `/emotion/observation` are assumptions that need
confirming with their consumers.

| Direction | Name | Type |
|---|---|---|
| subscribe | `/camera/image_raw` | `sensor_msgs/Image` (bgr8 or rgb8) |
| subscribe | `/voice/status` | `std_msgs/String` (JSON) |
| publish | `/vision/people` | `VisionPeople` |
| publish | `/vision/objects` | `VisionObjects` |
| publish | `/vision/events` | `VisionEvent` |
| publish | `/vision/status` | `std_msgs/String` (JSON, no names) |
| publish | `/person/recognized` | `RecognizedPerson` |
| publish | `/emotion/observation` | `EmotionObservation` |
| publish | `/security/event` | `SecurityEvent` (fall alert, no image) |
| service | `/vision/command` | `VisionCommand` (spoken settings, replies) |
| service | `/vision/identify_object` | `IdentifyObject` |
| service | `/vision/teach_object` | `TeachObject` |
| service | `/vision/where_is` | `WhereIs` |
| service | `/vision/enable` | `std_srvs/SetBool` (camera on or off) |

`/vision/status` carries `awaiting_reply`. This is the contract with the
voice engine: while it is true (a fall check, an offered conversation, a
confirmation or a teaching session is open), the voice side should send the
user's utterance to `/vision/command` first. If the result has
`handled=false` or `pass_to_dialogue=true`, the voice engine also answers
normally. The voice engine needs a small change to do this and was not
modified.

Vision has **no** motion or navigation interface. Voice-affect input into
vision (to disagree with the face) needs an in-process provider or a topic
from the voice side; it is not wired.

## 18. Memory, consent and privacy

What is kept, where and for how long is defined in code
(`privacy.DATA_INVENTORY`) and checked by tests:

| Data | Where | Created when | Deleted by |
|---|---|---|---|
| Face vectors, name, settings, consent record, audit log | SQLite, mode 0600, `~/.buddy/vision/faces.db` | after the person confirms out loud | "forget my face", API, optional purge of people not seen for N days |
| Rejected-match vectors, check-in cooldowns | same file | when corrected or after an offer | with the profile |
| Taught object vectors | SQLite, mode 0600 | after a spoken yes | "forget the pen", "forget everything" |
| Object sightings (if enabled) | SQLite, mode 0600 | only when turned on | forget, purge, disable |
| Privacy settings | `settings.json`, mode 0600 | when changed | delete the file |
| Camera frames, expression, posture, fall state | memory only | while analysed | gone in seconds |

No photos are stored anywhere. Static tests fail if any vision code writes
an image, if a module outside an allowlist writes files, if the package
imports motor or audio-output code, or if anything but the camera modules
opens a camera. **Nothing is encrypted at rest**; protection is file
permissions only.

Spoken controls: "turn off the camera", "don't greet me", "stop checking
on me", "what do you know about me", "forget my face" (asks for a yes within
20 s), "turn off fall detection" (asks first). Choices survive restarts. If
the settings file cannot be read, vision starts with the camera OFF.

Fall detection and face detection run for anyone in view by default (faces
are detected but never named or stored without consent). Whether that
default is acceptable in a shared home is a decision for the team.

## 19. Installation commands (UNVERIFIED on the Pi)

```
# system
sudo apt install -y python3-picamera2 python3-opencv
# python (see deployment/vision_requirements.txt; versions not pinned)
pip install --break-system-packages numpy onnxruntime pyyaml onnx
# models (one time, needs network)
python3 scripts/fetch_vision_models.py
# build
cd buddy_ws && colcon build --packages-select buddy_interfaces buddy_core
source install/setup.bash
# run
ros2 run buddy_core vision_camera --ros-args -p backend:=picamera2
ros2 run buddy_core vision_node --ros-args -p profile:=balanced
```

Profiles: `lightweight` (faces and objects only), `balanced` (adds
expression, pose, scenes), `enhanced` (faster schedules). Leave `profile`
empty to keep the saved choice. A frame budget skips optional stages if the
pipeline runs long; face recognition always runs.

## 20. Offline demonstration

No camera, no ROS, no network and nothing saved:

```
python -m buddy_core.vision.demo --image portrait.jpg --repeat 20
python -m buddy_core.vision.demo --video clip.mp4 --profile balanced
```

One JSON line per frame (people, objects, events) and a summary that lists
any missing model. All models load from local files, and the runtime vision
code imports no network library (only the model download tool does). A full
run with the network physically unplugged on the Pi has not been done.

## 21. Evaluation and benchmarks

Kinds of evidence, from weakest to strongest as proof of how Buddy behaves
at home: `mock`, `synthetic_logic`, `stub_only`, `static_check`,
`real_model_smoke`, `dataset_measured`, and a real-camera result. **There
are no real-camera results.** The 30 required test areas, the tests behind
each, what was and was not verified and how to check each on the robot are
in `docs/evaluation/test_matrix.json`. The generated report with confusion
matrices is `docs/evaluation/test_matrix.md` (rebuilt with
`python -m buddy_core.vision.eval_report`, and a test fails if it is out of
date).

Tools for consented, labelled captures from the robot:

```
python -m buddy_core.vision.evaluation identity --root <folder per person>
python -m buddy_core.vision.expression_eval zoo --root <folder per label>
python -m buddy_core.vision.object_eval --root <folder with labels.json>
python -m buddy_core.vision.benchmark --out benchmark_results/pi.json
```

**Raspberry Pi benchmark results: none.** The only benchmark file,
`docs/evaluation/vision_benchmark_sandbox.json`, was measured on a 2-core
x86 machine and says `is_raspberry_pi: false`. Its figures (face detect
about 5 ms at 320 px, face embedding 16 ms, expression 21 ms, object detect
29 ms, pose about 20 ms a pass and about 62 ms with the sideways passes,
peak memory 329 MB) describe that machine only and must not be read as Pi
numbers. Only a run with `machine.is_raspberry_pi: true` counts.

## 22. Unit test results

Last full run in the sandbox: 986 passed, 1 deselected
(`test_logging_utils.py::test_console_handler_created`, which also fails on
the base commit and predates this work). Flake8 is clean on all vision code
and tests. The ROS node is tested against stubs generated from the repo's
`.msg` and `.srv` files; real ROS 2 was never available.

Reproduce:

```
cd buddy_ws/src/buddy_core
python -m pytest test -q --ignore=test/test_copyright.py \
    --ignore=test/test_flake8.py --ignore=test/test_pep257.py \
    --deselect test/test_logging_utils.py::test_console_handler_created
python -m flake8 buddy_core/vision
```

Real-model tests need the models fetched first; without them those tests
skip.

Reproducible manual procedure on the robot, in this order:

1. `python -m buddy_core.vision.benchmark` and keep the Pi file.
2. Camera: `ros2 topic hz /camera/image_raw`; unplug and replug it and
   watch `/vision/events` for `camera_lost` then `camera_restored`.
3. Enrol yourself, a second consenting person, and test a stranger: the
   stranger must never be named. Log false accepts and rejects.
4. Say "stop" while a person is in view and while the robot moves (wheels
   off the ground): voice stop must be unaffected by vision load.
5. Teach three objects and ask "what is this?" in different light.
6. Staged, supervised fall on a mat with consent, then a day of ordinary
   activity to count false fall questions.
7. Unplug the network and repeat steps 3 and 5.

## 23. Known limitations

- Nothing was run on the Pi, with the real camera, with real ROS 2, or
  with a real microphone together with vision.
- Expression accuracy is low (55.5% on FER2013, fear and anger poorly
  recognised) and the model is overconfident until calibrated on target
  data. No fairness evaluation across skin tone, age, glasses or occlusion
  was possible.
- Face and object thresholds are uncalibrated. No false-accept rate exists.
- Fall thresholds are reasoned, tested only on scripted skeletons. A fall
  check in progress when the camera fails is dropped.
- Hand-shape gestures are not implemented.
- Approach and follow are mock only and not wired to ROS.
- The camera node conflicts by design with Yohan's `camera_node`; only one
  may run.
- Topic and service names other than `/emotion/observation` are
  assumptions; the voice engine needs a small change to forward replies.
- No at-rest encryption. File permissions only.
- The `Sobiya` branch still holds 409 `unknown_frames` captures of people
  who did not enrol, `known_faces.json`, about 36,000 dataset images with
  no recorded licence, and model binaries. Nothing was deleted. Cleaning
  that history needs Sobiya's agreement.
- The 8-second voice motion cap and the team's use of `emotion_model.h5`
  are open questions for the team.

## 24. Future research

- Calibrate expression confidence on consented captures from this robot and
  re-tune the proactive gates; evaluate fairness across skin tone, age,
  glasses and occlusion with properly labelled data.
- Collect a small consented multi-person set to measure identity FAR/FRR
  and set thresholds.
- Compare a better pose model for lying people and train or tune fall
  thresholds on staged data.
- Validate hand-shape gestures with the downloaded palm and hand models.
- With a distance sensor or a validated navigation stack, revisit approach
  and follow.
- Encrypt the stores at rest; add a per-person data viewer.
- Measure end-to-end latency and thermal behaviour on the Pi and tune the
  profiles.
