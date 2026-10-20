# Integrating head tracking with Ashandth's voice and Sobiya's vision

This branch merges **nothing** from `Ashandth_Voice`, `Sobiya_AdvancedVision`,
`buddy-ros2` or any integration branch. It depends only on what is in
`Yohan_Mapping` and reads the other branches' interfaces through plain data +
duck typing. Facts below were read directly from those branches (read-only).

## 1. What the other branches provide / lack

**Sobiya_AdvancedVision**
- `/vision/people` (`buddy_interfaces/VisionPeople` of `VisionPerson`): `person_key`,
  `bbox_x/y/width/height` (**pixels, top-left + size**), `known`, `gestures[]` (strings).
  `person_key` appears to be the tracker label (her `tracker.py` says such ids are "transient ... not an identity"; not confirmed for known people). `display_name`
  is filled only for people who agreed to be remembered. Head tracking reads **only** `person_key`
  and the box, and never publishes them.
- `VisionPeople.stamp` is the **publish** time (`get_clock().now()` in `_publish`), not the camera
  capture time. Head tracking stamps observations on receipt, so camera latency is not measured.
- Whether `bbox` is a face box or a whole-person box is **not confirmed** (she tracks "faces or
  people"). If it is a body box, set `face.vertical_aim` to about 0.12 so the head aims at the face area.
- `gestures.py`: `wave`, `raised_hand`, `both_hands_up`, `pointing` from **body pose** only.
  `pose.py` exposes wrists (`L_WRIST=15`, `R_WRIST=16`) with pixel coords and visibility but
  these are **not published over ROS**. **No hand-landmark / fingertip detection exists**; the
  MediaPipe palm detector is in her model manifest but not integrated.

**Ashandth_Voice**
- `/voice/command` (`VoiceCommand`) has `wake_word_detected`, but `voice_node._on_command` always
  sets it `False`. A real wake-word signal therefore needs a small change on her side (below).
- `/voice/status` JSON (1 Hz) contains `state` (`BUDDY_SPEAKING` when talking), `mic_topology`
  (`stereo`, `two_devices`, `single`, `simulated`, `none`), `awake`. Head tracking uses `state`
  for the no-turn-while-speaking rule and `mic_topology == "stereo"` as the synchronisation hint.
- The voice node owns the microphone (it refuses to start beside `buddy_voice`). Head tracking
  must **not** open the microphone; it takes direction estimates from a topic.

## 2. Exact changes needed (on the other branches, NOT made here)

| # | Owner | Change | Why |
|---|---|---|---|
| 1 | Ashandth | In `voice_node._on_command` (or where the wake word fires) publish a `VoiceCommand` with `wake_word_detected = True` at the moment of wake-word detection (can be an otherwise-empty command). | Without it `/voice/command` never signals a wake word. Alternative: publish `wake` on `/head/command`. |
| 2 | Ashandth | Only if `mic_topology == "stereo"`: add a small publisher in the node that owns the microphones that runs `SoundDirectionEstimator.estimate(ch0, ch1, fs, synchronized=True)` on the ~0.25 s windows after a wake word and publishes JSON to `/head/sound_direction` (schema in `ros_adapter.py`). | Only the mic owner can see the audio. If topology is not stereo, publish nothing: the fallback applies. |
| 3 | Sobiya | Publish wrist/fingertip points + gestures with side as JSON on `/head/hand_observation` (schema in `ros_adapter.hand_from_json`), using the existing `Pose.point(L_WRIST / R_WRIST)` and `GestureRecognizer` output. Keep `fingertips` empty until a hand-landmark model is integrated. | `VisionPerson` has no wrist points or gesture side. |
| 4 | Sobiya | (Optional) carry the camera capture time in the people message or header. | Enables true latency compensation. |
| 5 | Sobiya | (Optional) hand model: see section 4. | Real fingertip following. |

## 3. Merge / bring-up order

1. Merge this branch into the integration branch (it only adds files plus the `setup.py`
   entry point/launch glob and three `package.xml` exec dependencies).
2. Merge/rebase Sobiya's and Ashandth's branches on top. Resolve `package.xml`/`setup.py` by union.
3. `colcon build --packages-select buddy_interfaces buddy_core`.
4. `ros2 launch buddy_core head_tracking.launch.py` (mock). Check `/head/status` while Sobiya's
   vision is running: faces move the *mock* head. Confirm the log has no
   "VisionPeople not available" warning.
5. Apply changes 1-3 above; check `/head/events` shows `WAKE_WORD_DETECTED` when you say "Buddy".
6. Only then follow [head_tracking_physical_validation.md](head_tracking_physical_validation.md).

Locomotion/navigation (Yohan's `buddy_navigation`, `/movement/request`) is untouched; head tracking
imports nothing from `motors` or navigation and publishes only `/head/*` (a test enforces this).

## 4. Fingertip tracking: what would be needed

Reuse Sobiya's staged model manifest (`vision_models.json`, pinned `opencv/opencv_zoo` commit
`47534e27...`): the MediaPipe **palm detector** is already listed; a matching
**hand-landmark** model (21 points; index fingertip = landmark 8) from the same source would be
needed. Before adding it: (a) verify the licence file at the pinned commit (the MediaPipe models are
believed to be Apache-2.0 - **confirm, do not assume**), (b) record provenance (URL, commit, SHA-256),
(c) do **not** commit the `.onnx` (download at setup like her other models), (d) measure latency/CPU on
the Pi 4B (not measured here), (e) validate fingertip error against known target positions. Run it only on
the activated person's crop to keep CPU low.

## 5. Privacy

No images, audio or recordings are stored or published. Only track ids and boxes enter the head
logic, used transiently; status/events carry none. Nothing in this branch writes face databases,
audio or models.
