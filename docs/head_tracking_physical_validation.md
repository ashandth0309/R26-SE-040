# Head tracking - physical validation guide

Do these **in order**, stopping at the first failure. Until the checklist is complete,
head tracking is **not** hardware-verified, no matter how many tests pass.
Keep a hand on the power switch for steps 3-9; keep the robot on a stand with
wheels/legs clear and the neck free to move.

Record every result (date, value, pass/fail) in a copy of this file.

## 0. Preparation (no power to servos)

- [ ] `colcon build --packages-select buddy_interfaces buddy_core` on the Pi succeeds.
- [ ] `python3 -m pytest buddy_ws/src/buddy_core/test/test_head_*.py` passes on the Pi.
- [ ] Mechanical: head can move by hand through its full travel without binding;
      cables have slack; the camera is fixed to the head (not the body).
- [ ] `adafruit-circuitpython-servokit` installed; `i2cdetect -y 1` shows `0x40`.

## 1. Mock run on the Pi

`ros2 launch buddy_core head_tracking.launch.py` -> log says `mode: mock`.
`ros2 topic echo /head/status` shows `state: IDLE`. Publish `wake` to
`/head/command` -> state `LISTENING`. [ ] pass

## 2. Measure the mechanics (power ON the servo supply, nothing commanded yet)

With the servos **released** (unpowered or not driven), move the head by hand to each
mechanical stop and measure the servo angle each corresponds to (use a protractor or
`python -m buddy_core.head_tracking.bringup nudge` in 1-2 degree steps, never forcing).
Fill in:

| Axis | Physical min | Centre (facing forward / level) | Physical max |
|---|---|---|---|
| pan | | | |
| tilt | | | |

- [ ] Update `hardware.neck.*.min/center/max_angle_deg` in `config/buddy.yaml`
      **inside** the measured travel. (Current file has tilt centre == tilt max.)
- [ ] Confirm the PCA9685 channels (`pan.channel`, `tilt.channel`) by nudging.

## 3. First motion, tiny steps

Set `buddy.yaml` `hardware.enabled: true`, `hardware.simulation: false`,
`hardware.neck.enabled: true`; in `head_tracking.yaml` set `mode: hardware`,
`hardware_arm: true`, temporarily `calibration_confirmed: true`, `pan_sign: 1`,
`tilt_sign: 1`; `export BUDDY_HEAD_HARDWARE_ARMED=1`.

```
python -m buddy_core.head_tracking.bringup limits
python -m buddy_core.head_tracking.bringup center
python -m buddy_core.head_tracking.bringup nudge --axis pan  --deg 5
python -m buddy_core.head_tracking.bringup nudge --axis tilt --deg 5
```

- [ ] `center` points the camera straight ahead and level (adjust the centre angles if not).
- [ ] No grinding, buzzing against a stop, or brown-out of the Pi (check supply sag).
- [ ] **Direction.** For `nudge --axis pan --deg +5`, did the CAMERA VIEW move toward image
      right? yes -> `pan_sign: 1`; no -> `-1`. For tilt `+5`, toward image down? yes -> `1`; no -> `-1`.
      Write them into `head_tracking.yaml`. Repeat to confirm the opposite sign moves the other way.

## 4. Limits

Nudge ±10 degrees repeatedly toward each end (never past the margin the tool reports).
- [ ] The head never touches a hard stop at the reported limits.
- [ ] Widening/narrowing: adjust `limit_margin_deg` (default 3) if the head still touches.

## 5. Camera field of view and axis alignment

Place a marker at a known angle from the optical axis (e.g. tape on a wall at a measured
distance). Compute `horizontal_fov_deg` / `vertical_fov_deg`. Put them in `head_tracking.yaml`
(`face.servo` and `hand.servo`). [ ] values recorded: H ____ V ____

Check the camera is not rotated: a vertical edge must stay vertical in the image when panning.

## 6. Face following (needs Sobiya's vision on the Pi)

Stand about 1.5 m away, then walk slowly left/right, crouch/stand.
- [ ] The head turns toward you, face ends within ±10% of image centre.
- [ ] No oscillation (the head settles; watch `/head/status` pan/tilt for 20 s).
- [ ] Hold still: the servos do not jitter or buzz (dead zone works).
- [ ] Two people: the head keeps the first person when the second walks closer.
- [ ] Cover the camera / leave the frame: head holds ~1.5 s, then returns to neutral.
- [ ] Tune `kp`, `smoothing_alpha`, `max_step_deg`, velocity limits one at a time; note final values.

## 7. Emergency stop and fault

- [ ] Trigger the real emergency stop (`/safety_state` with `emergency_stop_active`): the head stops at once
      and ignores faces/hands/voice; clearing returns to `IDLE` and the head does not jump.
- [ ] `stop` then `resume` on `/head/command`.
- [ ] With the head tracking a face, **unplug the PCA9685 I2C/power**: state becomes `FAULT`, no spam in the log.
      Re-plug, `reset_fault`: head re-centres.
- [ ] Power-cycle: confirm what the head does at boot (it should not move until the node starts).

## 8. Microphones

First: do the two lavalier microphones appear as **one device with two channels**?
`python -m buddy_core.head_tracking.mic_check --list`. If they are two devices, they are
**not synchronized -> stop**: use the fallback (wake word + camera) and leave
`sound.enabled: false`. A stereo capture interface (one clock) would be required.

If one stereo device:
1. Measure centre-to-centre spacing; set `sound.mic_spacing_m`.
2. `python -m buddy_core.head_tracking.mic_check --device N --spacing D --seconds 6` while someone
   talks from 0 (front), -45 (left), +45 (right) degrees at 1.5 m, 5 repeats each.
3. Pass if: median within ±15 degrees of truth, spread < 15 degrees, correct sign, and ≥ 50% of windows accepted.
   Then measure channel skew with a clap at 0 degrees and set `delay_offset_s` / `swap_channels`.
4. Noise: repeat with TV/fan noise; confirm estimates are *rejected*, not wrong.
- [ ] Results table (true angle vs median vs spread) saved.
- [ ] Only then set `sound.enabled: true`. Note: sound localisation needs a node that owns the
      microphones to publish `/head/sound_direction` (see integration guide).

## 9. Voice call end to end (needs Ashandth's wake word + 8)

Call "Buddy" from left, right, front. While Buddy talks, call again.
- [ ] Head turns toward the caller within the accuracy measured in step 8, then the camera refines it.
- [ ] While Buddy is speaking the head does not turn.
- [ ] Repeated calls within 2.5 s do not cause repeated turns.

## 10. Hand following

- [ ] Raise a hand (activation gesture) while being tracked: `HAND_ACTIVATED` with `kind: wrist_fallback`.
- [ ] The head follows the wrist left/right/up/down within limits.
- [ ] Both hands up / `hand_off` / dropping the hand ends it and returns to face tracking.
- [ ] A different person's hand does not take over.
- [ ] **Do not describe this as finger tracking.** Only after a fingertip model is integrated and
      validated (point at targets of known position, measure error) may `index_fingertip` be claimed.

## 11. Soak

Run 30 minutes with people moving. [ ] No faults, no growth in memory/CPU (`top`),
servos not hot, no drift of the neutral pose. Record CPU % on the Pi 4B.

## Stop immediately if

the head hits a stop, a servo buzzes or heats, the Pi browns out, the head moves without
a target, or any behaviour differs from this guide. Set `mode: mock`, unset
`BUDDY_HEAD_HARDWARE_ARMED`, and investigate.
