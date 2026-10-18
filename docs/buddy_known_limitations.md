# BUDDY known limitations and open items

Written to be read before trusting anything else in this branch.

## What was not run

- **No ROS 2 anywhere in the build environment.** There has been no
  `colcon build`, no node started, no topic or service exercised on a real
  graph, no QoS or DDS check, no launch file executed. ROS wrappers were
  tested against strict stubs generated from the repo's own `.msg`/`.srv`
  files, which catches wrong field names and types, not transport problems.
- **No Raspberry Pi 4 measurement of any kind**: CPU, RAM, temperature,
  frame rate, latency, power, or audio underruns. The soak figure below is a
  desktop-class sandbox process.
- **No physical test**: microphones, speaker, camera, ultrasonic sensors,
  motors and the emergency stop button were never involved.
- **Models are faked** in the end-to-end tests (face, object, pose, embedder,
  speech recogniser, synthesiser). Recognition accuracy, fall-detection
  validity and speech accuracy are whatever the component owners measured;
  nothing here adds to or confirms them.

## Behaviour that is deliberately limited

- **"Come here" / person approach is refused on hardware.** There is no
  validated person-distance estimate and no navigation command for "go to a
  person". In the `simulation` profile the decision logic runs against a mock
  adapter that moves nothing and "finishes" after a timer; that is a demo of
  the refusal rules, not of approach.
- **Face enrolment by voice is not integrated.** Vision exposes no
  enrolment service. "This is my friend Sam" is explained, not acted on.
- **A spoken stop does not latch the supervisor's emergency stop.** It
  cancels navigation and sends STOP; the latch is for the hardware button
  and `/emergency_stop`.
- **Voice has no destination grammar by default.** "Go to the corner"
  works only for names listed in `navigation.locations`; "go to the kitchen"
  is ordinary conversation.
- **Spoken patrol needs a route.** The voice node cannot name one; the
  coordinator uses `default_patrol_route` (set for simulation, empty on the
  robot, where the physical config defines no routes). Empty means refused in
  words.
- **A request navigation silently refuses** (it only logs) is detected when
  its id shows up `idle`; the reason text is not available to Buddy, so the
  spoken message is a generic "I couldn't start that."
- **"Where did you last see my X?"** needs object memory switched on in
  vision; otherwise Buddy says it is not keeping track.
- **"Do you remember where my phone is?"** is not recognised as a location
  question.
- **The teach-an-object intent is heuristic** (abstract-word list, "this is"
  only). It may miss a valid phrasing; it is built to miss rather than to
  start teaching on ordinary remarks.
- **Obstacle validity is per direction**: if one of a pair of sensors fails
  and the other still reads, that side counts as measured.
- **A proactive offer dropped by the speech arbiter** (quiet mode,
  conversation in progress, cooldown) is not reported back to vision, so
  vision may consider it made.
- **Voice quality is not evaluated.** With no language model in the mock,
  conversation replies are fallbacks; Tests A and B pass only on "spoke and
  did not move".
- **Tamil** is not supported offline (voice component).

## Risks to check first on the robot

1. Two nodes fighting over the camera or microphone: the audit reports it,
   but the safest check is `ros2 node list`.
2. The 3-second gateway timeout adds latency to every non-command utterance
   if the coordinator is slow; it fails open, so conversation still works.
3. `/movement/request` has several publishers by design (navigation,
   coordinator, voice STOP). The supervisor, not the topic, is the gate.
4. Stale-data limits (`contracts.STALE_AFTER_S`) were chosen from the
   publish rates in the code, not measured; they may need tuning on the Pi.
5. Navigation holds off for 1 s when another source publishes a non-STOP
   movement on `/movement/request`. The coordinator refuses raw movement
   while a goal is running, so this should only show up if something outside
   the coordinator publishes there.

## Long-run result (Test Q), sandbox only

30.3 simulated minutes, 0 crashes, 0 collisions, 0 state violations; heap grew from 1.37 MB to 4.39 MB over 24 cycles (about 144 KB per cycle, cause not found). Command:
`python3 buddy_ws/src/buddy_orchestrator/tools/soak.py --minutes N`.
It is a mock soak and says nothing about a Pi.
