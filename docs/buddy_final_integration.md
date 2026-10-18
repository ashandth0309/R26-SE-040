# BUDDY final integration: voice + vision + navigation

Branch: `testing-integration`
Base: `origin/Sobiya_AdvancedVision` @ `15443042` (voice + vision)
Merged: `origin/Yohan_Mapping` @ `6247cf99` (navigation), no conflicts
Not merged anywhere: `main` and `buddy-ros2` are untouched.

**Honest status in one paragraph.** The three components are connected in
software, the interface contracts are written and tested, a central
coordinator enforces the motion, speech and camera rules, and an in-process
mock of the whole system passes the acceptance scenarios that can be mocked.
Nothing has been built or run under ROS 2, nothing has run on the Raspberry
Pi, and nothing has touched the robot. Read `buddy_known_limitations.md`.

## What was added

| Area | Change |
|---|---|
| `buddy_orchestrator` (new) | `buddy_coordinator` node, six launch files, config, interface contracts, tests, mock end-to-end harness, soak tool |
| `buddy_core` voice | utterance gateway, speech intake `/voice/say`, interposed movement/navigation topics with direct STOP, corrected navigation-state table |
| `buddy_core` safety | refuses movement toward a side with no valid reading |
| camera nodes | one camera owner across vision and navigation |
| `buddy_interfaces` | `SpeechRequest.msg`, `RouteUtterance.srv` |
| docs | this file and the five named in the README section |

## End-to-end flows (all mock-tested; see the matrix)

1. **Voice → navigation.** Utterance → engine splits stop / command /
   conversation → router (confidence check) → intake topic → coordinator
   gate → navigation → supervisor → motors → status → coordinator → spoken
   result. Ambiguous or negated sentences, low-confidence commands and
   unknown destinations never produce a motion request.
2. **Voice → vision.** Finished utterance → gateway → coordinator intent →
   vision service → result or honest refusal → speech. A camera question
   never reaches the language model.
3. **Teaching.** "This is my red pen" → vision teaching dialogue (ask for
   views, confirm) → stored → later recognised, with uncertainty wording.
4. **People.** Recognised person named only when one person is in view and
   identity is `recognized`.
5. **Companion.** Greetings, offers and fall check-ins are spoken through
   the arbiter with cooldowns and quiet mode. A suspected fall is a question
   ("Are you okay?"), never an automatic alarm or movement.
6. **Interruption.** "Stop" works through a direct route that does not need
   the coordinator. A latched emergency stop outranks everything and recovers
   only to IDLE.
7. **Patrol.** Coordinator supplies the configured route; result announced.

## Commits on this branch (after the base)

`f89af263` merge Yohan_Mapping · `ef5a32be` safety validity fix ·
`12819886` single camera owner · `c42427ca` voice hooks ·
`a0ad7a0e` coordinator core · `23451d3d` coordinator node, launch, tests ·
`d2f33908` mock end-to-end harness and scenarios A-P · and the commits that
follow it (more tests, tightened intent rules, soak tool, these docs).

## Results

- Orchestrator suite: 288 passed in 28.91s (mock + stub, no ROS)
- `buddy_core`: 1033 passed, 11 skipped in 5.86s
- `buddy_navigation`: 173 passed in 9.48s
- Lint: flake8 (max line length 99) clean on everything added or changed.
- Soak (Test Q, **sandbox mock, not a Pi**):
  - 30.3 simulated minutes, 24 cycles of a scripted mix (chat, vision question, moves, destination, patrol, stop, emergency stop), 210 s wall time in this sandbox.
  - 0 crashes, 0 collisions, 0 state-machine violations, speaker queue never above 0, at most 1 pending navigation request.
  - Python heap 1.37 MB after cycle 3, 4.39 MB after cycle 24 (about 144 KB per cycle of growth that is not explained; the test harness's own recordings, 308,105 rows, were dropped each cycle and an earlier run that kept them grew to 57 MB). Not diagnosed further; worth watching on a longer run.
  - Peak process memory 154 MB. CPU time is not comparable to a Pi.

## Reproduce

```bash
cd buddy_ws/src/buddy_orchestrator && python3 -m pytest test -p no:logging
cd ../buddy_core && python3 -m pytest test -p no:logging \
   --ignore=test/test_copyright.py --ignore=test/test_flake8.py --ignore=test/test_pep257.py
cd ../buddy_navigation && python3 -m pytest test -p no:logging \
   --ignore=test/test_copyright.py --ignore=test/test_flake8.py --ignore=test/test_pep257.py
python3 buddy_ws/src/buddy_orchestrator/tools/soak.py --minutes 10
```

## Next steps, in order

1. `colcon build` and `colcon test` on a ROS 2 Jazzy machine (Level 5).
2. `full_simulation.launch.py` and the scenario list on a real graph.
3. Pi bench with wheels off the ground (Level 6), recording CPU, RAM,
   temperature; tune stale limits and the vision/voice profiles.
4. Validate a person-distance estimate before ever enabling approach.
5. Add a face-enrolment service to vision, then wire "this is my friend Sam".
6. Level 7 only after the above are recorded.
