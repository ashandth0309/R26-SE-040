# Sobiya_AdvancedVision: phase-by-phase report

One entry per spec phase (grouped where several shipped together). "Tested"
means automated tests with mocks, scripted data or a public dataset. **No
phase has been tested on the Raspberry Pi, with the Pi Camera Module 2, or
under real ROS 2.** Test counts are for the whole suite at the end of the
phase group (the final run is 986 passed, 1 pre-existing unrelated
failure deselected).

| Phases | What shipped | Key files | Working (tested with mocks/data) | Not verified | Extra Pi dependencies | Safety and privacy limits |
|---|---|---|---|---|---|---|
| 1-2 | Audit of existing vision code, branch from `Ashandth_Voice` | `docs/sobiya_vision_audit.md` | Findings on old modules and privacy concerns | Nothing deleted; Sobiya-branch cleanup needs her approval | none | Reports `unknown_frames`, face DB and dataset concerns only |
| 3 | Shared camera pipeline | `frames`, `camera_sources`, `camera_service`, `camera_node`, `resources` | One owner, multi-subscriber stream, reconnect with backoff, stale-frame gate, privacy off | Real Pi Camera 2, colour order, fps; conflicts with Yohan's `camera_node` | `python3-picamera2` | Camera off releases the device; no frames written |
| 4-5 | Face detection, tracking, consent-based identity | `faces`, `tracker`, `embedding`, `identity`, `face_pipeline`, `face_store`, `enrollment`, `greeting` | Open-set matching, strangers stay unknown, consent-gated enrollment, correction, deletion | Real FAR/FRR, glasses, lighting, angles (no data) | YuNet, SFace models | Nothing stored before spoken yes; 0600 database; no photos |
| 6 | Expression with smoothing and uncertainty | `expression`, `expression_tracker`, `expression_eval` | Smoothing, unknown states, FER2013 evaluation with confusion matrices | Pi-camera accuracy; fairness slices (no labels) | MobileFaceNet model | Only for opted-in recognised people; never claims true emotion; model is overconfident |
| 7 | Voice/vision fusion | `multimodal` | Words beat faces; disagreement gives no label; clears stale hints | Real voice features with real face | none | Face is support only |
| 8 | Proactive interaction | `proactive` | Consent, sustain time, cooldowns, daily limit, refusals | Whether timing feels right | none | Offers are questions; strangers never offered |
| 9 | Approach policy with mock navigation | `approach` | Refuses unless every capability holds; bearing-only goal | Real navigation | none | Cannot claim safe approach; real hardware always refused |
| 10 | Object detection | `objects`, `object_stabilizer`, `object_eval` | Detector, stabilizer, evaluation harness | Real home accuracy (no dataset); a false "sheep" was seen | NanoDet model | People removed by default |
| 11 | Taught objects | `object_embedding`, `object_teach` | Teach/confirm/duplicate/correct/forget; hedged answers with similarity | Thresholds uncalibrated; Pi speed | MobileNetV2 model, `onnx` once | Stored only after yes; faces refused; vectors only |
| 12 | Object memory | `object_memory` | Place/time memory, honest "last seen" | Place recognition accuracy | none | Off by default; no people; 7-day retention |
| 13-15 | Posture, fall, gestures | `pose`, `activity`, `fall`, `gestures` | Posture smoothing, sideways pose, fall sequence and look-alikes, arm gestures (scripted skeletons) | Real falls, real pose error, hand-shape gestures (not implemented) | BlazePose model | Asks before alerting; not a medical device |
| 16 | Follow interface | `follow` | Identity lock, stop conditions, mock only | Real navigation | none | Disabled on hardware by default |
| 17-18 | Context manager, settings, spoken commands | `context`, `vision_settings`, `vision_commands` | Stage scheduling, frame budget, profiles, consent switches | Pi load behaviour | none | Camera off clears state; forget-me needs a yes |
| 19-20 | Privacy audit, factory, benchmark | `privacy`, `vision_factory`, `benchmark` | Static scans, data inventory, sandbox benchmark | Any Pi benchmark | none | No encryption at rest |
| 21 | ROS 2 node and interfaces | `vision_node`, new `.msg`/`.srv` | Core logic and strict-stub node tests; fall fix for slow pose rate | Real ROS 2; topic names are assumptions | ROS 2 | No motion interface |
| 22 | Scenarios A to H | `test_vision_scenarios.py` | All eight run end to end with scripted perception; camera watchdog; settings survive restart | Real people and cameras | none | Fail closed on unreadable settings |
| 23 | 30-area test matrix | `docs/evaluation/test_matrix.*`, `eval_report` | Every area mapped to real tests and honest limits | All real-camera and Pi evidence | none | No real-camera results exist |
| 24 | Documentation and offline demo | `docs/sobiya_advanced_vision.md`, `demo` | Doc checked against data by tests | Pi install steps | `deployment/vision_requirements.txt` | Lists open decisions |
