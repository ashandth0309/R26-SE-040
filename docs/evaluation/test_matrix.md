# Vision evaluation report

**Real-camera results: none.** Every automated result below is a mock, a synthetic scripted case, a source scan, or a measurement on a public dataset. None of it was produced with the Raspberry Pi Camera Module 2, so none of it tells you how well Buddy sees in your home.

Evidence levels: mock = scripted fakes; synthetic_logic = scripted skeletons/scores; real_model_smoke = a real model on a handful of public images (not an accuracy figure); dataset_measured = measured on a named public dataset (not the Pi camera); static_check = source-code scan; stub_only = ROS stubs; tool_only = evaluation harness exists, no data; hardware_required = cannot be measured without the robot. No area has a real-camera result.

## The 30 test areas

| # | Area | Evidence | Tests | Not verified |
|---|---|---|---|---|
| 1 | Camera acquisition | `mock` | 5 | Real Pi Camera Module 2 capture, format, exposure and frame rate. |
| 2 | Shared frame stream | `mock` | 4 | Behaviour under real Pi CPU load with all stages running. |
| 3 | Camera disconnection | `mock` | 7 | Unplugging the real ribbon cable or USB camera; a fall check in progress when the camera dies is dropped, not escalated. |
| 4 | Multiple people | `mock` | 5 | Crowded scenes, occlusion by other people, identity swaps on a real camera. |
| 5 | Unknown person | `mock` | 5 | Real false-accept rate of strangers against an enrolled gallery (no multi-person dataset in the sandbox). |
| 6 | Face recognition confidence | `real_model_smoke` | 7 | Thresholds are uncalibrated; no FAR/FRR/EER measured on real data. |
| 7 | Consent-based enrollment | `mock` | 7 | Spoken consent through the real voice engine and microphone. |
| 8 | Recognition with and without glasses | `tool_only` | 1 | No glasses/no-glasses data exists in the sandbox, so nothing is measured or claimed. |
| 9 | Lighting changes | `mock` | 5 | Real room lighting, backlight, night, IR; accuracy per lighting condition. |
| 10 | Head-angle changes | `mock` | 3 | Recognition and expression accuracy at real yaw/pitch angles. |
| 11 | Facial-expression smoothing | `mock` | 6 | Behaviour on a real face video; the time constants were chosen, not tuned on data. |
| 12 | Emotion prediction uncertainty | `dataset_measured` | 5 | Not representative of the Pi camera, home lighting or this household. Skin tone, age, glasses and occlusion are not labelled in FER2013, so none of those slices were evaluated. |
| 13 | Audio/visual disagreement | `mock` | 6 | Real voice features from a real microphone together with a real face. |
| 14 | Repeated-event suppression | `mock` | 6 | Long real-world sessions. |
| 15 | Proactive interaction cooldown | `mock` | 6 | Whether the chosen cooldowns feel right to real users. |
| 16 | User refusal | `mock` | 5 | Speech recognition of refusals on real audio. |
| 17 | Object recognition | `real_model_smoke` | 8 | Detection and taught-object precision/recall in a real home. Match thresholds are uncalibrated. |
| 18 | New object enrollment | `mock` | 7 | Teaching with a real hand-held object and the real MobileNetV2 features on the Pi. |
| 19 | Object memory retrieval | `mock` | 6 | Accuracy of place recognition in a real home. |
| 20 | Fall detection | `synthetic_logic` | 7 | Sensitivity to real falls, thresholds, pose-model accuracy on lying people. Not a medical or safety device. |
| 21 | False fall detections | `synthetic_logic` | 8 | Pets, laundry, yoga, napping on a sofa, shadows, real pose errors. |
| 22 | Gesture recognition | `synthetic_logic` | 6 | Real waving, pointing, clothing, distance, pose-model error. |
| 23 | Person-follow request cancellation | `mock` | 5 | The real navigation stack. Follow and approach are not wired to ROS. |
| 24 | Unsafe approach refusal | `mock` | 8 | Safe approach itself. Buddy cannot be said to approach people safely until navigation and distance safety are validated. |
| 25 | Loss of target tracking | `mock` | 6 | Real occlusion and re-identification. |
| 26 | ROS 2 topic compatibility | `stub_only` | 6 | Everything under real ROS 2 (QoS, build, colcon, discovery). Topic names other than /emotion/observation are assumptions. |
| 27 | Camera coexistence with voice integration | `stub_only` | 5 | Camera node vs Yohan's camera_node (same topic and node name) was never run together; voice and vision running together on the Pi. |
| 28 | Offline operation | `static_check` | 4 | A full run with the network physically disconnected on the Pi. |
| 29 | Raspberry Pi CPU and memory limits | `hardware_required` | 5 | All Pi CPU, memory, temperature and FPS figures. |
| 30 | End-to-end conversational latency | `hardware_required` | 1 | Camera to spoken reply latency, which also depends on the voice engine and speaker. |

Per-area detail (tests, what was measured, manual procedure) is in `docs/evaluation/test_matrix.json`.

## Expression recognition on FER2013 (cross-dataset)

FER2013 test set, 48x48 grayscale faces. It is a different camera, different people and different conditions from the robot. A facial expression is a visible pattern; it does not reveal how a person feels, and these numbers must not be read as the ability to detect emotions.

### Zoo MobileFaceNet (the model Buddy uses)

Images scored: 6636. Accuracy 55.5%, macro-F1 0.445. Faces not found: 314.

Seven classes:

| true \ predicted | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| **angry** | 196 | 121 | 23 | 60 | 173 | 68 | 65 |
| **disgust** | 15 | 54 | 1 | 13 | 4 | 16 | 3 |
| **fear** | 49 | 100 | 64 | 128 | 255 | 163 | 194 |
| **happy** | 9 | 9 | 9 | 1603 | 50 | 12 | 28 |
| **neutral** | 8 | 57 | 1 | 250 | 754 | 79 | 45 |
| **sad** | 38 | 73 | 9 | 114 | 494 | 368 | 45 |
| **surprise** | 4 | 9 | 27 | 88 | 32 | 12 | 644 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| angry | 0.614 | 0.278 | 0.382 | 706 |
| disgust | 0.128 | 0.509 | 0.204 | 106 |
| fear | 0.478 | 0.067 | 0.118 | 953 |
| happy | 0.711 | 0.932 | 0.806 | 1720 |
| neutral | 0.428 | 0.631 | 0.510 | 1194 |
| sad | 0.513 | 0.323 | 0.396 | 1141 |
| surprise | 0.629 | 0.789 | 0.700 | 816 |

Coarse groups (what Buddy actually uses):

| true \ predicted | positive | neutral | negative | surprise |
|---|---|---|---|---|
| **positive** | 1603 | 50 | 39 | 28 |
| **neutral** | 250 | 754 | 145 | 45 |
| **negative** | 315 | 926 | 1358 | 307 |
| **surprise** | 88 | 32 | 52 | 644 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.711 | 0.932 | 0.806 | 1720 |
| neutral | 0.428 | 0.631 | 0.510 | 1194 |
| negative | 0.852 | 0.467 | 0.604 | 2906 |
| surprise | 0.629 | 0.789 | 0.700 | 816 |

Neutral faces called negative: 145 of 1194 (12.1%).

Calibration: expected calibration error 0.321 before and 0.027 after temperature scaling (T=3.85, fitted on half the data, evaluated on the other half).

### Team emotion_model.h5 (not used by Buddy; for comparison)

Images scored: 6950. Accuracy 52.9%, macro-F1 0.485. Faces not found: 0.

Seven classes:

| true \ predicted | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| **angry** | 353 | 0 | 41 | 87 | 141 | 105 | 3 |
| **disgust** | 76 | 0 | 4 | 13 | 5 | 13 | 0 |
| **fear** | 198 | 0 | 213 | 173 | 187 | 219 | 34 |
| **happy** | 58 | 0 | 29 | 1550 | 78 | 51 | 8 |
| **neutral** | 73 | 0 | 33 | 180 | 773 | 171 | 3 |
| **sad** | 159 | 0 | 65 | 224 | 322 | 474 | 3 |
| **surprise** | 38 | 0 | 282 | 113 | 60 | 26 | 312 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| angry | 0.370 | 0.484 | 0.419 | 730 |
| disgust | n/a | 0.000 | n/a | 111 |
| fear | 0.319 | 0.208 | 0.252 | 1024 |
| happy | 0.662 | 0.874 | 0.754 | 1774 |
| neutral | 0.494 | 0.627 | 0.552 | 1233 |
| sad | 0.448 | 0.380 | 0.411 | 1247 |
| surprise | 0.860 | 0.375 | 0.523 | 831 |

Coarse groups (what Buddy actually uses):

| true \ predicted | positive | neutral | negative | surprise |
|---|---|---|---|---|
| **positive** | 1550 | 78 | 138 | 8 |
| **neutral** | 180 | 773 | 277 | 3 |
| **negative** | 497 | 655 | 1920 | 40 |
| **surprise** | 113 | 60 | 346 | 312 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.662 | 0.874 | 0.754 | 1774 |
| neutral | 0.494 | 0.627 | 0.552 | 1233 |
| negative | 0.716 | 0.617 | 0.663 | 3112 |
| surprise | 0.860 | 0.375 | 0.523 | 831 |

Neutral faces called negative: 277 of 1233 (22.5%).

Calibration: expected calibration error 0.067 before and 0.016 after temperature scaling (T=0.75, fitted on half the data, evaluated on the other half).

### Coverage that is missing

FER2013 carries no labels for skin tone, age group, glasses, occlusion, lighting or head angle, so none of those slices could be evaluated. Disgust has only 55 test images and fear is rarely recognised, so those per-class figures are unreliable. Before any claim about those conditions, collect consented, labelled captures from the robot itself and run `python -m buddy_core.vision.expression_eval`.
