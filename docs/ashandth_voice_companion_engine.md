# Voice Companion Engine (Ashandth_Voice)

Component owner: Ashandth Uthayashankar. Branch: `Ashandth_Voice`
(from `origin/buddy-ros2`). Status: **software complete and tested with
fakes and synthetic audio; NOT verified on the Raspberry Pi or under real
ROS 2.** See "Unverified" below before relying on anything here.

The old `buddy_voice.py` node is untouched and remains the validated demo
voice. The companion engine is a separate node, `voice_companion`, and
refuses to start while `buddy_voice` is running.

## 1. What it does

Offline-first spoken companion for BUDDY: listens through one or two USB
microphones, understands a small set of motion commands plus open
conversation, answers with Piper speech, tracks the user's emotional
signals conservatively, keeps consented memory, and can optionally use a
local or cloud language model. It never gives the language model control
of the robot.

## 2. Architecture

```
mics -> MicrophoneManager -> VAD -> EchoGuard -> VoiceEngine
                                         |            |
                                   Recognizer     TurnManager (state machine)
                                         |            |
                                   parse_command   DialogueManager
                                         |        (hooks > rules > ML > LLM > fallback)
                                   CommandRouter          |
                                         |          SpeakerManager -> aplay
                       /movement/request  /navigation/command
                                         |
                                safety_supervisor (never bypassed)
```

Modules (all in `buddy_core/voice/`):

| Area | Modules |
|---|---|
| Audio | `audio_devices`, `dsp`, `vad`, `audio_manager`, `speaker_manager`, `echo` |
| Commands | `voice_commands`, `command_router`, `action_gate` |
| Speech | `stt`, `tts_backends`, `language` |
| Conversation | `turn_manager`, `backchannel`, `variation`, `dialogue_manager`, `voice_engine` |
| Emotion | `emotion_audio`, `emotion_fusion`, `emotional_response` |
| Memory and personality | `memory_store`, `companion_commands`, `personality`, `proactive` |
| Online | `connectivity`, `llm_router` |
| Wiring | `config`, `engine_factory`, `runner`, `voice_node` |
| Deployment | `benchmark`, `scripts/start_buddy_companion.sh`, `deployment/` |

### Turn-taking states

SLEEPING, AWAKE (idle), USER_SPEAKING, THINKING, BUDDY_SPEAKING, plus
backchannel and interruption handling. A spoken **stop is legal in every
state** and is handled before any other work.

### Stop path

`stop` is matched loosely on purpose (a false stop is harmless). On a
hit the engine publishes `STOP` first, then cancels any pending reply and
current speech, then speaks a cached acknowledgement. It never waits on
the language model or the network; replies run on a worker thread and
stale replies are dropped. A stop word in a partial transcript triggers
the stop early.

### Microphones and echo

- Auto-discovery of input devices; up to two microphones.
- Independent microphones are **never summed**. One is selected, with
  hysteresis (`switch_margin_db`, `min_dwell_s`). Summing is allowed only
  for a validated stereo device.
- `EchoGuard` mutes recognition while Buddy speaks and for a short tail,
  learns speaker-to-mic leakage, and allows barge-in only when the input
  clearly exceeds the expected playback leakage. Optional Speex AEC adapter
  exists but is unverified.
- Mic loss while moving stops the robot.

### Safety layering

- The LLM may only *suggest* an action. `ActionGate` asks "Do you want me
  to ...?" and needs a yes within 15 s, then routes via `CommandRouter`.
- Voice motion is capped at 8 s (`router.max_motion_s`), republished at
  10 Hz, and expires on its own.
- Mute also stops the robot. Unmute needs the `/voice/mute` service or a
  button, not a spoken command.
- `/safety_state` emergency stop or movement-not-allowed inhibits motion.
- The router never talks to motors. It publishes to `/movement/request`,
  where `safety_supervisor` still decides.
- Turn commands map to ROTATE_LEFT / ROTATE_RIGHT.

### Emotion

- Text: the user's own words and corrections always override predictions.
  A denial ("I'm not sad") suppresses that label for 10 minutes.
- Voice: simple acoustic features (pitch, energy, MFCC) against a
  per-speaker baseline (needs at least 5 samples). Confidence is capped at
  0.6 and voice alone yields only a `low_energy` or `high_energy` hint.
  An `unknown` state is always possible.
- Camera: consumed from `EmotionObservation` messages. The voice engine
  never opens the camera (one camera owner).
- **No accuracy is claimed.** `emotion_audio.evaluate()` computes metrics
  only from labelled data you supply.
- Replies are supportive and non-clinical. Crisis statements get a fixed
  safety message that is never LLM-generated and never stored.

### Memory and privacy

- SQLite, off until the user consents by voice. Consent, correction,
  deletion, export (to `export_dir`) and size limits are supported.
- Credentials and payment or ID numbers are refused.
- Delete is secure delete plus `VACUUM`.
- Silence modes: quiet, silent, mute. Refused proactive suggestions are
  remembered and not repeated.
- Cloud use is off by default; the key is read from the environment
  variable named in `online.cloud_api_key_env` (default
  `BUDDY_LLM_API_KEY`). No key is ever stored in the repo.

### Language

English works offline with the English Vosk and Piper models. **Tamil is
not supported offline**: those models cannot recognise or speak Tamil.
Tamil text is detected and Buddy says so honestly. Tamil would need a
Tamil recogniser and voice, or a cloud model, which has not been built or
tested.

### Online and offline

Providers: local Ollama and any OpenAI-compatible endpoint. Both are off
by default. Calls use bounded retries, a circuit breaker, and fall back
to the offline rules. Reply text is cleaned and length-limited before
speaking.

## 3. Configuration

`config/voice_companion.yaml` (all keys optional) and
`config/personality.yaml` (tone, repetition limits). Profiles:

| Profile | Effect |
|---|---|
| lightweight | Audio emotion, backchannels and proactive suggestions off; no LLM |
| balanced (default) | Local features on, no LLM |
| enhanced | Balanced plus the LLM if configured |

CPU and memory cost of each profile is **not measured on the Pi yet**.
Run the benchmark (section 7) to measure.

ROS parameters: `movement_topic`, `voice_command_topic`, `status_topic`,
`safety_topic`, `emotion_topic`, `mute_service`, `navigation_topic`,
`navigation_status_topic`, `dry_run`, `allow_coexist`, `profile`,
`vosk_model_path`, `piper_model_path`.

## 4. ROS interfaces

| Direction | Topic or service | Type |
|---|---|---|
| publish | `/movement/request` | `buddy_interfaces/MovementCommand` |
| publish | `/voice/command` | `buddy_interfaces/VoiceCommand` |
| publish | `/voice/status` | `std_msgs/String` (JSON) |
| subscribe | `/safety_state` | `buddy_interfaces/SafetyState` |
| subscribe | `/emotion/observation` | `buddy_interfaces/EmotionObservation` |
| service | `/voice/mute` | `std_srvs/SetBool` |
| publish (opt.) | `/navigation/command` | `buddy_nav_interfaces/NavigationCommand` |
| subscribe (opt.) | `/navigation/status` | `buddy_nav_interfaces/NavigationStatus` |

The emotion topic name is an assumption (vision topics are not defined in
the repo docs); confirm it with the vision owner. Navigation is disabled
until the navigation package exists on the branch; its command codes
mirror `Yohan_Mapping` and are unverified here.

## 5. Dependencies

Do not commit model binaries; `.gitignore` excludes `*.onnx`,
`vosk-model-*/`, `*.gguf`, `.buddy/`, `buddy_exports/`.

System (Raspberry Pi OS or Ubuntu 24.04 ARM64):

```
sudo apt install portaudio19-dev libasound2-dev alsa-utils
```

Python (see `deployment/voice_requirements.txt`; versions not yet pinned):
`numpy`, `pyyaml`, `sounddevice`, `vosk`, `piper-tts`. Optional: `scipy`,
`scikit-learn` (existing intent model), `webrtcvad`, `speexdsp`.

Optional services: Ollama for a local LLM.

Models (downloaded on the robot, kept out of git):

- Vosk English small model, directory path in `BUDDY_VOSK_MODEL`.
- Piper English voice `.onnx` plus its `.json`, path in `BUDDY_PIPER_MODEL`.

Not installed in the development sandbox: ROS 2, Vosk, Piper, webrtcvad.
Everything touching those uses injectable backends that tests fake.

## 6. Running

Development, no hardware or models:

```
ros2 run buddy_core voice_companion --ros-args -p dry_run:=true
```

On the robot (UNVERIFIED):

```
export BUDDY_VOSK_MODEL=/home/ashandth/models/vosk-model-small-en-us-0.15
export BUDDY_PIPER_MODEL=/home/ashandth/models/piper/en_US-lessac-medium.onnx
scripts/start_buddy_companion.sh
```

`deployment/buddy_companion.service` is provided but not enabled. It
conflicts with `buddy.service` on purpose.

## 7. Performance measurement

```
python3 -m buddy_core.voice.benchmark --out benchmark_results/pi.json
python3 -m buddy_core.voice.benchmark --vosk-model ... --piper-model ...
```

The report records the machine. Only a run with
`machine.is_raspberry_pi: true` describes the robot. The targets in the
file are starting goals, not results. **No Pi numbers exist yet.**

## 8. Tests

Run: `cd buddy_ws/src/buddy_core && python -m pytest test -q`

| File | Covers |
|---|---|
| `test_voice_audio.py` | device discovery, DSP, VAD, mic selection and drift, speaker priority and failover, echo guard, config |
| `test_voice_commands.py` | command parsing, stop/wake/sleep, router limits and expiry, STT and TTS fakes, message-code match |
| `test_voice_engine.py` | state machine, stop priority, barge-in, backchannels, reply cancellation, mute, mic loss |
| `test_voice_emotion.py` | features, baseline, fusion, corrections, non-clinical replies, crisis reply |
| `test_voice_memory.py` | consent, correction, deletion, export, limits, sensitive refusal, silence modes, personality, proactive refusal memory |
| `test_voice_online.py` | language policy, circuit breaker, retries, fallback, action gate, key handling |
| `test_voice_integration.py` | factory, runner, simulated end-to-end flows |
| `test_voice_node.py` | node with strict ROS stubs built from the repo `.msg` files |
| `test_voice_deploy.py` | benchmark harness, start script syntax, no secrets, ignore rules |

All audio in tests is synthetic. Last full run in the sandbox: 483 passed,
1 failed. The failure, `test_logging_utils.py::test_console_handler_created`,
also fails on the previous commit without these changes, so it predates
this work. Flake8 is clean on all voice files.

## 9. Unverified (must be checked on the robot)

- Anything under real ROS 2 (tests use stubs generated from the .msg files).
- Real Vosk recognition accuracy, including the loose stop grammar.
- Real Piper synthesis and `aplay` output on the Pi, and audio latency.
- Two-microphone behaviour, drift, switching and echo leakage on real hardware.
- Barge-in and echo thresholds with the real speaker and room.
- Speex AEC adapter and webrtcvad path.
- CPU, memory and thermal cost of each profile; stop latency end to end.
- Local and cloud LLM behaviour.
- Emotion accuracy: none measured, none claimed.
- Vision topic name and navigation command codes.
- Startup script and systemd unit.

**Pi deployment (Phase 9) is not complete until it has been physically
tested.** Suggested first checks: run the benchmark; say "stop" while the
robot is moving with wheels off the ground and time it; unplug a mic while
moving; speak over Buddy; confirm `safety_supervisor` still blocks motion
on an obstacle.

## 10. Safety and privacy limits

- Voice is not a safety system. The physical emergency stop and
  `safety_supervisor` remain authoritative.
- Speech recognition can mishear. A false stop is tolerated; a false
  motion command is limited by whole-utterance matching, the 8 s cap and
  the safety supervisor.
- Voice emotion is a weak hint, not a diagnosis. Buddy gives no clinical
  advice.
- Memory stores only what the user consented to and can be exported or
  erased by voice command.
- Microphone audio is never written to disk by the engine. Only Buddy's own synthesised speech is written to a temporary WAV for playback.
