"""The central BUDDY coordinator (pure Python, no ROS).

It owns no device and drives no motor.  It sits between the three
components and the safety path:

* motion intake - voice movement and navigation requests arrive on the
  coordinator's intake topics, are checked against robot state, safety and
  navigation status, and only then forwarded.  STOP is never delayed and
  also has a direct route that does not need the coordinator at all.
* utterance routing - requests that need the camera ("what is this?",
  "this is my red pen", "where did you last see my book?") are turned into
  typed vision requests.  A vision question is *always* answered from the
  vision result or an honest "I can't see right now"; it is never left to
  the conversational model, which could invent an object.
* vision events - fall check-ins, greetings and proactive offers are
  spoken through the one speech arbiter, subject to quiet mode, emergency
  state and cooldowns.
* ownership and readiness - hardware-ownership conflicts and "is it really
  safe to move" are computed from received messages.

Safety rules this file keeps (each has a test):

* Nothing here talks to a motor.  The only things it can send are
  navigation requests, STOP, and relayed voice movement.
* Unknown or stale means no: missing, stale or contradictory inputs refuse
  motion.
* Emergency stop outranks every other state and every speech class except
  emergency speech.
* A camera question never falls through to a model when vision fails.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from buddy_orchestrator import hardware
from buddy_orchestrator.contracts import (
    INTERFACE_VERSION,
    STALE_AFTER_S,
    Err,
)
from buddy_orchestrator.models import (
    MOTION_NAV_KINDS,
    MOVE_STOP,
    SAFE_NAV_KINDS,
    MoveIn,
    NavRequest,
    NavStatusIn,
    ObstacleIn,
    PersonIn,
    RouteAnswer,
    SafetyIn,
    VisionEventIn,
)
from buddy_orchestrator.ports import Ports
from buddy_orchestrator.profiles import Profile
from buddy_orchestrator.readiness import Capabilities, evaluate
from buddy_orchestrator.speech import Kind, SpeechArbiter
from buddy_orchestrator.states import (
    MOTION_READY,
    MOVING,
    ConversationState,
    RobotState,
    conversation_machine,
    robot_machine,
)
from buddy_orchestrator.vision_intents import Intent, parse_vision_intent

SOURCE = "buddy_coordinator"

# Spoken refusals (plain, no blame, no pretending).
SAY = {
    "estop": "Emergency stop is active, so I can't move.",
    "unsafe": "I can't check my sensors right now, so I'm not moving.",
    "disabled": "Movement is switched off at the moment.",
    "no_pose": "I don't know where I am well enough to do that.",
    "busy": "I'm already doing something. Say stop first.",
    "not_ready": "I'm not ready to move yet.",
    "lost_nav": "I lost contact with navigation, so I stopped.",
    "no_vision": "I can't see right now.",
    "camera_off": "My camera is off, so I can't look at that.",
    "vision_timeout": "I couldn't look just now. Could you try again?",
    "intro": ("Would they like me to remember their face? They need to "
              "tell me themselves. I only save a face when that person "
              "agrees."),
    "ask_name": "What would you like to call it?",
    "approach_no": "I can see you, but I can't approach safely yet.",
}

APPROACH_REFUSALS = {
    "person_not_recognised":
        "I'm not sure who's asking, so I'll stay here.",
    "quiet_mode": "Okay, I'll keep my distance.",
    "person_disabled_approach": "Okay, I'll keep my distance.",
    "too_soon": "I only just did that. Give me a moment.",
    "safety_not_allowing_movement": SAY["estop"],
    "already_approaching": "I'm already on my way.",
    "emergency_in_progress": "I need to deal with something else first.",
    "target_stale": "I lost sight of you for a moment. Could you say it "
                    "again?",
    "bearing_unreliable": "I can't tell where you are well enough.",
}

# Vision events that may be spoken, and how.
EVENT_SPEECH = {
    "greeting_due": (Kind.PROACTIVE, "Hey! Nice to see you again."),
    "proactive_offer": (Kind.PROACTIVE,
                        "Hey, you seem a little quiet. Would you like "
                        "some company?"),
    "fall_suspected": (Kind.STATUS, "Are you okay? Do you need help?"),
    "fall_alert": (Kind.STATUS, "Are you okay? Do you need help?"),
    "lost_after_descent": (Kind.STATUS,
                           "I lost sight of you. Are you okay?"),
    "teach_prompt": (Kind.REPLY, ""),
    "camera_lost": (Kind.STATUS,
                    "My camera stopped working, so I can't see right "
                    "now."),
}
FALL_KINDS = frozenset({"fall_suspected", "fall_alert",
                        "lost_after_descent"})


@dataclass
class CoordinatorConfig:
    #: Navigation kinds whose result is announced when they finish.
    announce_nav_kinds: tuple = ("start_patrol", "go_to_location")
    raw_motion_grace_s: float = 0.6
    init_timeout_s: float = 10.0
    #: True only after a person-distance source has been validated on
    #: hardware.  Nothing in this repository validates one.
    person_distance_validated: bool = False
    image_width_px: float = 640.0
    horizontal_fov_deg: float = 62.2          # Pi Camera Module 2
    sim_approach_duration_s: float = 3.0
    teach_name_timeout_s: float = 20.0
    people_max_age_s: float = STALE_AFTER_S["vision_people"]


@dataclass
class PendingNav:
    request_id: str
    kind: str
    issued: float
    announce: bool


@dataclass
class Counters:
    nav_forwarded: int = 0
    nav_refused: int = 0
    movement_relayed: int = 0
    movement_refused: int = 0
    utterances_claimed: int = 0
    utterances_passed: int = 0
    stops: int = 0
    vision_failures: int = 0


class BuddyCoordinator:
    """Coordinates voice, vision and navigation.  See the module docs."""

    def __init__(self, profile: Profile, ports: Ports, *,
                 config: CoordinatorConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.profile = profile
        self.ports = ports
        self.cfg = config or CoordinatorConfig()
        self._clock = clock
        self.robot = robot_machine(clock)
        self.conversation = conversation_machine(clock)
        self.speech = SpeechArbiter(ports.speak, clock=clock)
        self.capabilities = Capabilities()
        self.counters = Counters()

        # latest inputs and when they arrived (monotonic)
        self.safety: SafetyIn | None = None
        self.obstacle: ObstacleIn | None = None
        self.nav: NavStatusIn | None = None
        self.voice: dict = {}
        self.vision_status: dict = {}
        self.people: list[PersonIn] = []
        self._seen: dict[str, float] = {}

        self.conflicts: list[hardware.Conflict] = []
        self.pending: dict[str, PendingNav] = {}
        self._raw_motion_until = -1e12
        self._started = clock()
        self._awaiting_teach_name_until = -1e12
        self._fall_prompt_at = -1e12
        self._announced_estop = False
        self._lost_nav_announced = False
        self.approach = self._make_approach()
        self._approach_started = -1e12
        self.log: list[tuple[float, str]] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _note(self, what: str) -> None:
        self.log.append((self._clock(), what))
        del self.log[:-200]

    def _fresh(self, key: str, name: str) -> bool:
        seen = self._seen.get(key)

        return seen is not None and \
            self._clock() - seen <= STALE_AFTER_S[name]

    @property
    def safety_fresh(self) -> bool:
        return self._fresh("safety", "safety_state")

    @property
    def obstacle_fresh(self) -> bool:
        return self._fresh("obstacle", "obstacle_status")

    @property
    def nav_fresh(self) -> bool:
        return self._fresh("nav", "navigation_status")

    @property
    def estop(self) -> bool:
        return bool(self.safety_fresh and self.safety
                    and self.safety.emergency_stop_active)

    @property
    def motion_inputs_ok(self) -> bool:
        """Everything motion safety depends on is arriving and healthy."""

        return bool(
            self.safety_fresh and self.obstacle_fresh and self.nav_fresh
            and self.safety and self.obstacle and self.nav
            and not self.safety.emergency_stop_active)

    def _say(self, kind: Kind, text: str) -> bool:
        self._sync_speech_context()

        return self.speech.say(kind, text, SOURCE)

    def _sync_speech_context(self) -> None:
        c = self.speech.context
        c.shutting_down = self.robot.state is RobotState.SHUTTING_DOWN
        c.emergency_stop = self.robot.state is RobotState.EMERGENCY_STOP
        c.quiet_mode = str(self.voice.get("silence_mode", "normal")) \
            != "normal"
        state = str(self.voice.get("state", ""))
        c.user_speaking = state == "USER_SPEAKING"
        c.buddy_speaking = state == "BUDDY_SPEAKING"
        c.conversation_active = state in ("PROCESSING", "USER_SPEAKING") \
            or self.conversation.state in (
                ConversationState.VISION_REQUEST,
                ConversationState.PROCESSING)

    def _conv_to(self, target: ConversationState) -> None:
        if not self.conversation.go(target, "coordinator"):
            self.conversation.go(ConversationState.LISTENING, "recover")
            self.conversation.go(target, "coordinator")

    # ------------------------------------------------------------------
    # stopping
    # ------------------------------------------------------------------

    def stop_all(self, reason: str) -> None:
        """Stop everything that moves.  Never raises, never waits."""

        self.counters.stops += 1
        self._raw_motion_until = -1e12
        self.pending.clear()
        self._note(f"stop_all: {reason}")

        for send in (lambda: self.ports.movement_stop(reason),
                     lambda: self.ports.nav_send(NavRequest(
                         "stop", source=SOURCE,
                         request_id=f"coord-stop-{self.counters.stops}"))):
            try:
                send()
            except Exception as exc:        # a dead port must not stop
                self._note(f"stop path error: {exc}")  # the other one

        if self.approach.active is not None:
            self.approach.cancel(self._clock(), "stop")

    # ------------------------------------------------------------------
    # state derivation
    # ------------------------------------------------------------------

    def _sync_robot_state(self) -> None:
        now = self._clock()
        r = self.robot

        if r.state is RobotState.SHUTTING_DOWN:
            return

        if r.state is RobotState.STARTING:
            r.go(RobotState.INITIALIZING, "tick")

        if self.estop:
            if r.state is not RobotState.EMERGENCY_STOP:
                r.go(RobotState.EMERGENCY_STOP, "supervisor latch")
                self.stop_all("emergency stop active")

                if not self._announced_estop:
                    self._announced_estop = True
                    self._say(Kind.EMERGENCY, SAY["estop"])

            return

        self._announced_estop = False

        if r.state is RobotState.EMERGENCY_STOP:
            if self.safety_fresh:
                r.go(RobotState.IDLE if self.motion_inputs_ok
                     else RobotState.DEGRADED_MODE, "latch cleared")
            return

        if r.state is RobotState.INITIALIZING:
            if self.motion_inputs_ok:
                r.go(RobotState.IDLE, "inputs arriving")
            elif now - self._started > self.cfg.init_timeout_s:
                r.go(RobotState.DEGRADED_MODE, "inputs missing")

            return

        if not self.motion_inputs_ok:
            was_moving = r.state in MOVING or r.state is RobotState.PAUSED

            if r.state is not RobotState.DEGRADED_MODE:
                r.go(RobotState.DEGRADED_MODE, self._why_degraded())

                if was_moving:
                    self.stop_all("motion inputs lost")

                    if not self._lost_nav_announced:
                        self._lost_nav_announced = True
                        self._say(Kind.STATUS, SAY["lost_nav"])

            return

        self._lost_nav_announced = False

        if r.state is RobotState.DEGRADED_MODE:
            r.go(RobotState.IDLE, "inputs recovered")

        self._follow_navigation_state()

    def _why_degraded(self) -> str:
        missing = [name for name, ok in (
            ("safety", self.safety_fresh),
            ("obstacle", self.obstacle_fresh),
            ("navigation", self.nav_fresh)) if not ok]

        return "stale: " + ", ".join(missing) if missing else "unhealthy"

    def _follow_navigation_state(self) -> None:
        nav, r = self.nav, self.robot

        if nav is None:
            return

        if nav.state == "running":
            target = RobotState.PATROL_ACTIVE \
                if nav.active_command == "start_patrol" \
                else RobotState.NAVIGATION_ACTIVE
            r.go(target, "navigation running")
        elif nav.state == "paused":
            r.go(RobotState.PAUSED, "navigation paused")
        elif r.state in MOVING or r.state is RobotState.PAUSED:
            if self._clock() >= self._raw_motion_until:
                r.go(RobotState.IDLE, f"navigation {nav.state}")

    # ------------------------------------------------------------------
    # inputs: safety, sensing, navigation
    # ------------------------------------------------------------------

    def on_safety(self, s: SafetyIn) -> None:
        self.safety = s
        self._seen["safety"] = self._clock()
        self.capabilities.safety_supervisor = True
        self._sync_robot_state()

    def on_obstacle(self, o: ObstacleIn) -> None:
        self.obstacle = o
        self._seen["obstacle"] = self._clock()
        self.capabilities.obstacle_sensors = not o.degraded
        self._sync_robot_state()

    def on_nav_status(self, s: NavStatusIn) -> None:
        self.nav = s
        self._seen["nav"] = self._clock()
        self.capabilities.navigation = True
        self._sync_robot_state()
        pending = self.pending.get(s.request_id)

        if pending is None:
            return

        if s.state in ("succeeded", "failed"):
            del self.pending[s.request_id]

            if pending.announce:
                self._announce_nav_result(pending, s)
        elif s.state in ("cancelled", "idle"):
            del self.pending[s.request_id]

    def _announce_nav_result(self, p: PendingNav, s: NavStatusIn) -> None:
        if s.state == "succeeded":
            text = "Patrol finished." if p.kind == "start_patrol" \
                else "I'm here."
        else:
            reason = (s.reason or "something went wrong").rstrip(".")
            text = f"I couldn't finish that: {reason}."

        self._say(Kind.STATUS, text)

    # ------------------------------------------------------------------
    # motion intake
    # ------------------------------------------------------------------

    def _motion_refusal(self, req: NavRequest) -> tuple[str, Err] | None:
        """Why a motion request must not go ahead, or None."""

        s, nav = self.robot.state, self.nav

        if self.estop or s is RobotState.EMERGENCY_STOP:
            return "estop", Err.REFUSED_SAFETY

        if s in MOVING or (s is RobotState.PAUSED
                           and req.kind != "resume") \
                or self._clock() < self._raw_motion_until:
            return "busy", Err.BUSY

        if s is RobotState.PAUSED and req.kind == "resume":
            pass
        elif s is not RobotState.IDLE:
            return "not_ready", Err.REFUSED_SAFETY

        if not self.motion_inputs_ok:
            return "unsafe", Err.STALE

        assert nav is not None and self.safety is not None

        if not self.safety.movement_allowed:
            return "unsafe", Err.REFUSED_SAFETY

        if not nav.motion_enabled:
            return "disabled", Err.REFUSED_SAFETY

        if not nav.sensors_healthy:
            return "unsafe", Err.REFUSED_SAFETY

        if req.kind in ("start_patrol", "go_to_location") \
                and not nav.pose_trusted:
            return "no_pose", Err.REFUSED_SAFETY

        return None

    def on_voice_navigation(self, req: NavRequest) -> bool:
        """A NavigationCommand from the voice node.  True if forwarded."""

        if req.is_stop:
            self.stop_all("voice stop")

            return True

        if req.kind == "resume" and self.estop:
            self.counters.nav_refused += 1
            self._say(Kind.STATUS, SAY["estop"])

            return False

        if req.kind in SAFE_NAV_KINDS:
            return self._forward_nav(req)

        refusal = self._motion_refusal(req) \
            if req.kind in MOTION_NAV_KINDS or req.kind == "resume" \
            else ("not_ready", Err.NOT_UNDERSTOOD)

        if refusal:
            self.counters.nav_refused += 1
            self._note(f"refused {req.kind}: {refusal[0]}")
            self._say(Kind.STATUS, SAY[refusal[0]])

            return False

        return self._forward_nav(req)

    def _forward_nav(self, req: NavRequest) -> bool:
        try:
            self.ports.nav_send(req)
        except Exception as exc:
            self._note(f"nav send failed: {exc}")
            self.counters.nav_refused += 1

            return False

        self.counters.nav_forwarded += 1

        if req.kind in MOTION_NAV_KINDS and req.request_id:
            self.pending[req.request_id] = PendingNav(
                req.request_id, req.kind, self._clock(),
                req.kind in self.cfg.announce_nav_kinds)

        return True

    def on_voice_movement(self, m: MoveIn) -> bool:
        """A raw MovementCommand from the voice node.  True if relayed."""

        if m.command == MOVE_STOP:
            self.stop_all("voice stop")

            return True

        refused = None

        if self.estop or self.robot.state is RobotState.EMERGENCY_STOP:
            refused = "estop"
        elif self.robot.state not in MOTION_READY \
                or self.robot.state is RobotState.PAUSED:
            refused = "not_ready"
        elif not self.motion_inputs_ok or not (
                self.safety and self.safety.movement_allowed):
            refused = "unsafe"
        elif self.nav and self.nav.state in ("running", "paused"):
            refused = "busy"

        if refused:
            self.counters.movement_refused += 1
            self._note(f"refused raw movement: {refused}")
            self.ports.movement_stop(f"refused: {refused}")

            return False

        try:
            self.ports.movement_relay(m)
        except Exception as exc:
            self._note(f"movement relay failed: {exc}")
            self.counters.movement_refused += 1

            return False

        self.counters.movement_relayed += 1
        self._raw_motion_until = self._clock() + m.valid_for_s \
            + self.cfg.raw_motion_grace_s

        return True

    # ------------------------------------------------------------------
    # inputs: voice and vision
    # ------------------------------------------------------------------

    def on_voice_status(self, status: dict) -> None:
        self.voice = status
        self._seen["voice"] = self._clock()
        c = self.capabilities
        c.microphone = bool(status.get("mic_active")) \
            and not status.get("muted")
        c.speaker = not status.get("speaker_failed", False)
        c.internet = str(status.get("cloud", "")) in ("online", "connected")
        voice_state = str(status.get("state", ""))

        if voice_state == "USER_SPEAKING":
            self._conv_to(ConversationState.USER_SPEAKING)
        elif voice_state == "BUDDY_SPEAKING":
            self._conv_to(ConversationState.BUDDY_SPEAKING)
        elif voice_state == "PROCESSING":
            if self.conversation.state is not ConversationState.VISION_REQUEST:
                self._conv_to(ConversationState.PROCESSING)
        elif self.conversation.state is not ConversationState.VISION_REQUEST:
            self._conv_to(ConversationState.LISTENING)

    def on_vision_status(self, status: dict) -> None:
        self.vision_status = status
        self._seen["vision"] = self._clock()
        ok = bool(status.get("camera_ok")) \
            and bool(status.get("camera_enabled", True))
        self.capabilities.camera = ok
        self.capabilities.vision = True

    def on_people(self, people: list[PersonIn]) -> None:
        self.people = list(people)
        self._seen["people"] = self._clock()

    def _people_now(self) -> list[PersonIn] | None:
        """People visible now, or None if the data is too old."""

        seen = self._seen.get("people")

        if seen is None or self._clock() - seen > self.cfg.people_max_age_s:
            return None

        return self.people

    def on_vision_event(self, e: VisionEventIn) -> None:
        if e.kind == "camera_lost":
            self.capabilities.camera = False
        elif e.kind == "camera_restored":
            self.capabilities.camera = True

        entry = EVENT_SPEECH.get(e.kind)

        if entry is None:
            return

        kind, default = entry
        text = e.suggested_text or default

        if e.kind in FALL_KINDS:
            # Ask first. A suspected fall is never an automatic alert and
            # never starts any movement toward the person.
            if not (e.needs_voice_check or e.kind != "fall_alert"):
                return

            self._fall_prompt_at = self._clock()

        if text:
            self._say(kind, text)

    def graph_changed(self, nodes: list[str],
                      publisher_counts: dict[str, int]) -> list:
        """Audit hardware ownership against the live graph."""

        found = hardware.audit(nodes, publisher_counts)

        if [c.message for c in found] != [c.message for c in self.conflicts]:
            for c in found:
                self._note(f"ownership: {c.message}")

        self.conflicts = found

        return found

    # ------------------------------------------------------------------
    # utterance routing
    # ------------------------------------------------------------------

    def route_utterance(self, text: str, confidence: float = 0.0,
                        request_id: str = "") -> RouteAnswer:
        """Handle one finished utterance that is not a voice command.

        ``claimed=False`` means "ordinary conversation, let the voice
        engine answer".  Vision requests are always claimed, even when
        vision is unavailable, so a model never invents what Buddy sees.
        """

        try:
            return self._route(text, confidence)
        except Exception as exc:
            self._note(f"route error: {exc!r}")
            self.counters.vision_failures += 1

            return RouteAnswer(False, error=Err.UNAVAILABLE)
        finally:
            if self.conversation.state is ConversationState.VISION_REQUEST:
                self._conv_to(ConversationState.PROCESSING)

    def _claim(self, reply: str, intent: str,
               error: Err = Err.OK) -> RouteAnswer:
        self.counters.utterances_claimed += 1

        return RouteAnswer(True, reply, intent, error)

    def _pass(self) -> RouteAnswer:
        self.counters.utterances_passed += 1

        return RouteAnswer(False)

    def _route(self, text: str, confidence: float) -> RouteAnswer:
        now = self._clock()
        vision = self.ports.vision

        if now < self._awaiting_teach_name_until and vision is not None:
            return self._teach_with_name(text)

        intent = parse_vision_intent(text)

        if intent is None:
            return self._offer_to_vision(text)

        self._conv_to(ConversationState.VISION_REQUEST)

        if intent.kind is Intent.PERSON_INTRO:
            return self._claim(SAY["intro"], intent.kind.value)

        if intent.kind is Intent.COME_HERE:
            return self._come_here()

        if intent.kind is Intent.WHO_IS_THIS:
            return self._who_is_this()

        if intent.kind is Intent.CHECK_FLOOR:
            return self._check_floor()

        needs_camera = intent.kind in (Intent.IDENTIFY_OBJECT,
                                       Intent.TEACH_OBJECT)

        if vision is None or not self.capabilities.vision:
            return self._claim(SAY["no_vision"], intent.kind.value,
                               Err.UNAVAILABLE)

        if needs_camera and not self.capabilities.camera:
            return self._claim(SAY["camera_off"], intent.kind.value,
                               Err.UNAVAILABLE)

        if intent.kind is Intent.IDENTIFY_OBJECT:
            return self._identify(vision)

        if intent.kind is Intent.TEACH_OBJECT:
            return self._teach_start(vision, intent.name)

        return self._where_is(vision, intent.name)

    def _vision_call(self, fn, *args):
        try:
            return fn(*args), None
        except TimeoutError:
            self.counters.vision_failures += 1

            return None, Err.TIMEOUT
        except Exception as exc:
            self._note(f"vision call failed: {exc!r}")
            self.counters.vision_failures += 1

            return None, Err.UNAVAILABLE

    def _identify(self, vision) -> RouteAnswer:
        result, err = self._vision_call(vision.identify)

        if err is not None:
            reply = SAY["vision_timeout"] if err is Err.TIMEOUT \
                else SAY["no_vision"]

            return self._claim(reply, Intent.IDENTIFY_OBJECT.value, err)

        text = str(result.get("text", "")).strip()

        if not result.get("success") or not text:
            return self._claim(SAY["no_vision"],
                               Intent.IDENTIFY_OBJECT.value,
                               Err.UNAVAILABLE)

        return self._claim(text, Intent.IDENTIFY_OBJECT.value)

    def _teach_start(self, vision, name: str) -> RouteAnswer:
        if not name:
            self._awaiting_teach_name_until = \
                self._clock() + self.cfg.teach_name_timeout_s

            return self._claim(SAY["ask_name"], Intent.TEACH_OBJECT.value)

        return self._teach_call(vision, "start", name)

    def _teach_with_name(self, text: str) -> RouteAnswer:
        self._awaiting_teach_name_until = -1e12
        lowered = text.lower().strip(" .!?")

        if lowered in ("no", "nope", "cancel", "never mind", "stop"):
            return self._claim("Okay, never mind.",
                               Intent.TEACH_OBJECT.value)

        return self._teach_call(self.ports.vision, "start", lowered)

    def _teach_call(self, vision, action: str, name: str) -> RouteAnswer:
        result, err = self._vision_call(vision.teach, action, name)

        if err is not None or result is None:
            return self._claim(SAY["no_vision"], Intent.TEACH_OBJECT.value,
                               err or Err.UNAVAILABLE)

        reply = str(result.get("reply", "")).strip() or SAY["no_vision"]
        error = Err.OK if result.get("ok") else Err.REFUSED_CONSENT

        return self._claim(reply, Intent.TEACH_OBJECT.value, error)

    def _where_is(self, vision, name: str) -> RouteAnswer:
        result, err = self._vision_call(vision.where_is, name)

        if err is not None or result is None:
            return self._claim(SAY["no_vision"], Intent.WHERE_IS.value,
                               err or Err.UNAVAILABLE)

        return self._claim(str(result.get("text", "")).strip()
                           or "I don't know where that is.",
                           Intent.WHERE_IS.value)

    def _offer_to_vision(self, text: str) -> RouteAnswer:
        """Settings, privacy, teaching answers and replies to offers."""

        vision = self.ports.vision

        if vision is None or not self.capabilities.vision:
            return self._pass()

        result, err = self._vision_call(vision.command, text, "")

        if err is not None or result is None or not result.handled:
            return self._pass()

        if result.pass_to_dialogue or not result.reply:
            return self._pass()

        return self._claim(result.reply, "vision_command")

    def _who_is_this(self) -> RouteAnswer:
        kind = Intent.WHO_IS_THIS.value

        if not self.capabilities.camera:
            return self._claim(SAY["camera_off"], kind, Err.UNAVAILABLE)

        people = self._people_now()

        if people is None:
            return self._claim(SAY["no_vision"], kind, Err.STALE)

        if not people:
            return self._claim("I don't see anyone right now.", kind)

        if len(people) > 1:
            return self._claim(
                f"I can see {len(people)} people. I won't name anyone "
                "out loud when there's more than one of you.", kind)

        p = people[0]

        if p.known and p.identity_state == "recognized" and p.display_name:
            return self._claim(f"I think that's {p.display_name}.", kind)

        return self._claim("I can see someone, but I don't know who "
                           "they are.", kind)

    def _check_floor(self) -> RouteAnswer:
        kind = Intent.CHECK_FLOOR.value

        if not self.capabilities.camera:
            return self._claim(SAY["camera_off"], kind, Err.UNAVAILABLE)

        people = self._people_now()

        if people is None:
            return self._claim(SAY["no_vision"], kind, Err.STALE)

        lying = [p for p in people if p.posture == "lying"
                 or p.fall_state in ("suspect", "asked", "alerted")]

        if lying:
            return self._claim(
                "I can see someone lying down. I can't tell whether "
                "they're resting or need help. Are you okay?", kind)

        return self._claim(
            "I don't see anyone on the floor, but I can only check what "
            "my camera can see.", kind)

    # ------------------------------------------------------------------
    # person approach ("come here")
    # ------------------------------------------------------------------

    def _make_approach(self):
        from buddy_core.vision.approach import (
            ApproachController,
            MockNavigationAdapter,
        )

        coordinator = self

        class LiveAdapter(MockNavigationAdapter):
            """Capabilities from real status; never actually moves."""

            def capabilities(self):
                return coordinator._approach_caps()

            def send(self, goal):
                # There is no navigation command for "go to a person".
                # Only the simulation profile accepts a goal, and even then
                # nothing moves: the goal is just recorded and "finished".
                if not coordinator.profile.mock_person_approach:
                    return False

                return super().send(goal)

        adapter = LiveAdapter()
        self._approach_adapter = adapter

        return ApproachController(adapter)

    def _approach_caps(self):
        from buddy_core.vision.approach import NavCapabilities

        nav, safety = self.nav, self.safety
        simulated = self.profile.mock_person_approach
        ok = self.motion_inputs_ok

        return NavCapabilities(
            available=bool(ok),
            distance_reliable=simulated or self.cfg.person_distance_validated,
            localization_ok=bool(ok and nav and nav.pose_trusted),
            obstacle_avoidance_ok=bool(ok and nav and nav.sensors_healthy),
            movement_allowed=bool(ok and safety and safety.movement_allowed),
            emergency_stop_active=self.estop,
            follow_validated=False)

    def _come_here(self) -> RouteAnswer:
        from buddy_core.vision.approach import ApproachRequest

        kind = Intent.COME_HERE.value
        people = self._people_now()
        now = self._clock()

        if not self.capabilities.camera or not people:
            return self._claim(SAY["approach_no"] if people
                               else "I can't see you right now, so I'll "
                               "stay here.", kind, Err.UNAVAILABLE)

        target = people[0] if len(people) == 1 else None
        bearing = self._bearing(target) if target else None
        recognized = bool(target and target.known
                          and target.identity_state == "recognized")
        decision = self.approach.request(ApproachRequest(
            now=now, person_key=target.person_key if target else None,
            recognized=recognized, invited=True, bearing_deg=bearing,
            bearing_confidence=0.7 if bearing is not None else 0.0,
            target_stamp=self._seen.get("people", now),
            quiet_mode=self.speech.context.quiet_mode,
            emergency=self.estop))

        if not decision.allowed:
            self._note(f"approach refused: {decision.reason}")
            reply = APPROACH_REFUSALS.get(decision.reason, SAY["approach_no"])

            return self._claim(reply, kind, Err.REFUSED_SAFETY)

        if not self.profile.mock_person_approach:
            # Defence in depth: only the mock adapter accepts a goal.
            self.approach.cancel(now, "not simulated")

            return self._claim(SAY["approach_no"], kind, Err.REFUSED_SAFETY)

        self._approach_started = now

        return self._claim("Okay, coming over.", kind)

    def _bearing(self, p: PersonIn) -> float | None:
        if p.bbox_width <= 0:
            return None

        cx = p.bbox_x + p.bbox_width / 2.0
        frac = cx / self.cfg.image_width_px - 0.5

        return frac * self.cfg.horizontal_fov_deg

    def _update_approach(self, now: float) -> None:
        ctl = self.approach

        if ctl.active is None:
            return

        people = self._people_now() or []
        goal = ctl.active

        if self.profile.mock_person_approach \
                and now - self._approach_started \
                >= self.cfg.sim_approach_duration_s:
            self._approach_adapter.finish(goal.request_id, True)

        state = ctl.update(
            now, person_visible=bool(people),
            quiet_mode=self.speech.context.quiet_mode,
            emergency=self.estop)

        if state == "succeeded":
            self._say(Kind.STATUS, "Here I am. (simulated)")
        elif state == "cancelled":
            self._say(Kind.STATUS, "Okay, I've stopped.")

    # ------------------------------------------------------------------
    # periodic
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Call at about 10 Hz."""

        self._sync_robot_state()
        self._update_approach(self._clock())
        self._sync_speech_context()
        now = self._clock()

        for rid, p in list(self.pending.items()):
            if now - p.issued > 600:
                del self.pending[rid]

    def shutdown(self) -> None:
        self.robot.go(RobotState.SHUTTING_DOWN, "shutdown")
        self.stop_all("shutdown")
        self._sync_speech_context()

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def readiness(self):
        return evaluate(
            profile=self.profile, safety=self.safety,
            safety_fresh=self.safety_fresh, obstacle=self.obstacle,
            obstacle_fresh=self.obstacle_fresh, nav=self.nav,
            nav_fresh=self.nav_fresh, conflicts=self.conflicts,
            capabilities=self.capabilities)

    def status(self) -> dict:
        report = self.readiness()
        active = self.approach.active

        return {
            "interface_version": INTERFACE_VERSION,
            "profile": self.profile.name,
            "robot_state": self.robot.state.value,
            "conversation_state": self.conversation.state.value,
            "ready_for_motion": report.ready_for_motion,
            "readiness": report.summary,
            "capabilities": self.capabilities.as_dict(),
            "hardware_conflicts": [c.message for c in self.conflicts],
            "pending_navigation": sorted(self.pending),
            "approach": {"active": active is not None,
                         "simulated": self.profile.mock_person_approach},
            "counters": dict(self.counters.__dict__),
            "speech": {"sent": self.speech.sent,
                       "dropped": dict(self.speech.dropped)},
            "state_violations": len(self.robot.violations)
            + len(self.conversation.violations),
        }
