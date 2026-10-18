"""An in-process BUDDY: real components, no ROS, no hardware.

Real: the coordinator, the voice engine / command router / dialogue /
speaker queue (with simulated audio backends), the vision pipeline logic
(with fake detectors) and the navigation service driving a mock robot in
a 2-D world with mock ultrasonic sensors.

Faked: microphones, speaker, camera, face / object / pose models, motors,
the ROS transport (direct calls replace topics) and the safety supervisor
(a latch mirroring its documented behaviour; the real supervisor is
covered by its own tests).

Every result produced with this harness is a MOCK result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skeletons import hold, lying, standing, transition
from test_vision_context import (
    FakeFaces,
    FakeObjects,
    FakePose,
    person_store,
)
from test_vision_node_core import core_for
from test_voice_integration import make

from buddy_core.vision.objects import ObjectDetection
from buddy_core.vision.vision_node import Outputs
from buddy_core.vision.vision_settings import VisionSettings
from buddy_core.voice.speaker_manager import Priority
from buddy_core.voice.stt import Transcript
from buddy_core.voice.voice_engine import GatewayAnswer
from buddy_navigation.core.commands import MoveCode, MoveRequest
from buddy_navigation.core.config import load_navigation_config
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.service import CommandRequest, NavigationService
from buddy_navigation.sim.mock_robot import MockRobot
from buddy_navigation.sim.mock_ultrasonic import MockUltrasonicArray
from buddy_navigation.sim.world import Box, six_by_six_world
from buddy_orchestrator.coordinator import (
    BuddyCoordinator,
    CoordinatorConfig,
)
from buddy_orchestrator.models import (
    NAV_KINDS,
    MoveIn,
    NavRequest,
    NavStatusIn,
    ObstacleIn,
    PersonIn,
    SafetyIn,
    VisionEventIn,
)
from buddy_orchestrator.ports import Ports, VisionCommandResult
from buddy_orchestrator.profiles import get_profile
from buddy_orchestrator.speech import Utterance

NAV_CONFIG = (Path(__file__).resolve().parents[2] / "buddy_navigation"
              / "config" / "navigation_sim.yaml")
PERSON_BOX = (100, 50, 560, 470)
# A bright, textured frame: the object-teaching quality check rejects the
# flat grey test image as "too dark / not sharp".
GOOD_IMAGE = np.random.default_rng(7).integers(
    90, 200, (480, 640, 3), dtype=np.uint8)


class VisionAdapter:
    """VisionPort over the real VisionNodeCore."""

    def __init__(self, core):
        self.core = core

    def command(self, text, person_id=""):
        r = self.core.command(text, person_id)

        return VisionCommandResult(r.handled, r.reply, r.pass_to_dialogue)

    def identify(self):
        return self.core.identify_object()

    def teach(self, action, text):
        return self.core.teach_object(action, text)

    def where_is(self, name):
        return self.core.where_is(name)


class System:
    """Voice + vision + navigation + coordinator on one shared timeline."""

    def __init__(self, profile="simulation", *, person=None,
                 objects=None, pose_frames=None, with_vision=True,
                 trusted=True, start=(0.5, 0.5), seed=1):
        self.profile = profile
        self.t = 0.0
        self.crashes = []

        # ----- navigation -----
        self.setup = load_navigation_config(NAV_CONFIG)
        self.world = six_by_six_world(1.0)
        self.nav = NavigationService(self.setup)
        self.robot = MockRobot(self.world, self.setup.footprint,
                               Pose2D(*start, 0.0))
        self.array = MockUltrasonicArray(self.world, self.setup.sensors)
        self.trusted = trusted
        self.estop = False
        self.moves = []              # every request that reached the robot

        # ----- vision -----
        self.image = GOOD_IMAGE
        self.camera_dead = False
        self.vision_core = None
        self.vclock = None
        self.visible_people = []
        self.events = []
        self.people_out = []

        if with_vision:
            store, pid = person_store() if person == "known" \
                else (None, None)
            dets = list(objects or [])
            kw = {"objects": FakeObjects(dets)}

            if pose_frames is not None:
                kw["pose"] = FakePose(pose_frames)
                kw["settings"] = VisionSettings(
                    profile="enhanced",
                    overrides={"pose": 0.5, "objects": 0.5})

            self.vision_core, self.vclock = core_for(
                FakeFaces(pid if person == "known" else None,
                          present=person is not None),
                store, **kw)

        # ----- voice -----
        self.comp, self.runner, self.src, self.sink, self.be, \
            self.vc = make(_nav_settings(),
                           navigation_sink=_NavSink(self))
        self.comp.router._movement = _MoveSink(self)
        self.spoken = []
        speaker = self.comp.speaker
        original = speaker.say

        def recording(text, samples, rate, priority=None, **kw):
            self.spoken.append((self.t, text, int(priority)
                                if priority is not None else None))

            return original(text, samples, rate, priority=priority, **kw)

        speaker.say = recording
        self.comp.engine.turn.wake()
        self.comp.engine.turn.nothing_to_say()

        # ----- coordinator -----
        vision = VisionAdapter(self.vision_core) \
            if self.vision_core else None
        self.coord = BuddyCoordinator(
            get_profile(profile),
            Ports(self._speak, self._nav_send, self._movement_stop,
                  self._movement_relay, vision),
            config=CoordinatorConfig(
                default_patrol_route="demo_loop"
                if profile == "simulation" else ""),
            clock=lambda: self.t)
        self.comp.engine.utterance_gateway = self._gateway
        self._next = {"safety": 0.0, "obstacle": 0.0, "nav": 0.0,
                      "vision": 0.0, "frame": 0.0, "voice": 0.0}
        self.run(1.5)

    # ----- voice wiring -----

    def _gateway(self, text, confidence):
        a = self.coord.route_utterance(text, confidence or 0.0)

        return GatewayAnswer(a.claimed, a.reply, a.intent)

    def _speak(self, u: Utterance):
        engine = self.comp.engine
        priority = Priority(min(max(int(u.priority), 0), 3))

        if engine.quiet_mode and priority > Priority.HIGH:
            return

        if u.interrupt and priority <= Priority.HIGH:
            self.comp.speaker.cancel_low_priority()

        engine.say(u.text, priority)

    # ----- coordinator ports -----

    def _nav_send(self, r: NavRequest):
        self.nav.handle(CommandRequest(
            r.kind, r.request_id, r.duration_s, r.route_id, r.repeat,
            r.location_name), self.t)

        if r.kind == "stop":
            self._robot_stop()

    def _movement_stop(self, reason):
        self._robot_stop()

    def _movement_relay(self, m: MoveIn):
        code = MoveCode(m.command)
        self._apply(MoveRequest(code, max(m.linear_speed, 0.1),
                                max(m.valid_for_s, 0.1), "relay"))

    def _robot_stop(self):
        self._apply(MoveRequest(MoveCode.STOP, 0.0, 0.2, "stop"))

    def _apply(self, request):
        if self.estop and request.code is not MoveCode.STOP:
            return

        self.moves.append((self.t, request))
        self.robot.apply(request, self.t)

    # ----- user actions -----

    def say(self, text, confidence=0.9, wait=True):
        """One finished user utterance.

        It goes through the engine's own split between stop words,
        commands and conversation, and then the coordinator gateway.
        """

        turn = self.comp.engine.turn
        turn.wake()
        turn.user_started()
        turn.user_ended()
        self.comp.engine._handle_awake_text(None, Transcript(
            text, confidence))

        if wait:
            self.wait_quiet()
        else:
            self.run(0.05)

    def wait_quiet(self, limit=15.0):
        """Run until nothing is queued or playing (or ``limit``)."""

        end = self.t + limit
        self.run(0.2)

        while self.t < end and self._speaking():
            self.run(0.2)

    def _speaking(self):
        return bool(self.comp.speaker.queued) or \
            self.be.playback.is_playing()

    command = say

    def stop(self):
        """The user says 'stop'.

        The router publishes it on the relay and on the direct route; the
        direct route does not touch the coordinator.
        """

        result = self.command("stop")

        return result

    def press_estop(self):
        self.estop = True
        self.nav.set_estop(True)
        self._robot_stop()

    def release_estop(self):
        self.estop = False
        self.nav.set_estop(False)

    def settle(self, seconds=0.3):
        self.run(seconds)

    # ----- time -----

    def run(self, seconds, dt=0.01):
        end = self.t + seconds

        while self.t < end:
            self._step(dt)

    def _step(self, dt):
        sample = self.array.step(self.robot.pose, self.t)

        if sample:
            self.nav.update_sample(sample)

        self.nav.set_pose(self.robot.pose, self.trusted)
        request, _ = self.nav.tick(self.t)

        if request is not None:
            self._apply(request)

        self.robot.step(self.t, dt)
        self.t += dt
        self.vc.advance(dt)

        if self.t >= self._next["frame"] and self.vision_core:
            self._next["frame"] = self.t + 0.5

            if not self.camera_dead:
                self._vision_frame()

        self._publish_statuses()
        self.coord.tick()

        try:
            self.runner.step()
        except Exception as exc:                    # pragma: no cover
            self.crashes.append(repr(exc))

    def _vision_frame(self):
        self.vclock.t += 0.5
        out = self.vision_core.on_frame(self.image)

        if out is None:
            return

        self._consume(out)

    def _consume(self, out: Outputs, keep_people=False):
        people = [PersonIn(
            person_key=d["person_key"], known=bool(d["known"]),
            display_name=d["display_name"],
            identity_state=d["identity_state"], posture=d["posture"],
            motion=d["motion"], fall_state=d["fall_state"],
            bbox_x=float(d["box"][0]),
            bbox_width=float(d["box"][2] - d["box"][0]))
            for d in out.people]
        if not keep_people:
            self.people_out = people
            self.coord.on_people(people)

        for e in out.events:
            ev = VisionEventIn(
                e["kind"], e["person_key"], e["severity"],
                e["suggested_text"], bool(e["needs_voice_check"]))
            self.events.append(ev)
            self.coord.on_vision_event(ev)

    def _publish_statuses(self):
        n = self._next
        t = self.t

        if t >= n["safety"]:
            n["safety"] = t + 0.25
            self.coord.on_safety(SafetyIn(self.estop, not self.estop))

        if t >= n["obstacle"]:
            n["obstacle"] = t + 0.25
            states = self.nav.monitor.snapshot(t)
            ok = not self.nav.coordinator.assessor.assess(states).degraded
            self.coord.on_obstacle(ObstacleIn(ok, ok, ok, ok, not ok))

        if t >= n["nav"]:
            n["nav"] = t + 0.5
            s = self.nav.status(t)
            self.coord.on_nav_status(NavStatusIn(
                s.state, s.request_id, s.active_command, s.reason,
                s.motion_enabled, s.pose_trusted, s.sensors_healthy,
                s.emergency_stop_active, s.progress))

        if t >= n["voice"]:
            n["voice"] = t + 1.0
            status = self.comp.status()
            self.coord.on_voice_status(status)

            if self.vision_core:
                self.vision_core.on_voice_status(status)

        if t >= n["vision"] and self.vision_core:
            n["vision"] = t + 1.0
            if self.camera_dead:
                self.vclock.t += 1.0         # time passes, no frames

            self.vision_core.check_camera(self.vclock.t)
            deferred = self.vision_core.take_deferred()
            self._consume(deferred, keep_people=True)
            self.coord.on_vision_status(self.vision_core.status())

    # ----- queries -----

    @property
    def texts(self):
        return [text for _, text, _ in self.spoken]

    def spoke(self, fragment):
        return any(fragment.lower() in t.lower() for t in self.texts)

    @property
    def collided(self):
        return self.robot.collided

    def add_obstacle(self, xmin, ymin, xmax, ymax):
        self.world.add_obstacle(Box(xmin, ymin, xmax, ymax))


def _nav_settings():
    from buddy_core.voice.config import VoiceSettings

    s = VoiceSettings()
    s.navigation.enabled = True
    s.navigation.locations = ["corner", "far", "start"]

    return s


class _NavSink:
    """Voice navigation sink: relay to the coordinator, STOP also direct."""

    def __init__(self, system: System):
        self.system = system

    def send_navigation(self, command, request_id, source, duration_s,
                        location_name):
        kind = NAV_KINDS[int(command)]
        req = NavRequest(kind, request_id, source, duration_s,
                         location_name=location_name)
        self.system.coord.on_voice_navigation(req)

        if kind == "stop":                 # the direct route
            self.system.nav.handle(CommandRequest("stop", request_id),
                                   self.system.t)
            self.system._robot_stop()


class _MoveSink:
    """Voice movement sink: relay to the coordinator, STOP also direct."""

    def __init__(self, system: System):
        self.system = system

    def send_movement(self, command, linear, angular, valid_for_s,
                      emergency, sequence_id, source):
        m = MoveIn(int(command), source, int(sequence_id), linear,
                   angular, valid_for_s, emergency)
        if int(command) == 0:              # the direct route, first
            self.system._robot_stop()

        self.system.coord.on_voice_movement(m)


CUP_DETECTION = ObjectDetection("cup", .9, (200, 150, 400, 350), "other")


def person_detection():
    return ObjectDetection("person", .9, PERSON_BOX, "other")


def fall_frames():
    frames = [p for _, p in hold(standing(), 0.0, 3.0, fps=2)]
    frames += [p for _, p in transition(standing(), lying(), 0.0, 0.5,
                                        fps=2)]
    frames += [p for _, p in hold(lying(), 0.0, 60.0, fps=2)]

    return frames
