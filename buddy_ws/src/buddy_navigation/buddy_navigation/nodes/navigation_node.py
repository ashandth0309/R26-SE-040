"""ROS 2 navigation node for BUDDY.

Thin wrapper around ``NavigationService``.  Interfaces:

Subscribes
  /ultrasonic/<front_left|front_right|rear_left|rear_right>  sensor_msgs/Range
  /navigation/command    buddy_nav_interfaces/NavigationCommand
  /safety_state          buddy_interfaces/SafetyState
  /movement/request      buddy_interfaces/MovementCommand (override detection)
  /sim/pose              geometry_msgs/PoseStamped (pose_source: sim only)

Publishes
  /movement/request      buddy_interfaces/MovementCommand (only while a goal
                         runs, then a short burst of STOP)
  /navigation/status     buddy_nav_interfaces/NavigationStatus
  /navigation/local_map  nav_msgs/OccupancyGrid (robot-relative, expiring)

Action server
  /patrol_route          buddy_interfaces/action/PatrolRoute

This node never talks to motors.  Requests go through the existing safety
supervisor and motor executor.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import rclpy
from buddy_interfaces.msg import MovementCommand, SafetyState
from buddy_nav_interfaces.msg import NavigationCommand, NavigationStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Range

from buddy_navigation.core.commands import STOP_REQUEST
from buddy_navigation.core.config import load_navigation_config
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.local_map import LocalObstacleMap
from buddy_navigation.core.presets import SENSOR_NAMES
from buddy_navigation.core.sensor_model import SensorStatus
from buddy_navigation.core.service import (
    OWN_SOURCE,
    CommandRequest,
    NavigationService,
)
from buddy_navigation.nodes.ros_convert import (
    fill_movement_command,
    range_to_sample,
)

COMMAND_KINDS = {
    NavigationCommand.COMMAND_STOP: "stop",
    NavigationCommand.COMMAND_MOVE_FORWARD: "move_forward",
    NavigationCommand.COMMAND_TURN_LEFT: "turn_left",
    NavigationCommand.COMMAND_TURN_RIGHT: "turn_right",
    NavigationCommand.COMMAND_START_PATROL: "start_patrol",
    NavigationCommand.COMMAND_GO_TO_LOCATION: "go_to_location",
    NavigationCommand.COMMAND_CANCEL: "cancel",
    NavigationCommand.COMMAND_PAUSE: "pause",
    NavigationCommand.COMMAND_RESUME: "resume",
}

STATE_CODES = {
    "idle": NavigationStatus.STATE_IDLE,
    "running": NavigationStatus.STATE_RUNNING,
    "paused": NavigationStatus.STATE_PAUSED,
    "succeeded": NavigationStatus.STATE_SUCCEEDED,
    "failed": NavigationStatus.STATE_FAILED,
    "cancelled": NavigationStatus.STATE_CANCELLED,
}


def default_config_path() -> str:
    """Installed physical config, or the source tree copy when run in place."""

    try:
        from ament_index_python.packages import get_package_share_directory

        return str(
            Path(get_package_share_directory("buddy_navigation"))
            / "config"
            / "navigation.yaml"
        )
    except Exception:
        return str(
            Path(__file__).resolve().parents[2] / "config" / "navigation.yaml"
        )


class NavigationNode(Node):
    """Guarded navigation, patrol and local mapping."""

    def __init__(self) -> None:
        super().__init__("buddy_navigation")

        self.declare_parameter("config_file", default_config_path())
        self.declare_parameter("control_rate_hz", 10.0)

        path = str(self.get_parameter("config_file").value)
        self.setup = load_navigation_config(path)
        self.lock = threading.RLock()
        self.sequence = 0
        self.service = NavigationService(
            self.setup,
            log=lambda text: self.get_logger().info(text),
        )
        self.local_map = LocalObstacleMap()
        self.specs = {spec.name: spec for spec in self.setup.sensors}
        self.last_state = ""

        for warning in self.setup.warnings:
            self.get_logger().warning(warning)

        self.get_logger().info(
            f"BUDDY navigation started: profile={self.setup.profile}, "
            f"motion_enabled={self.setup.navigation.allow_motion}, "
            f"pose_source={self.setup.pose_source}"
        )

        self.request_publisher = self.create_publisher(
            MovementCommand, "/movement/request", 10
        )
        self.status_publisher = self.create_publisher(
            NavigationStatus, "/navigation/status", 10
        )
        self.map_publisher = self.create_publisher(
            OccupancyGrid, "/navigation/local_map", 1
        )

        for name in SENSOR_NAMES:
            self.create_subscription(
                Range,
                f"/ultrasonic/{name}",
                lambda msg, n=name: self._on_range(n, msg),
                10,
            )

        self.create_subscription(
            NavigationCommand, "/navigation/command", self._on_command, 10
        )
        self.create_subscription(
            SafetyState, "/safety_state", self._on_safety, 10
        )
        self.create_subscription(
            MovementCommand, "/movement/request", self._on_movement, 10
        )

        if self.setup.pose_source == "sim":
            self.create_subscription(
                PoseStamped, "/sim/pose", self._on_sim_pose, 10
            )
        elif self.setup.pose_source == "odom":
            self.get_logger().error(
                "pose_source 'odom' is reserved but not implemented; "
                "the pose stays untrusted, so path and patrol goals are "
                "refused."
            )

        period = 1.0 / float(self.get_parameter("control_rate_hz").value)
        self.create_timer(period, self._tick)
        self.create_timer(1.0, self._publish_status)
        self.create_timer(0.5, self._publish_map)

        self._setup_patrol_action()

    # -- helpers ------------------------------------------------------

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -- subscribers --------------------------------------------------

    def _on_range(self, name: str, msg) -> None:
        spec = self.specs[name]
        now = self._now_s()
        sample = range_to_sample(
            name,
            float(msg.range),
            max(float(msg.min_range), spec.min_range_m),
            min(float(msg.max_range), spec.max_range_m)
            if msg.max_range > 0
            else spec.max_range_m,
            now,
        )

        with self.lock:
            self.service.update_sample(sample)
            pose = self.service.pose if self.service.pose_trusted else None
            distance = (
                sample.distance_m
                if sample.status is SensorStatus.OK
                else None
            )
            self.local_map.update(spec, sample.status, distance, now, pose)

    def _on_command(self, msg) -> None:
        kind = COMMAND_KINDS.get(int(msg.command))

        if kind is None:
            self.get_logger().warning(
                f"Ignoring unknown navigation command {int(msg.command)}."
            )

            return

        request = CommandRequest(
            kind=kind,
            request_id=str(msg.request_id),
            duration_s=float(msg.duration_s),
            route_id=str(msg.route_id),
            repeat=bool(msg.repeat),
            location_name=str(msg.location_name),
        )

        with self.lock:
            ack = self.service.handle(request, self._now_s())

        level = self.get_logger().info if ack.accepted else (
            self.get_logger().warning
        )
        level(
            f"Command '{kind}' from '{msg.source}': "
            f"{'accepted' if ack.accepted else 'refused'} ({ack.reason})"
        )

    def _on_safety(self, msg) -> None:
        with self.lock:
            self.service.set_estop(bool(msg.emergency_stop_active))

    def _on_movement(self, msg) -> None:
        with self.lock:
            self.service.note_external_movement(
                self._now_s(), str(msg.source), int(msg.command)
            )

    def _on_sim_pose(self, msg) -> None:
        q = msg.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        pose = Pose2D(
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            yaw,
        )

        with self.lock:
            self.service.set_pose(pose, True)

    # -- timers -------------------------------------------------------

    def _tick(self) -> None:
        now = self._now_s()

        with self.lock:
            request, status = self.service.tick(now)
            self.local_map.expire(now)

        if request is not None:
            message = MovementCommand()
            self.sequence += 1
            fill_movement_command(
                message,
                request,
                self.sequence,
                self.get_clock().now().to_msg(),
            )
            message.source = OWN_SOURCE
            self.request_publisher.publish(message)

        if status.state != self.last_state:
            self.last_state = status.state
            self._publish_status()

    def _publish_status(self) -> None:
        with self.lock:
            status = self.service.status(self._now_s())

        message = NavigationStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.state = STATE_CODES[status.state]
        message.request_id = status.request_id
        message.active_command = status.active_command
        message.reason = status.reason
        message.motion_enabled = status.motion_enabled
        message.pose_trusted = status.pose_trusted
        message.sensors_healthy = status.sensors_healthy
        message.emergency_stop_active = status.emergency_stop_active
        message.progress = float(status.progress)
        self.status_publisher.publish(message)

    def _publish_map(self) -> None:
        with self.lock:
            grid = self.local_map.to_occupancy_grid(
                "map" if self.service.pose_trusted else "base_link"
            )

        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = grid.frame_id
        message.info.resolution = float(grid.resolution_m)
        message.info.width = grid.width
        message.info.height = grid.height
        message.info.origin.position.x = float(grid.origin_x)
        message.info.origin.position.y = float(grid.origin_y)
        message.info.origin.orientation.w = 1.0
        message.data = [int(v) for v in grid.data.reshape(-1)]
        self.map_publisher.publish(message)

    # -- patrol action (needs rclpy.action, untestable without ROS) -----

    def _setup_patrol_action(self) -> None:
        try:
            from buddy_interfaces.action import PatrolRoute
            from rclpy.action import ActionServer, CancelResponse, GoalResponse
            from rclpy.callback_groups import ReentrantCallbackGroup
        except ImportError as exc:
            self.get_logger().warning(
                f"Patrol action server unavailable: {exc}"
            )

            return

        self._patrol_action = PatrolRoute

        self._action_server = ActionServer(
            self,
            PatrolRoute,
            "/patrol_route",
            execute_callback=self._execute_patrol,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    def _execute_patrol(self, goal_handle):
        action = self._patrol_action
        result = action.Result()
        request = CommandRequest(
            "start_patrol",
            route_id=str(goal_handle.request.route_id),
            repeat=bool(goal_handle.request.repeat),
        )

        with self.lock:
            ack = self.service.handle(request, self._now_s())

        if not ack.accepted:
            goal_handle.abort()
            result.success = False
            result.reason = ack.reason

            return result

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                with self.lock:
                    self.service.handle(
                        CommandRequest("cancel"), self._now_s()
                    )

                goal_handle.canceled()
                result.success = False
                result.reason = "Patrol cancelled."

                return result

            with self.lock:
                feedback = self.service.patrol.feedback()

            message = action.Feedback()
            message.state = feedback.state.value
            message.current_waypoint = int(feedback.current_waypoint)
            message.total_waypoints = int(feedback.total_waypoints)
            message.progress = float(feedback.progress)
            goal_handle.publish_feedback(message)

            if feedback.state.value in ("completed", "failed", "cancelled"):
                break

            time.sleep(0.2)

        success = feedback.state.value == "completed"

        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        result.success = success
        result.reason = feedback.reason

        return result

    def destroy_node(self) -> bool:
        with self.lock:
            self.service.handle(CommandRequest("stop"), self._now_s())

        try:
            # Say STOP explicitly instead of relying on command expiry.
            message = MovementCommand()
            fill_movement_command(
                message,
                STOP_REQUEST,
                self.sequence + 1,
                self.get_clock().now().to_msg(),
            )
            self.request_publisher.publish(message)
        except Exception:
            pass

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = NavigationNode()
        from rclpy.executors import MultiThreadedExecutor

        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
