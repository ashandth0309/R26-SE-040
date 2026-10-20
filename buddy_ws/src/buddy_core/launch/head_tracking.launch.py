"""Launch BUDDY head tracking.

Mock mode by default.  Real servos additionally need every gate listed in
docs/head_tracking_architecture.md (config flags AND the environment
variable BUDDY_HEAD_HARDWARE_ARMED=1).  This launch file never sets that
variable.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("head_config_path", default_value=""),
        DeclareLaunchArgument("buddy_config_path", default_value=""),
        Node(
            package="buddy_core",
            executable="head_tracking_node",
            name="head_tracking",
            output="screen",
            parameters=[{
                "head_config_path": LaunchConfiguration("head_config_path"),
                "buddy_config_path": LaunchConfiguration(
                    "buddy_config_path"),
            }],
        ),
    ])
