"""Launch BUDDY navigation (physical profile by default).

This starts only the navigation node.  It expects the obstacle sensor node,
safety supervisor and motor executor to be running already (see the demo
scripts in buddy_bringup).  Physical motion stays disabled until the
configuration is edited, see docs/yohan_mapping_navigation.md.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('buddy_navigation')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution(
                [share, 'config', 'navigation.yaml']
            ),
            description='Navigation configuration YAML',
        ),
        Node(
            package='buddy_navigation',
            executable='navigation_node',
            name='buddy_navigation',
            output='screen',
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
