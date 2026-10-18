"""Launch the camera node (mock backend unless told otherwise).

Only one process may own the Pi camera.  Do not start this next to another
camera owner such as a face-recognition node that opens the device itself.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='mock'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('fps', default_value='10.0'),
        Node(
            package='buddy_navigation',
            executable='camera_node',
            name='buddy_camera',
            output='screen',
            parameters=[{
                'backend': LaunchConfiguration('backend'),
                'width': LaunchConfiguration('width'),
                'height': LaunchConfiguration('height'),
                'fps': LaunchConfiguration('fps'),
            }],
        ),
    ])
