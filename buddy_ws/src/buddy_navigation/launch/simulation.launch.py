"""Run the navigation stack against the simulated world (no hardware).

Starts the simulator, the REAL safety supervisor and the navigation node
with the simulation profile.  The motor executor is intentionally not
started: the simulator consumes ``/movement/safe`` instead.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('buddy_navigation')

    return LaunchDescription([
        DeclareLaunchArgument('fault_sensor', default_value=''),
        DeclareLaunchArgument('fault_mode', default_value='no_response'),
        Node(
            package='buddy_navigation',
            executable='sim_node',
            name='buddy_navigation_sim',
            output='screen',
            parameters=[{
                'fault_sensor': LaunchConfiguration('fault_sensor'),
                'fault_mode': LaunchConfiguration('fault_mode'),
            }],
        ),
        Node(
            package='buddy_core',
            executable='safety_supervisor',
            name='safety_supervisor',
            output='screen',
        ),
        Node(
            package='buddy_navigation',
            executable='navigation_node',
            name='buddy_navigation',
            output='screen',
            parameters=[{
                'config_file': PathJoinSubstitution(
                    [share, 'config', 'navigation_sim.yaml']
                ),
            }],
        ),
    ])
