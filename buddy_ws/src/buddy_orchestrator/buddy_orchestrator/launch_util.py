"""Turn ``launch_support`` specs into ``launch_ros`` actions."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _resolve(value):
    if isinstance(value, tuple) and value and value[0] == "share":
        return get_package_share_directory(value[1]) + "/" + value[2]

    return value


def description(specs):
    return LaunchDescription([
        Node(package=s["package"], executable=s["executable"],
             name=s["name"], output="screen",
             parameters=[{k: _resolve(v)
                          for k, v in s["parameters"].items()}])
        for s in specs])
