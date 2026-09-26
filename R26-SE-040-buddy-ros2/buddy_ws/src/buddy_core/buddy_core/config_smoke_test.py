from pathlib import Path

import rclpy
from rclpy.node import Node

from buddy_core.config_loader import (
    BuddyConfigError,
    find_project_root,
    load_buddy_config,
)


class BuddyConfigSmokeTest(Node):
    def __init__(self):
        super().__init__("buddy_config_smoke_test")

        try:
            project_root = find_project_root()

            base_path = project_root / "config" / "buddy.yaml"
            development_path = (
                project_root / "config" / "development.yaml"
            )

            config = load_buddy_config(
                base_path=base_path,
                override_path=development_path,
            )

        except BuddyConfigError as exc:
            self.get_logger().error(
                f"BUDDY configuration failed: {exc}"
            )
            raise

        self.get_logger().info(
            "BUDDY configuration loaded successfully."
        )

        self.get_logger().info(
            f"Robot ID: {config['robot']['id']}"
        )

        self.get_logger().info(
            f"Robot name: {config['robot']['name']}"
        )

        self.get_logger().info(
            f"Environment: {config['runtime']['environment']}"
        )

        self.get_logger().info(
            f"Log level: {config['runtime']['log_level']}"
        )

        self.get_logger().info(
            f"Camera enabled: {config['camera']['enabled']}"
        )

        self.get_logger().info(
            f"Audio enabled: {config['audio']['enabled']}"
        )

        self.get_logger().info(
            f"Movement enabled: {config['features']['movement']}"
        )

        self.get_logger().info(
            f"Navigation enabled: {config['features']['navigation']}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = BuddyConfigSmokeTest()
    finally:
        if node is not None:
            node.destroy_node()

        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
