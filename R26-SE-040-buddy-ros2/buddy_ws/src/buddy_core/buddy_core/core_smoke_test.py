import rclpy
from rclpy.node import Node


class BuddyCoreSmokeTest(Node):
    def __init__(self):
        super().__init__('buddy_core_smoke_test')

        self.get_logger().info(
            'BUDDY core ROS 2 package is running.'
        )

        self.timer = self.create_timer(
            2.0,
            self.heartbeat
        )

    def heartbeat(self):
        self.get_logger().info('BUDDY core heartbeat OK.')


def main(args=None):
    rclpy.init(args=args)

    node = BuddyCoreSmokeTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
