import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from trajectory_publisher.CubicSpline.cubic_spline_planner import calc_spline_course

import numpy as np


class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher_node')

        # Target course
        self.declare_parameter('waypoints_x', [0.0, 100.0, 100.0, 50.0, 60.0])
        self.declare_parameter('waypoints_y', [0.0, 0.0, -30.0, -20.0, 0.0])
        
        self.declare_parameter('scale_factor', 1.0)
        
        self.declare_parameter('ds', 0.2)
        
        self.declare_parameter('frame_id', 'odom')
        
        self.frame_id = self.get_parameter('frame_id').value
        
        ax = self.get_parameter('waypoints_x').value
        ay = self.get_parameter('waypoints_y').value
        scale_factor = self.get_parameter('scale_factor').value
        ds = self.get_parameter('ds').value
        
        # -------------------------
        # Validation
        # -------------------------
        if len(ax) < 2:
            raise ValueError("At least 2 waypoints are required")

        if len(ax) != len(ay):
            raise ValueError("waypoints_x and waypoints_y must have the same length")

        if ds <= 0.0:
            raise ValueError("Parameter 'ds' must be greater than 0")
        
        # -------------------------
        # Generate spline path
        # -------------------------
        ax = [_ * scale_factor for _ in ax]
        ay = [_ * scale_factor for _ in ay]
        cx, cy, cyaw, ck, s = calc_spline_course(x=ax, y=ay, ds=ds)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        now = self.get_clock().now().to_msg()
        
        for _cx, _cy, _cyaw in zip(cx, cy, cyaw):
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.frame_id

            pose.pose.position.x = _cx
            pose.pose.position.y = _cy
            
            pose.pose.orientation.w = np.cos(_cyaw / 2)
            pose.pose.orientation.z = np.sin(_cyaw / 2)

            self.path_msg.poses.append(pose)
        
        # -------------------------
        # Publisher
        # -------------------------
        # 1. Define the QoS Profile
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # 2. Create the publisher with the custom QoS
        self.publisher_ = self.create_publisher(Path, 'plan', qos_profile)
        
        # 3. Publish a message (will be stored for late subscribers)
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.path_msg)



def main(args=None):
    # 1. Inicialización
    rclpy.init(args=args)

    # 2. Creación de nodo
    nodo = PathPublisher()

    try:
        # 3. Procesamiento de mensajes y callback
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        print('\n')
        pass
    else:
        # 4. Finalización 
        rclpy.shutdown()


if __name__ == '__main__':
    main()