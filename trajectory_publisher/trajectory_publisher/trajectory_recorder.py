import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry, Path

import numpy as np


class PathRecorder(Node):
    def __init__(self):
        super().__init__('path_recorder_node')
        
        # Creación del suscriptor
        self.sub = self.create_subscription(Odometry, 'odom', self.sub_callback, 10)
        
        # Creación de publisher
        self.path_pub = self.create_publisher(Path, 'path', 10)
    
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        
        self.first_odom_msg = True
        self.first_pose = Pose()
        self.last_pose = Pose()
        
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    
    # Programación de función de callback
    def sub_callback(self, msg: Odometry):
        if self.first_odom_msg:
            self.first_pose = msg.pose.pose
            self.first_odom_msg = False
            
            self.last_pose = msg.pose.pose
            
            return
        
        _x_diff = msg.pose.pose.position.x - self.last_pose.position.x
        _y_diff = msg.pose.pose.position.y - self.last_pose.position.y
        _rho = np.hypot(_x_diff, _y_diff)
        
        if _rho > 0.05:  # Cada 5 cm (aprox)
            self.last_pose = msg.pose.pose
            
            _pose = PoseStamped()
            _pose.header.stamp = self.get_clock().now().to_msg()
            _pose.header.frame_id = "odom"
            _pose.pose = msg.pose.pose
            
            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_msg.poses.append(_pose)
            
            # self.path_pub.publish(self.path_msg)


    # Programación de función de callback
    def timer_callback(self):
        # Publicar el mensaje
        self.path_pub.publish(self.path_msg)

def main(args=None):
    # 1. Inicialización
    rclpy.init(args=args)

    # 2. Creación de nodo
    nodo = PathRecorder()

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