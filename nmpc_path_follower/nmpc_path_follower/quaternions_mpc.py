import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from rclpy.wait_for_message import wait_for_message 

from geometry_msgs.msg import TwistStamped, Pose, Point
from nav_msgs.msg import Odometry, Path

from visualization_msgs.msg import Marker

from nmpc_path_follower.RigidBodyQuaternion import RigidBodyQuaternions

import numpy as np

# Transformation helper to get Yaw from Quaternion
def get_yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

def calc_nearest_index(x, y, path_x, path_y, current_ind):
    dx = x - path_x
    dy = y - path_y
    
    d = dx ** 2 + dy ** 2

    mind = min(d)
    mind = np.sqrt(mind)

    ind = np.argmin(d)
    # if ind < current_ind:
    #     ind = current_ind

    return ind, mind

class QuaternionsMPCNode(Node):
    def __init__(self):
        super().__init__('nmpc_path_follower_node')
        
        # Publishers & Subscribers
        self.velocity_publisher = self.create_publisher(TwistStamped, '/skd1_base_controller/cmd_vel', 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/ground_truth/odometry', self.odom_callback, 10)
        
        self.first_path_msg, self.plan = wait_for_message(
            Path, self, 'plan', 
            qos_profile = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE
            )
        )
        
        if self.first_path_msg and isinstance(self.plan, Path):
            self.path_len = len(self.plan.poses)
            self.path_x = np.array([_.pose.position.x for _ in self.plan.poses])
            self.path_y = np.array([_.pose.position.y for _ in self.plan.poses])
            self.path_yaw = np.array([
                np.arctan2(
                    2.0 * _.pose.orientation.w * _.pose.orientation.z,
                    1.0 - 2.0 * (_.pose.orientation.z ** 2)
                )
                for _ in self.plan.poses])
            self.path_qw = np.array([_.pose.orientation.w for _ in self.plan.poses])
            self.path_qz = np.array([_.pose.orientation.z for _ in self.plan.poses])
        else:
            raise TypeError
        
        self.N = 10
        self.dt = 0.1
        # Robot State
        self.x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.Nx = len(self.x0)
        self.u0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Nu = len(self.u0)
        
        # Odometría
        self.first_odom_msg = True
        self.first_pose = Pose()
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.frame_id = "odom" # Default frame, updated by odom callback
        
        # Punto inicial del path
        self.ind = 0
        
        # Goal State (x, y, yaw)
        self.goal_x = 10.0
        self.goal_y = 10.0
        self.goal_yaw = np.pi / 2        
        
        self.mpc = RigidBodyQuaternions(self.N, self.dt)
        self.mpc.setup()
        self.mpc.set_x0(self.x0)
        # mpc.set_initial_guess(x0, u0)
        
        # Update Reference
        # self.mpc.update_reference(self.x_sp)
    
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.path = Path()
        self.last_path_x = 0.0
        self.last_path_y = 0.0
        self.dist_threshold = 0.01  # Update path every 10cm
        
        # Marcadores de la ventana
        self.marker_pub = self.create_publisher(Marker, 'markers', 10)
        
        self.get_logger().info("NMPC Path Follower init ready.")

    def odom_callback(self, msg: Odometry):
        if self.first_odom_msg:
            self.first_pose = msg.pose.pose
            self.first_odom_msg = False
            
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = get_yaw_from_quaternion(msg.pose.pose.orientation)
        
        self.x0[0] = msg.pose.pose.position.x
        self.x0[1] = msg.pose.pose.position.y
        self.x0[2] = msg.pose.pose.position.z
        self.x0[3] = msg.pose.pose.orientation.w
        self.x0[4] = msg.pose.pose.orientation.x
        self.x0[5] = msg.pose.pose.orientation.y
        self.x0[6] = msg.pose.pose.orientation.z
        
    def control_loop(self):
        if self.first_odom_msg or not self.first_path_msg:
            return
        
        self.get_logger().debug("Into control loop.")
        
        self.ind, _ = calc_nearest_index(
            x = self.current_x, y = self.current_y,
            path_x = self.path_x,
            path_y = self.path_y,
            current_ind = self.ind,
        )
        
        self.x_sp = np.zeros((self.Nx, self.N+1))
        # Ventana completa dentro del Path
        if self.ind+self.N+1 < self.path_len:
            _first = int(self.ind)
            _last = int(self.ind+self.N+1)
            
            self.x_sp[0] = self.path_x[_first:_last]
            self.x_sp[1] = self.path_y[_first:_last]
            self.x_sp[3] = self.path_qw[_first:_last]
            self.x_sp[6] = self.path_qz[_first:_last]
        else: # Rellenar
            _first = int(self.ind)
            _fill = int(self.N-(self.path_len-self.ind)+1)
            
            self.x_sp[0] = np.concatenate((np.array(self.path_x[_first:]), np.full(_fill, self.path_x[-1])))
            self.x_sp[1] = np.concatenate((np.array(self.path_y[_first:]), np.full(_fill, self.path_y[-1])))
            self.x_sp[3] = np.concatenate((np.array(self.path_qw[_first:]), np.full(_fill, self.path_qw[-1])))
            self.x_sp[6] = np.concatenate((np.array(self.path_qz[_first:]), np.full(_fill, self.path_qz[-1])))
        
        # Update Reference
        self.mpc.update_reference(self.x_sp)
        
        self.goal_x = self.x_sp[0,-1]
        self.goal_y = self.x_sp[1,-1]
        
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        
        # Stop condition logic to check if we are "close enough"
        if np.hypot(dx, dy) < 0.05:
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
        #     # Optional: Add logic here to stop the timer or signal success
        else:
            U_sol, e_t, i_c, status = self.mpc.solve(self.x0)
            self.get_logger().info("%d: %s %d iterations in %.6f seconds" % ( self.get_clock().now().nanoseconds ,status, i_c, e_t))
            
            v = U_sol[0]
            w = U_sol[5]
            
            msg.twist.linear.x = v
            msg.twist.angular.z = w 
        
        self.get_logger().info("vx: %.4f - wz: %.4f" % (msg.twist.linear.x, msg.twist.angular.z))
        # self.get_logger().info("dx: %.4f - dy: %.4f" % (dx, dy))
        
        # Publish cmd_vel    
        self.velocity_publisher.publish(msg)
        
        
        self.publish_reference_window_markers()
    
    def publish_reference_window_markers(self):
        self.marker_msg = Marker()
        
        self.marker_msg.header.frame_id = 'odom'
        
        self.marker_msg.ns = 'lqr_target'
        self.marker_msg.id = 42
        
        self.marker_msg.type = Marker.SPHERE_LIST
        self.marker_msg.action = Marker.ADD
        
        _scale = 0.1
        self.marker_msg.scale.x = _scale
        self.marker_msg.scale.y = _scale
        self.marker_msg.scale.z = _scale
        
        self.marker_msg.color.a = 1.0
        self.marker_msg.color.r = 1.0
        
        self.marker_msg.frame_locked = True
        self.marker_msg.lifetime.sec = 5
        
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        
        self.marker_msg.points = [
            Point(x=_x, y=_y, z=0.0)
            for _x, _y in zip(self.x_sp[0], self.x_sp[1])
        ]
        
        self.marker_pub.publish(self.marker_msg)


def main(args=None):
    # 1. Inicialización
    rclpy.init(args=args)

    # 2. Creación de nodo
    nodo = QuaternionsMPCNode()

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
