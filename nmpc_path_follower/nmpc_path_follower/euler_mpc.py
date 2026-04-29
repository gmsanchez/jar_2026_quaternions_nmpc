import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from rclpy.wait_for_message import wait_for_message 

from geometry_msgs.msg import TwistStamped, Pose, Point #, PoseStamped
from nav_msgs.msg import Odometry, Path

from visualization_msgs.msg import Marker

from nmpc_path_follower.RigidBody2DEuler import RigidBody2DEuler

import numpy as np
from datetime import datetime

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
    if ind < current_ind:
        ind = current_ind

    return ind, mind

class EulerMPCNode(Node):
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
            # @TODO Mejorar la descripción del error
            raise TypeError
        
        self.N = 10
        self.dt = 0.1
        # Robot State
        self.x0 = np.array([0.0, 0.0, 0.0])
        self.Nx = len(self.x0)
        self.u0 = np.array([0.0, 0.0])
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

        self.mpc = RigidBody2DEuler(self.N, self.dt)
        self.mpc.setup()
        self.mpc.set_x0(self.x0)
        self.mpc.set_initial_guess(self.x0, self.u0)
        
        # Update Reference
        # self.mpc.update_reference(self.x_sp)
        self.hist_t = []
        self.hist_X0 = []
        self.hist_U0 = []
        self.hist_Xsp = []
        self.hist_iter = []
        self.hist_et = []
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        
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
        self.x0[2] = self.current_yaw
        
        # Check if we should update the path
        # distance_moved = math.hypot(self.current_x - self.last_path_x, 
        #                             self.current_y - self.last_path_y)
        # distance_moved = np.hypot(self.current_x - self.last_path_x,
        #                           self.current_y - self.last_path_y)
        
        # if distance_moved > self.dist_threshold:
        #     print("updating path")
        #     self.update_path(msg.pose.pose)

    # def update_path(self, pose):
    #     self.path.header.stamp = self.get_clock().now().to_msg()
    #     self.path.header.frame_id = self.frame_id
        
    #     new_pose = PoseStamped()
    #     new_pose.header = self.path.header
    #     new_pose.pose = pose
        
    #     self.path.poses.append(new_pose)
    #     self.path_publisher.publish(self.path)
        
    #     # Update last recorded position
    #     self.last_path_x = self.current_x
    #     self.last_path_y = self.current_y
        
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
        self.get_logger().info("Waypoint %d of %d" % (self.ind, self.path_len))
        
        self.x_sp = np.zeros((self.Nx, self.N+1))
        # Ventana completa dentro del Path
        if self.ind+self.N+1 < self.path_len:
            _first = int(self.ind)
            _last = int(self.ind+self.N+1)
            
            self.x_sp[0] = self.path_x[_first:_last]
            self.x_sp[1] = self.path_y[_first:_last]
            self.x_sp[2] = self.path_yaw[_first:_last]
        else: # Rellenar
            _first = int(self.ind)
            _fill = int(self.N-(self.path_len-self.ind)+1)
            
            self.x_sp[0] = np.concatenate((np.array(self.path_x[_first:]), np.full(_fill, self.path_x[-1])))
            self.x_sp[1] = np.concatenate((np.array(self.path_y[_first:]), np.full(_fill, self.path_y[-1])))
            self.x_sp[2] = np.concatenate((np.array(self.path_yaw[_first:]), np.full(_fill, self.path_yaw[-1])))
        
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
        if self.ind >= (self.path_len-1) or np.hypot(dx, dy) < 0.1:
            self.velocity_publisher.publish(TwistStamped())
            self.get_logger().info("Path Done!!")
            
            _dt_object = datetime.fromtimestamp(self.get_clock().now().nanoseconds / 1e9)
            _filename = _dt_object.strftime("%Y-%m-%d_%H-%M-%S_Euler") + ".npz"
            np.savez(_filename, t=self.hist_t,
                     x_sp=self.hist_Xsp, x0=self.hist_X0, u0=self.hist_U0,
                     i_c=self.hist_iter, e_t=self.hist_et,
                     path_x=self.path_x, path_y=self.path_y, path_qw=self.path_qw, path_qz=self.path_qz, path_yaw=self.path_yaw,
                     Q_pos=self.mpc.Q_pos, Q_yaw=self.mpc.Q_yaw, Qn_pos=self.mpc.Qn_pos, Qn_yaw=self.mpc.Qn_yaw,
                     R=self.mpc.R
            )
            
            self.destroy_node()
            raise SystemExit
        else:
            self.hist_t.append(self.get_clock().now().nanoseconds)
            self.hist_Xsp.append(self.x_sp.copy())
            self.hist_X0.append(self.x0.copy())
            U_sol, e_t, i_c, status = self.mpc.solve(self.x0)
                        
            self.get_logger().debug("%d: %s %d iterations in %.6f seconds" % ( self.get_clock().now().nanoseconds ,status, i_c, e_t))
            self.hist_U0.append(np.copy(U_sol))
            self.hist_iter.append(i_c)
            self.hist_et.append(e_t)
            
            msg.twist.linear.x = U_sol[0]
            msg.twist.angular.z = U_sol[1]
        
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
        
        # self.marker_msg.pose.position.x = self.path_x[self.ind]
        # self.marker_msg.pose.position.y = self.path_y[self.ind]
        # self.marker_msg.pose.position.z = 0.0
        # self.marker_msg.pose.orientation.w = np.cos(self.path_qw[self.ind] / 2)
        # self.marker_msg.pose.orientation.z = np.sin(self.path_qz[self.ind] / 2)
        
        self.marker_pub.publish(self.marker_msg)


def main(args=None):
    # 1. Inicialización
    rclpy.init(args=args)

    # 2. Creación de nodo
    nodo = EulerMPCNode()

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
