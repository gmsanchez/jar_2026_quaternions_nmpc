from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    
    # # Path 01 (Default) - Stanley control
    # # https://atsushisakai.github.io/PythonRobotics/modules/6_path_tracking/stanley_control/stanley_control.html
    # waypoints_x = [0.0, 100.0, 100.0, 50.0, 60.0]
    # waypoints_y = [0.0, 0.0, -30.0, -20.0, 0.0]
    # scale_factor = 0.1
    
    # # Path 02 - Rear wheel feedback control
    # # https://atsushisakai.github.io/PythonRobotics/modules/6_path_tracking/rear_wheel_feedback_control/rear_wheel_feedback_control.html
    # waypoints_x = [0.0, 6.0, 12.5, 5.0, 7.5, 3.0, -1.0]
    # waypoints_y = [0.0, 0.0, 5.0, 6.5, 3.0, 5.0, -2.0]
    # scale_factor = 0.75
    
    # # Path 03 - Linear–quadratic regulator (LQR) steering control
    # # https://atsushisakai.github.io/PythonRobotics/modules/6_path_tracking/lqr_steering_control/lqr_steering_control.html
    # waypoints_x = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]
    # waypoints_y = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]
    # scale_factor = 0.5
    
    # # Path 04 - Linear–quadratic regulator (LQR) speed and steering control
    # # https://atsushisakai.github.io/PythonRobotics/modules/6_path_tracking/lqr_speed_and_steering_control/lqr_speed_and_steering_control.html
    # waypoints_x = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    # waypoints_y = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    # scale_factor = 0.25
    
    # # Path 05 - Model predictive speed and steering control
    # # https://atsushisakai.github.io/PythonRobotics/modules/6_path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html
    # waypoints_x = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    # waypoints_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # scale_factor = 0.25
    
    # # Path 06
    # waypoints_x = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    # waypoints_y = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    # scale_factor = 0.1
    
    # # Path 07
    # waypoints_x = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    # waypoints_y = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    # scale_factor = 0.1
    
    # Path 02 (modified)
    waypoints_x = [1.0, 6.0, 12.5, 5.0, 7.5, 3.0, -3.0]
    waypoints_y = [0.0, 0.0, 5.0, 6.5, 3.0, 5.0, -2.0]
    scale_factor = 0.75
    
    return LaunchDescription([
        Node(
            package='trajectory_publisher',
            # namespace='',
            executable='cubic_spline_publisher',
            name='cubic_spline_publisher_node',
            # arguments=['--ros-args', '--log-level', 'info']
            parameters=[{
                'waypoints_x': waypoints_x,
                'waypoints_y': waypoints_y,
                'scale_factor': scale_factor,
                'ds': 0.1,
            }]
        ),
    ])