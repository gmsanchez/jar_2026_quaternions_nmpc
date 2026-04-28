from launch import LaunchDescription
from launch_ros.actions import Node

import numpy as np


def generate_launch_description():
    
    a = 5.0
    N_waypoints = 12  # A OJO - No cambiar así nomás

    # Muestreo fino de la curva
    num_samples = 1000
    t_fine = np.linspace(0, 2*np.pi, num_samples)
    x_fine = a * np.cos(t_fine)
    y_fine = a * np.sin(t_fine) * np.cos(t_fine)

    # Calcular distancias entre puntos consecutivos
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    ds = np.hypot(dx, dy)

    # Longitud de arco acumulada
    s = np.insert(np.cumsum(ds), 0, 0)
    total_length = s[-1]
    
    # Longitudes de arco deseadas para los waypoints
    s_waypoints = np.linspace(0, total_length, N_waypoints + 1)

    # Interpolación para obtener los (x, y) correspondientes
    waypoints_x = np.interp(s_waypoints, s, x_fine)
    waypoints_y = np.interp(s_waypoints, s, y_fine)
    
    # Rotar
    _aux = waypoints_x.copy()
    waypoints_x = waypoints_y
    waypoints_y = -_aux + a
    
    scale_factor = 1.0
    
    return LaunchDescription([
        Node(
            package='trajectory_publisher',
            executable='cubic_spline_publisher',
            name='cubic_spline_publisher_node',
            parameters=[{
                'waypoints_x': waypoints_x.tolist(),
                'waypoints_y': waypoints_y.tolist(),
                'scale_factor': scale_factor,
            }]
        ),
    ])