import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import numpy as np


class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher_node')
        
        self.declare_parameter('scale_factor', 5.0)
        
        self.declare_parameter('ds', 0.2)

        self.declare_parameter('frame_id', 'odom')
        
        self.frame_id = self.get_parameter('frame_id').value
    
        scale_factor = self.get_parameter('scale_factor').value
        self.ds = self.get_parameter('ds').value
        
        # -------------------------
        # Validation
        # -------------------------
        if self.ds <= 0.0:
            raise ValueError("Parameter 'ds' must be greater than 0")
        
        # -------------------------
        # Generate path
        # -------------------------
        vertices = [
            [0, 0],
            [scale_factor, 0],
            [scale_factor, scale_factor],
            [0, scale_factor],
        ]
        
        # Listas para almacenar las coordenadas y la orientación (Yaw)
        cx, cy, cyaw = [], [], []
                
        # Recorrer cada lado del cuadrado
        for i in range(4):
            # Obtener el inicio y fin del lado
            inicio = np.array(vertices[i])
            fin = np.array(vertices[(i + 1) % 4])

            # Calcular el vector dirección
            direccion = fin - inicio
            longitud = np.linalg.norm(direccion)

            # Normalizar la dirección
            direccion_normalizada = direccion / longitud
            
            # Calcular la orientación (Yaw) en función del lado
            if i == 0:
                yaw = 0  # Yaw = 0 (eje X positivo)
            elif i == 1:
                yaw = np.pi / 2  # Yaw = pi/2 (eje Y negativo)
            elif i == 2:
                yaw = np.pi  # Yaw = pi (eje X negativo)
            else:
                yaw = -np.pi / 2  # Yaw = -pi/2 (eje Y positivo)
            
            # Generar puntos en ese lado
            num_puntos = int(np.floor(longitud / self.ds))
            for j in range(num_puntos + 1):  # +1 para incluir el punto final
                punto = inicio + direccion_normalizada * j * self.ds
                cx.append(punto[0])
                cy.append(punto[1])
                cyaw.append(yaw)

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