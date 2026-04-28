import casadi
# import utils

def quat_normalize(q):
    """Normalize a quaternion to unit length"""
    q_norm = casadi.norm_2(q)
    if q_norm == 0:
        return q
    else:
        return casadi.vertcat(q / casadi.norm_2(q))

# def ca_quat_normalize(q):
#     # q_norm = casadi.norm_2(q)
#     return utils.vertcat(q / casadi.norm_2(q))

def quaternion_conjugate(q):
    q_s = q[0]
    q_v = q[1:4]
    return casadi.vertcat(q_s, -q_v[0], -q_v[1], -q_v[2])


def vector_to_quaternion(x):
    """
    :param x: a vector in R^3
    :return: a quaternion q, where the scalar part is located in q[0]
    """
    return casadi.vertcat(0, x[0], x[1], x[2])


def quaternion_product(q1, q2):
    """
    :param q1: a quaternion [q_s, q_v], where q_s is the scalar part and q_v the vector part.
    :param q2: a quaternion [q_s, q_v], where q_s is the scalar part and q_v the vector part.
    :return: the quaternion product q1 * q2
    """
    q_s1 = q1[0]
    q_v1 = q1[1:4]
    q_s2 = q2[0]
    q_v2 = q2[1:4]
    return casadi.vertcat(q_s1 * q_s2 - casadi.dot(q_v1, q_v2),
                        q_s1 * q_v2 + q_s2 * q_v1 + casadi.cross(q_v1, q_v2))


def quaternion_vector_product(q, r):
    q_s = q[0]
    q_v = q[1:4]
    return casadi.vertcat(-casadi.dot(q_v, r), q_s * r + casadi.cross(q_v, r))


def quaternion_as_rotation_matrix(q):
    """
    :param q: a quaternion [q_s, q_v], where q_s is the scalar part and q_v the vector part.
    :return: the rotation matrix representation of the given quaternion.
    """
    q_s = q[0]
    q_v = q[1:4]
    S_q = casadi.skew(q_v)
    return casadi.vertcat(casadi.DM.eye(3) + 2.0*q_s*S_q + 2.0*casadi.mtimes(S_q, S_q))


def quaternion_vector_rotation(q, r):
    q_conj = quaternion_conjugate(q)
    q_r = vector_to_quaternion(r)
    # return quaternion_product(quaternion_product(q_conj, q_r), q)[1:]
    return casadi.vertcat(quaternion_product(quaternion_product(q, q_r), q_conj)[1:])

def quaternion_to_euler_angles(qvec):
    # import numpy as np
    """
    :param qvec: quaternion as a vector with the scalar part in [0] and the vector part in [1:3]
    :return: roll, pitch and yaw
    """
    qx = qvec[1]  # odom_msg.pose.pose.orientation.x
    qy = qvec[2]  # odom_msg.pose.pose.orientation.y
    qz = qvec[3]  # odom_msg.pose.pose.orientation.z
    qw = qvec[0]  # odom_msg.pose.pose.orientation.w

    ysqr = qy * qy
    t0 = -2.0 * (ysqr + qz * qz) + 1.0
    t1 = +2.0 * (qx * qy - qw * qz)
    t2 = -2.0 * (qx * qz + qw * qy)
    t3 = +2.0 * (qy * qz - qw * qx)
    t4 = -2.0 * (qx * qx + ysqr) + 1.0

    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    # pitch = -np.arcsin(t2)
    # roll = -np.arctan2(t3, t4)
    # yaw = -np.arctan2(t1, t0)
    pitch = -casadi.arcsin(t2)
    roll = -casadi.arctan2(t3, t4)
    yaw = -casadi.arctan2(t1, t0)
    return casadi.vertcat([roll, pitch, yaw])


def euler_angles_to_quaternion(evec):
    # import numpy as np
    """
    :param evec: euler angles (roll, pitch and yaw) as a vector expressed in radians
    :return: a quaternion with the scalar part in [0] and the vector part in [1:3]
    """
    roll = evec[0]
    pitch = evec[1]
    yaw = evec[2]
    cr = casadi.cos(roll * 0.5)
    sr = casadi.sin(roll * 0.5)
    cp = casadi.cos(pitch * 0.5)
    sp = casadi.sin(pitch * 0.5)
    cy = casadi.cos(yaw * 0.5)
    sy = casadi.sin(yaw * 0.5)
    
    _w = cr * cp * cy + sr * sp * sy
    _x = sr * cp * cy - cr * sp * sy
    _y = cr * sp * cy + sr * cp * sy
    _z = cr * cp * sy - sr * sp * cy
    
    return casadi.vertcat([_w, _x, _y, _z])


def quat_error_lie(q, q_ref):
    dq = quaternion_product(quaternion_conjugate(q_ref), q)
    return 2.0 * dq[1:]  # Lie-algebra rotation error
#     dq = quat_mul(quat_inv(q_ref), q)
#     return 2 * dq[1:]   # Lie-algebra rotation error

def real(q):
    return casadi.vertcat(q[0])