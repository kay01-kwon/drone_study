import numpy as np

def otimes(q1, q2):
    """
    Quaternion multiplication
    :param q1: Left quaternion
    :param q2: Right quaternion
    :return: Multiplication
    """
    qw, qx, qy, qz = q1

    q1_L = np.array([
        [qw, -qx, -qy, -qz],
        [qx, qw, -qz, qy],
        [qy, qz, qw, -qx],
        [qz, -qy, qx, qw]
    ])
    return q1_L @ q2

def vec_to_quaternion_form(w):
    wx, wy, wz = w
    return np.array([0.0, wx, wy, wz])

def quaternion_to_rotm(q):
    """
    Convert quaternion to rotation matrix (body to world)
    :param q: quaternion [qw, qx, qy, qz]
    :return: 3x3 rotation matrix
    """
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])

    return R

def pitch_to_rotm(th):
    """
    Convert pitch to rotation matrix (body to world)
    :param th: Pitch
    :return: Rotation matrix
    """
    cth = np.cos(th)
    sth = np.sin(th)
    rotm = np.array([[cth, sth],
                     [-sth, cth]])
    return rotm

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    :param q: quaternion [qw, qx, qy, qz]
    :return: Euler angles [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def compute_angle_axis(R_des, R):
    """
    Compute angle axis on SO(3) using vee map

    This gives the error vector that points along the axis of the rotation
    needed to align R with R_des, scaled by the sine of the rotation angle

    R_des = w_R_d, R = w_R_b
    R_des^T R = d_R_w w_R_b = d_R_b
    By rodrigues' rotation formula,
    d_R_b - d_R_b^T = 2 * sin(theta) * [axis]_x
    --> sin(theta) * [axis]_x = 0.5 * (d_R_b - d_R_b^T)
    Args:
        R: current rotation matrix (3x3)
        R_des: desired rotation matrix (3x3)

    Returns:
        e_R: rotation error vector (3,)

    """
    # Compute skew-symmetric error matrix
    d_R_b = R_des.T @ R
    R_err = 0.5*(d_R_b - d_R_b.T)
    e_R = vee_map(R_err)
    return e_R

def vee_map(S):
    """
    Extract s vector from skew-symmetric matrix S
    :param S:
    S = [ [0, -z, y],
        [z, 0, -x],
        [-y, x, 0] ]

    :return: s = [x, y, z]
    """
    sx = -S[1,2]
    sy = S[0,2]
    sz = -S[0,1]
    s = np.array([sx, sy, sz])
    return s

def transform_linear_vel(v_world,q):
    """
    Transform linear velocity vector from world to body frame
    :param v: linear velocity expressed by world frame
    :param q: body to world frame
    :return: R(q)^T v
    """
    R = quaternion_to_rotm(q)
    v_body = R.T @ v_world
    return v_body