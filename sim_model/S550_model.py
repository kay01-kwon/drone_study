"""
S550 simulation model
- Full 6 DOF Dynamics (Position + Attitude)
- Quaternion-based attitude dynamics
- COM offset included
- Ground contact included
"""

import numpy as np
from dataclasses import dataclass
from utils.math_tool import quaternion_to_rotm, vec_to_quaternion_form, otimes, quaternion_to_euler
from typing import Tuple

@dataclass
class S550_state:
    """
    Complete state of S550
    """
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    w: np.ndarray

class S550_Sim_Model:
    """
    S550_Sim_Model
    :param : DynamicParams
    """

    def __init__(self, DynamicParams):

        # Dynamics parameter
        self.m = DynamicParams['m']
        Jxx = DynamicParams['MoiArray'][0]
        Jyy = DynamicParams['MoiArray'][1]
        Jzz = DynamicParams['MoiArray'][2]
        self.J = np.diag([Jxx, Jyy, Jzz])
        self.J_inv = np.diag([1/Jxx, 1/Jyy, 1/Jzz])
        self.r_off = DynamicParams['com_offset']
        self.g = -9.81
        self.g_vec = np.array([0, 0, self.g])

        self.e3 = np.array([0, 0, 1.0])

        # Landing gear geometry (body frame) - from CAD drawing
        self.landing_gear_width = 0.28845  # Distance between landing gear legs [m]
        self.landing_gear_height = 0.28837  # Height from ground to COM [m]
        # Contact points in body frame (2 legs, left and right)
        # COM is at origin, contact points are below at -height
        self.contact_points_body = np.array([
            [0.0, self.landing_gear_width/2, -self.landing_gear_height],   # Right leg
            [0.0, -self.landing_gear_width/2, -self.landing_gear_height]   # Left leg
        ])
        self.tip_over_angle = np.deg2rad(80)  # Maximum roll/pitch before stuck [rad]

    def pack_state(self, p, v, q, w):
        """Pack state into vector"""
        return np.concatenate([p, v, q, w])

    def unpack_state(self, s):
        """
        Unpack state from vector
        p: position in the world frame
        v_world: velocity in the world frame
        q: body to world frame (quaternion)
        w: angular velocity in the body frame
        """
        p = s[0:3]
        v_world = s[3:6]
        q = s[6:10]
        w = s[10:13]
        return p, v_world, q, w

    def pack_control_input(self,f,M):
        """Pack control input into vector"""
        return np.concatenate([f, M])

    def unpack_control_input(self, s):
        """Unpack control input from vector"""
        f = s[0]
        M = s[1:4]
        return f, M

    def dynamics(self, t, s, u):
        """
        Dynamics function of S550 - COM offset included
        :param t: time (Not use, but required for rk4 interface)
        :param s: state vector [13]
        :param u: control input - Thrust [1], Moment [3]
        :return: dsdt state derivative [13]
        """

        # Unpack state and control input
        p, v_world, q, w = self.unpack_state(s)
        # Normalize quaternion
        q = q/np.linalg.norm(q,2)
        f, M = self.unpack_control_input(u)

        # Simple ground contact (like Gazebo)
        R = quaternion_to_rotm(q)
        z = p[2]

        # Ground contact force (simple spring-damper)
        if z < 0.0:
            K_ground = 5000.0  # N/m
            D_ground = 200.0   # N*s/m
            mu = 0.8           # Friction coefficient

            # Normal force
            f_normal = np.array([0.0, 0.0, -K_ground * z - D_ground * v_world[2]])
            f_normal[2] = max(0.0, f_normal[2])  # Only upward

            # Friction force (proportional to normal force and velocity)
            v_horizontal = np.array([v_world[0], v_world[1], 0.0])
            v_h_norm = np.linalg.norm(v_horizontal)
            if v_h_norm > 1e-6:
                f_friction = -mu * f_normal[2] * v_horizontal / v_h_norm
            else:
                f_friction = np.zeros(3)
            f_friction += -D_ground * v_horizontal  # Viscous damping

            f_ground = f_normal + f_friction
        else:
            f_ground = np.zeros(3)

        # Linear dynamics
        dpdt = v_world
        dvdt = 1/self.m * (R @ (f*self.e3) + f_ground) + self.g_vec

        # Angular dynamics (simplified - no ground torque)
        w_quat = vec_to_quaternion_form(w)
        dqdt = 0.5*otimes(q, w_quat)
        J_w = self.J @ w
        dwdt = self.J_inv @ (M - np.cross(w,J_w)
                         - np.cross(self.r_off, f*self.e3))

        return self.pack_state(dpdt, dvdt, dqdt, dwdt)