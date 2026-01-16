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

        # Landing gear geometry (body frame)
        self.landing_gear_width = 0.277  # Distance between landing gear legs [m]
        self.landing_gear_height = 0.256  # Landing gear leg length [m]
        # Contact points in body frame (2 legs, left and right)
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

        # Ground contact
        z = p[2]
        R = quaternion_to_rotm(q)
        vx = v_world[0]
        vy = v_world[1]
        vz = v_world[2]

        # Check ground contact using landing gear geometry
        contact_detected = False
        f_contact_total = np.zeros(3)
        M_contact_total = np.zeros(3)

        # Ground contact parameters
        K_ground = 10e3  # Ground stiffness [N/m]
        D_ground = 100   # Ground damping [N*s/m]
        mu_friction = 0.5  # Friction coefficient
        v_threshold = 1e-3  # Velocity threshold for static friction [m/s]
        epsilon = 1e-8

        # Check each landing gear contact point
        for contact_point_body in self.contact_points_body:
            # Transform contact point to world frame
            r_contact_world = p + R @ contact_point_body
            z_contact = r_contact_world[2]

            if z_contact <= 0.0:  # Contact detected
                contact_detected = True

                # Penetration depth
                penetration = -z_contact

                # Contact point velocity (world frame)
                r_contact_body = contact_point_body  # Vector from COM to contact in body
                v_contact_world = v_world + R @ np.cross(w, r_contact_body)
                vx_c = v_contact_world[0]
                vy_c = v_contact_world[1]
                vz_c = v_contact_world[2]

                # Normal force at this contact point
                N = K_ground * penetration - D_ground * vz_c
                N = max(0.0, N)  # Only push, no pull
                f_normal = np.array([0.0, 0.0, N])

                # Friction force at this contact point
                v_horizontal = np.array([vx_c, vy_c, 0.0])
                v_h_norm = np.linalg.norm(v_horizontal)

                # Applied force (for static friction check)
                f_thrust = R @ (f * self.e3)
                f_applied = f_thrust + self.m * self.g_vec + f_normal
                f_applied_h = np.array([f_applied[0], f_applied[1], 0.0])
                f_applied_h_norm = np.linalg.norm(f_applied_h)

                if v_h_norm < v_threshold:  # Potentially static
                    f_static_max = mu_friction * N
                    if f_applied_h_norm <= f_static_max:
                        f_friction = -f_applied_h
                    else:
                        f_friction = -mu_friction * N * f_applied_h / (f_applied_h_norm + epsilon)
                else:  # Moving: kinetic friction
                    f_friction = -mu_friction * N * v_horizontal / (v_h_norm + epsilon)

                # Add viscous damping
                f_friction += -D_ground * v_horizontal

                # Total force from this contact point
                f_contact = f_normal + f_friction
                f_contact_total += f_contact

                # Moment from this contact point (r × F)
                r_contact_body_to_com = contact_point_body
                M_contact = np.cross(R @ r_contact_body_to_com, f_contact)
                M_contact_total += M_contact

        if contact_detected:
            # Check if tipped over (roll or pitch > 80 degrees)
            roll, pitch, yaw = quaternion_to_euler(q)
            tipped_over = (abs(roll) > self.tip_over_angle) or (abs(pitch) > self.tip_over_angle)

            if tipped_over:
                # Stuck condition: very strong damping
                dpdt = v_world
                dvdt = 1.0/self.m * (R@(f*self.e3) + self.m*self.g_vec + f_contact_total)

                w_quat = vec_to_quaternion_form(w)
                dqdt = 0.5*otimes(q, w_quat)

                # Extremely strong damping when tipped over
                dwdt = -500 * w
            else:
                # Normal ground contact
                dpdt = v_world
                dvdt = 1.0/self.m * (R@(f*self.e3) + self.m*self.g_vec + f_contact_total)

                w_quat = vec_to_quaternion_form(w)
                dqdt = 0.5*otimes(q, w_quat)

                # Angular dynamics with contact torque
                J_w = self.J @ w
                M_com_offset = -np.cross(self.r_off, f*self.e3)
                M_applied = M + M_com_offset + M_contact_total - np.cross(w, J_w)

                # Angular friction torque
                w_norm = np.linalg.norm(w)
                w_threshold = 0.05
                mu_angular = 0.8
                M_applied_norm = np.linalg.norm(M_applied)

                if w_norm < w_threshold:
                    M_static_max = mu_angular * np.linalg.norm(f_contact_total) * 0.15
                    if M_applied_norm <= M_static_max:
                        M_friction = -M_applied
                    else:
                        M_friction = -M_static_max * M_applied / (M_applied_norm + epsilon)
                else:
                    M_friction = -mu_angular * np.linalg.norm(f_contact_total) * 0.15 * w / (w_norm + epsilon)

                M_viscous = -200 * w
                dwdt = self.J_inv @ (M_applied + M_friction + M_viscous)

        else:
            # If not in contact with ground
            dpdt = v_world
            dvdt = 1/self.m * R @ (f*self.e3) + self.g_vec

            w_quat = vec_to_quaternion_form(w)
            dqdt = 0.5*otimes(q, w_quat)
            J_w = self.J @ w
            dwdt = self.J_inv @ (M - np.cross(w,J_w)
                             - np.cross(self.r_off, f*self.e3))

        return self.pack_state(dpdt, dvdt, dqdt, dwdt)