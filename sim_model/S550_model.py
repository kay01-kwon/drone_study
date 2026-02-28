"""
S550 simulation model
- Full 6 DOF Dynamics (Position + Attitude)
- Quaternion-based attitude dynamics
- COM offset included
- Front/Rear landing gear contact dynamics (like S550_3d_model)
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
    S550_Sim_Model with front/rear landing gear contact dynamics
    :param : DynamicParams
    """

    def __init__(self, DynamicParams):

        # Dynamics parameter
        self.m = DynamicParams['m']
        Jxx = DynamicParams['MoiArray'][0]
        Jyy = DynamicParams['MoiArray'][1]
        Jzz = DynamicParams['MoiArray'][2]
        self.Jyy = Jyy
        self.J = np.diag([Jxx, Jyy, Jzz])
        self.J_inv = np.diag([1/Jxx, 1/Jyy, 1/Jzz])

        # COM offset [x_off, y_off, z_off]
        self.x_off = DynamicParams['com_offset'][0]
        self.y_off = DynamicParams['com_offset'][1]
        self.z_off = DynamicParams['com_offset'][2]
        self.r_off = np.array(DynamicParams['com_offset'])

        # Gravity
        self.g = 9.81
        self.g_vec = np.array([0, 0, -self.g])

        self.e3 = np.array([0, 0, 1.0])

        # Landing gear geometry (matching S550_3d_model)
        self.x_g = 0.256 / 2.0   # Half of front-rear distance [m]
        self.y_g = 0.288 / 2.0   # Half of left-right distance [m]
        self.h_g = 0.258         # Landing gear height [m]

        # Front landing gear contact parameters
        self.r_f = np.sqrt((self.x_g - self.x_off)**2
                           + (self.h_g + self.z_off)**2)
        self.Jyy_f = Jyy + self.m * self.r_f**2
        self.delta_f = np.arctan2(self.x_g - self.x_off,
                                  self.h_g + self.z_off)

        # Rear landing gear contact parameters
        self.r_b = np.sqrt((self.x_g + self.x_off)**2
                           + (self.h_g + self.z_off)**2)
        self.Jyy_b = Jyy + self.m * self.r_b**2
        self.delta_b = np.arctan2(self.x_g + self.x_off,
                                  self.h_g + self.z_off)

        # Normal force (no thrust)
        self.N_f = self.m * self.g / 2.0
        self.N_b = self.N_f

        # Contact state
        self.is_contact = True
        self.z_liftoff_threshold = 0.005  # 5mm hysteresis

        # Tip over angle limit
        self.tip_over_angle = np.deg2rad(80)

        # Initial height offset for coordinate transformation (like PX4)
        self.z_offset_initial = 0.0

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
        R = quaternion_to_rotm(q)
        v_body = R @ v_world
        w = s[10:13]
        return p, v_body, q, w

    def _unpack_state_world(self, state):
        """Unpack state from world frame"""
        p = state[0:3]
        v_world = state[3:6]
        q = state[6:10]
        w = state[10:13]
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
        Dynamics function of S550 with front/rear landing gear contact
        :param t: time (Not use, but required for rk4 interface)
        :param s: state vector [13]
        :param u: control input - Thrust [1], Moment [3]
        :return: dsdt state derivative [13]
        """

        # Unpack state and control input
        p, v_world, q, w = self._unpack_state_world(s)
        # Normalize quaternion
        q = q / np.linalg.norm(q, 2)
        f, M = self.unpack_control_input(u)

        # Get Euler angles for contact detection
        roll, pitch, yaw = quaternion_to_euler(q)

        sdot = self._flight_dynamics(s, u)

        if self.is_contact:
            # Check gear heights to determine contact mode
            z_front, z_rear = self._compute_gear_heights(s)
            z_contact_tol = 0.005  # 5mm tolerance

            # Both gears near ground → full contact
            if z_front < z_contact_tol and z_rear < z_contact_tol:
                self._compute_Normal_force(u, pitch)

                # Both normal forces positive → fully grounded
                if self.N_f > 0 and self.N_b > 0:
                    K_w_damp = 500.0
                    # Damp all angular velocities, keep position/velocity at rest
                    dpdt = np.zeros(3)
                    dvdt = np.zeros(3)
                    w_quat = vec_to_quaternion_form(w)
                    dqdt = 0.5 * otimes(q, w_quat)
                    dwdt = -K_w_damp * w
                    sdot = self.pack_state(dpdt, dvdt, dqdt, dwdt)
                # Both negative → liftoff
                elif self.N_f <= 0.0 and self.N_b <= 0.0:
                    sdot = self._flight_dynamics(s, u)
                    self.is_contact = False
                # Front only → front contact
                elif self.N_f > 0 and self.N_b <= 0.0:
                    sdot = self._front_dynamics(s, u)
                # Rear only → rear contact
                elif self.N_b > 0 and self.N_f <= 0.0:
                    sdot = self._rear_dynamics(s, u)

            # Only front gear on ground (pitch > 0 → nose up)
            elif z_front < z_contact_tol:
                sdot = self._front_dynamics(s, u)

            # Only rear gear on ground (pitch < 0 → nose down)
            elif z_rear < z_contact_tol:
                sdot = self._rear_dynamics(s, u)

            # Both gears above ground → flight
            else:
                sdot = self._flight_dynamics(s, u)
                self.is_contact = False

        else:
            # Flight mode - check for landing
            z_front, z_rear = self._compute_gear_heights(s)
            vz = v_world[2]

            if (z_front <= 0.0 or z_rear <= 0.0) and vz <= 0.0:
                # Landing detected
                self.is_contact = True
                print(f'Landing detected: z_front={z_front:.4f}, z_rear={z_rear:.4f}, '
                      f'vz={vz:.4f}, pitch={np.rad2deg(pitch):.2f}deg')

                if np.abs(pitch) <= 5.0 * np.pi / 180.0:
                    # Near-level landing → full contact
                    K_w_damp = 500.0
                    dpdt = np.zeros(3)
                    dvdt = np.zeros(3)
                    w_quat = vec_to_quaternion_form(w)
                    dqdt = 0.5 * otimes(q, w_quat)
                    dwdt = -K_w_damp * w
                    sdot = self.pack_state(dpdt, dvdt, dqdt, dwdt)
                elif pitch > 0.0:
                    sdot = self._rear_dynamics(s, u)
                else:
                    sdot = self._front_dynamics(s, u)
            else:
                sdot = self._flight_dynamics(s, u)

        return sdot

    def _compute_Normal_force(self, u, pitch):
        """Compute normal forces on front and rear landing gear."""
        f, M = self.unpack_control_input(u)
        My = M[1]  # Pitch moment

        # Front normal force
        self.N_f = 0.5 * (self.m * self.g * (1.0 + self.x_off / self.x_g)
                         - f + My / self.x_g)

        # Rear normal force
        self.N_b = 0.5 * (self.m * self.g * (1.0 - self.x_off / self.x_g)
                         - f - My / self.x_g)

    def _compute_gear_heights(self, state):
        """Compute world-frame z-coordinates of front and rear landing gear."""
        p = state[0:3]
        q = state[6:10]
        q = q / np.linalg.norm(q, 2)
        roll, pitch, yaw = quaternion_to_euler(q)

        sth = np.sin(pitch)
        cth = np.cos(pitch)

        # Front gear z in world frame (simplified: mainly pitch dependent)
        z_front = (p[2]
                   - sth * (self.x_g - self.x_off)
                   - cth * (self.h_g + self.z_off))

        # Rear gear z in world frame
        z_rear = (p[2]
                  + sth * (self.x_g + self.x_off)
                  - cth * (self.h_g + self.z_off))

        return z_front, z_rear

    def _flight_dynamics(self, state, u):
        """Flight dynamics (no contact)."""
        p, v_world, q, w = self._unpack_state_world(state)
        q = q / np.linalg.norm(q, 2)
        f, M = self.unpack_control_input(u)
        R = quaternion_to_rotm(q)

        dpdt = v_world
        dvdt = 1 / self.m * (R @ (f * self.e3)) + self.g_vec

        w_quat = vec_to_quaternion_form(w)
        dqdt = 0.5 * otimes(q, w_quat)
        J_w = self.J @ w
        # Include COM offset coupling
        dwdt = self.J_inv @ (M - np.cross(w, J_w)
                             - np.cross(self.r_off, f * self.e3))

        return self.pack_state(dpdt, dvdt, dqdt, dwdt)

    def _front_dynamics(self, state, u):
        """Front landing gear contact dynamics."""
        p, v_world, q, w = self._unpack_state_world(state)
        q = q / np.linalg.norm(q, 2)
        f, M = self.unpack_control_input(u)
        R = quaternion_to_rotm(q)
        roll, pitch, yaw = quaternion_to_euler(q)

        W_f = R @ (f * self.e3)
        fz = W_f[2]

        self.N_f = self.m * self.g - fz

        # Flight condition check
        z_front, _ = self._compute_gear_heights(state)
        if self.N_f <= 0.0 and z_front > self.z_liftoff_threshold:
            sdot = self._flight_dynamics(state, u)
            self.is_contact = False
        else:
            cth = np.cos(pitch)
            sth = np.sin(pitch)
            M_contact = self.m * self.g * ((self.h_g + self.z_off) * sth
                                           - (self.x_g - self.x_off) * cth)
            My = M[1]  # Pitch moment

            # Pitch dynamics (primary rotation in contact)
            dqdt_pitch = w[1]
            dwdt_pitch = 1.0 / self.Jyy_f * (My + self.x_g * f + M_contact)

            # Position follows rotation around front gear
            dpdt_x = self.r_f * w[1] * np.cos(pitch - self.delta_f)
            dpdt_z = -self.r_f * w[1] * np.sin(pitch - self.delta_f)
            dpdt = np.array([dpdt_x, 0.0, dpdt_z])

            dvdt_x = (self.r_f * dwdt_pitch * np.cos(pitch - self.delta_f)
                      + self.r_f * (w[1] ** 2) * (-np.sin(pitch - self.delta_f)))
            dvdt_z = (self.r_f * dwdt_pitch * (-np.sin(pitch - self.delta_f))
                      + self.r_f * (w[1] ** 2) * (-np.cos(pitch - self.delta_f)))
            dvdt = np.array([dvdt_x, 0.0, dvdt_z])

            # Quaternion update (mainly pitch change)
            w_quat = vec_to_quaternion_form(w)
            dqdt = 0.5 * otimes(q, w_quat)

            # Angular velocity: only pitch rate changes, others damped
            K_damp = 100.0
            dwdt = np.array([-K_damp * w[0], dwdt_pitch, -K_damp * w[2]])

            sdot = self.pack_state(dpdt, dvdt, dqdt, dwdt)

        return sdot

    def _rear_dynamics(self, state, u):
        """Rear landing gear contact dynamics."""
        p, v_world, q, w = self._unpack_state_world(state)
        q = q / np.linalg.norm(q, 2)
        f, M = self.unpack_control_input(u)
        R = quaternion_to_rotm(q)
        roll, pitch, yaw = quaternion_to_euler(q)

        W_f = R @ (f * self.e3)
        fz = W_f[2]

        self.N_b = self.m * self.g - fz

        # Flight condition check
        _, z_rear = self._compute_gear_heights(state)
        if self.N_b <= 0.0 and z_rear > self.z_liftoff_threshold:
            sdot = self._flight_dynamics(state, u)
            self.is_contact = False
        else:
            cth = np.cos(pitch)
            sth = np.sin(pitch)
            M_contact = self.m * self.g * ((self.h_g + self.z_off) * sth
                                           + (self.x_g + self.x_off) * cth)
            My = M[1]  # Pitch moment

            # Pitch dynamics
            dqdt_pitch = w[1]
            dwdt_pitch = 1.0 / self.Jyy_b * (My - self.x_g * f + M_contact)

            # Position follows rotation around rear gear
            dpdt_x = -self.r_b * w[1] * np.cos(pitch + self.delta_b)
            dpdt_z = self.r_b * w[1] * np.sin(pitch + self.delta_b)
            dpdt = np.array([dpdt_x, 0.0, dpdt_z])

            dvdt_x = (-self.r_b * dwdt_pitch * np.cos(pitch + self.delta_b)
                      + self.r_b * (w[1] ** 2) * (-np.sin(pitch + self.delta_b)))
            dvdt_z = (self.r_b * dwdt_pitch * np.sin(pitch + self.delta_b)
                      + self.r_b * (w[1] ** 2) * (-np.cos(pitch + self.delta_b)))
            dvdt = np.array([dvdt_x, 0.0, dvdt_z])

            # Quaternion update
            w_quat = vec_to_quaternion_form(w)
            dqdt = 0.5 * otimes(q, w_quat)

            # Angular velocity: only pitch rate changes, others damped
            K_damp = 100.0
            dwdt = np.array([-K_damp * w[0], dwdt_pitch, -K_damp * w[2]])

            sdot = self.pack_state(dpdt, dvdt, dqdt, dwdt)

        return sdot

    def get_state(self, state, z_offset=0.0):
        """
        Get state with coordinate transformation from CM to Body origin.
        Like mavros/local_position/odom:
        - Position: World frame (Body origin, not CM)
        - Velocity: Body frame

        Coordinate transformation (from lecture slide):
        Position: ^W p_{W→B} = ^W p_{W→CM} - R_B^W · ^B p_{B→CM}
        Velocity: ^B v = (R_B^W)^T · ^W v

        Args:
            state: [p_cm(3), v_world(3), q(4), w(3)] - 13 dim
            z_offset: Initial height offset to subtract (like PX4 hg offset)

        Returns:
            state_out: [p_body_origin(3), v_body(3), q(4), w(3)] - 13 dim
                       Position in World frame (Body origin)
                       Velocity in Body frame
        """
        # Unpack state (CM frame)
        p_cm = state[0:3].copy()
        v_world = state[3:6].copy()
        q = state[6:10].copy()
        w = state[10:13].copy()

        # Normalize quaternion
        q = q / np.linalg.norm(q)

        # Rotation matrix: Body to World
        R_B_W = quaternion_to_rotm(q)

        # Position transformation: CM to Body origin (World frame)
        # ^W p_{W→B} = ^W p_{W→CM} - R_B^W · ^B p_{B→CM}
        p_body_origin = p_cm - R_B_W @ self.r_off

        # Apply height offset (like PX4 subtracts initial hg)
        p_body_origin[2] -= z_offset

        # Velocity transformation: World to Body frame
        # ^B v = (R_B^W)^T · ^W v = R_W^B · ^W v
        v_body = R_B_W.T @ v_world

        # Pack output state
        state_out = np.zeros(13)
        state_out[0:3] = p_body_origin  # Position in World frame (Body origin)
        state_out[3:6] = v_body         # Velocity in Body frame
        state_out[6:10] = q             # Quaternion (Body to World)
        state_out[10:13] = w            # Angular velocity in Body frame

        return state_out

    def set_initial_height_offset(self, state):
        """
        Set initial height offset from current state.
        Like PX4/Pixhawk EKF2 that subtracts initial hg from mocap rigid body.

        Call this once at initialization after drone is on ground.

        Args:
            state: Initial state [p_cm(3), v_world(3), q(4), w(3)]
        """
        # Get body origin position at initial state
        q = state[6:10]
        q = q / np.linalg.norm(q)
        R_B_W = quaternion_to_rotm(q)

        # Body origin height = CM height - rotated z_offset
        p_cm = state[0:3]
        p_body_origin = p_cm - R_B_W @ self.r_off

        # Store initial height as offset
        self.z_offset_initial = p_body_origin[2]

    def get_transformed_state(self, state):
        """
        Convenience method: get state with stored initial height offset.

        Returns:
            state_out: Position (World, Body origin), Velocity (Body frame)
        """
        return self.get_state(state, self.z_offset_initial)

    def get_landing_gear_height(self):
        """Get landing gear height (hg) for reference."""
        return self.h_g

    def clamp_ground(self, state):
        """Clamp state to prevent ground penetration after ODE step."""
        z_front, z_rear = self._compute_gear_heights(state)
        min_z_gear = min(z_front, z_rear)

        if not self.is_contact:
            return state

        # In contact mode: prevent penetration
        if min_z_gear < 0.0:
            state[2] -= min_z_gear

        # If both gears are near ground, enforce full ground contact
        z_front, z_rear = self._compute_gear_heights(state)
        q = state[6:10]
        q = q / np.linalg.norm(q, 2)
        roll, pitch, yaw = quaternion_to_euler(q)

        h_diff = self.x_g * np.abs(np.sin(pitch))
        if max(z_front, z_rear) < h_diff + 0.005:
            # Both gears effectively on ground
            # Zero out velocities
            state[3:6] = 0.0  # v = 0
            state[10:13] = 0.0  # w = 0
            # Reset to level attitude (keep yaw)
            state[6:10] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            # Reset height for level contact
            state[2] = self.h_g + self.z_off

        return state