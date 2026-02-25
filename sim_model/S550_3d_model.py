"""
S550 2D model
- 3 DOF Dynamics (Position (x, z) + Attitude (theta))
- Ground contact included
"""

import numpy as np
from utils.math_tool import pitch_to_rotm
from dataclasses import dataclass

@dataclass
class S550_3D_state:
    """
    State of S550 3D simulation
    p - x, z
    v - vx, vz
    theta - pitch
    q - pitch rate
    """
    p: np.ndarray
    v: np.ndarray
    theta: float
    q: float

class S550_3D_Sim_Model:
    """
    S550_2D_Sim_Model
    """

    def __init__(self, DynamicParam):

        # Dynamics parameter
        self.m = DynamicParam['m']
        self.Jyy = DynamicParam['MoiArray'][1]
        self.x_off = DynamicParam['com_offset'][0]
        self.z_off = DynamicParam['com_offset'][2]

        # Landing gear info
        self.x_g = 0.256/2.0
        self.y_g = 0.288/2.0
        self.h_g = 0.258

        # Arm length, MOI, and initial angle for contact
        # Front landing gear contact
        self.r_f = np.sqrt((self.x_g - self.x_off)**2
                           + (self.h_g + self.z_off)**2)

        self.Jyy_f = self.Jyy + self.m * self.r_f**2

        self.delta_f = np.arctan2(self.x_g - self.x_off,
                                  self.h_g + self.z_off)

        self.r_b = np.sqrt((self.x_g + self.x_off)**2
                           + (self.h_g + self.z_off)**2)

        self.Jyy_b = self.Jyy + self.m * self.r_b**2

        self.delta_b = np.arctan2(self.x_g + self.x_off,
                                  self.h_g + self.z_off)
        # Gravity (2D: x, z)
        self.g = 9.81
        self.g_vec = np.array([0, -self.g])

        # Normal force info (No thrust)
        self.N_f = self.m * self.g / 2.0
        self.N_b = self.N_f

        # Basis
        self.e1 = np.array([1.0, 0.0])
        self.e2 = np.array([0, 1.0])

        self.is_contact = True
        self.z_liftoff_threshold = 0.005  # 5mm hysteresis for contact-flight transition

    def pack_state(self, p, v, theta, q):
        """Pack state into vector (6 dim)"""
        return np.concatenate([p, v, [theta], [q]])

    def _unpack_state(self, state):
        """
        Unpack state from vector
        p: x, z in the world frame
        v_world: vx, vz in the world frame
        theta: Pitch
        q: pitch rate
        """

        p = state[0:2]
        v_world = state[2:4]
        theta = state[4]
        q = state[5]
        return p, v_world, theta, q

    def _pack_control_input(self, f, My):
        """Pack control input into vector (2 dim)"""
        return np.concatenate([f, My])

    def _unpack_control_input(self, u):
        """Unpack control input from vector"""
        f = u[0]
        My = u[1]
        return f, My

    def dynamics(self, t, state, u):
        """
        Dynamics function of S550 2d
        :param t: time
        :param state: p, v_world, theta, q
        :param u: f, My
        :return: sdot
        """

        theta = state[4]
        q = state[5]

        sdot = self._flight_dynamics(state, u)

        if self.is_contact:
            # Check gear heights to determine contact mode
            z_front, z_rear = self._compute_gear_heights(state)
            z_contact_tol = 0.005  # 5mm tolerance for gear on ground

            # Both gears near ground → full contact
            if z_front < z_contact_tol and z_rear < z_contact_tol:
                self._compute_Normal_force(u)

                # Both normal forces positive → fully grounded
                if self.N_f > 0 and self.N_b > 0:
                    K_q_damp = 500.0
                    sdot = np.array([0.0, 0.0,
                                     0.0, 0.0,
                                     0.0, -K_q_damp * q])
                # Both negative → liftoff
                elif self.N_f <= 0.0 and self.N_b <= 0.0:
                    sdot = self._flight_dynamics(state, u)
                    self.is_contact = False
                # Front only → front contact
                elif self.N_f > 0 and self.N_b <= 0.0:
                    sdot = self._front_dynamics(state, u)
                # Rear only → rear contact
                elif self.N_b > 0 and self.N_f <= 0.0:
                    sdot = self._rear_dynamics(state, u)

            # Only front gear on ground (theta > 0 → nose up)
            elif z_front < z_contact_tol:
                sdot = self._front_dynamics(state, u)

            # Only rear gear on ground (theta < 0 → nose down)
            elif z_rear < z_contact_tol:
                sdot = self._rear_dynamics(state, u)

            # Both gears above ground → flight
            else:
                sdot = self._flight_dynamics(state, u)
                self.is_contact = False

        else:
            # Flight mode - check for landing
            z_front, z_rear = self._compute_gear_heights(state)
            vz = state[3]

            if (z_front <= 0.0 or z_rear <= 0.0) and vz <= 0.0:
                # Landing detected
                self.is_contact = True
                print(f'Landing detected: z_front={z_front:.4f}, z_rear={z_rear:.4f}, '
                      f'vz={vz:.4f}, theta={np.rad2deg(theta):.2f}deg')

                if np.abs(theta) <= 5.0 * np.pi / 180.0:
                    # Near-level landing → full contact
                    K_q_damp = 500.0
                    sdot = np.array([0.0, 0.0,
                                     0.0, 0.0,
                                     0.0, -K_q_damp * q])
                elif theta > 0.0:
                    sdot = self._rear_dynamics(state, u)
                else:
                    sdot = self._front_dynamics(state, u)
            else:
                sdot = self._flight_dynamics(state, u)

        return sdot


    def _compute_Normal_force(self, u):

        # Unpack control input
        f, My = self._unpack_control_input(u)

        # Front normal force
        self.N_f = 0.5*(self.m * self.g * (1.0 + self.x_off/self.x_g)
                    - f + My/self.x_g)

        # Rear normal force
        self.N_b = 0.5*(self.m * self.g * (1.0 - self.x_off/self.x_g)
                    - f - My/self.x_g)

    def _compute_gear_heights(self, state):
        """Compute world-frame z-coordinates of front and rear landing gear.

        Front gear is at body-frame offset (x_g/2 - x_off, -(h_g + z_off)) from CM.
        Rear gear is at body-frame offset (-(x_g/2 + x_off), -(h_g + z_off)) from CM.
        """
        p = state[0:2]
        theta = state[4]

        sth = np.sin(theta)
        cth = np.cos(theta)

        # Front gear z in world frame
        z_front = (p[1]
                   - sth * (self.x_g - self.x_off)
                   - cth * (self.h_g + self.z_off))

        # Rear gear z in world frame
        z_rear = (p[1]
                  + sth * (self.x_g + self.x_off)
                  - cth * (self.h_g + self.z_off))

        return z_front, z_rear

    def _flight_dynamics(self, state, u):
        """
        No contact condition --> Flight!
        """
        # Unpack state and control input
        p, v_world, theta, q = self._unpack_state(state)
        f, My = self._unpack_control_input(u)
        R = pitch_to_rotm(theta)

        dpdt = v_world
        dvdt = f/self.m*R@self.e2 + self.g_vec
        dthdt = q
        dqdt = 1/self.Jyy*(My + self.x_off*f)

        return self.pack_state(dpdt, dvdt, dthdt, dqdt)

    def _front_dynamics(self, state, u):
        # Unpack state and control input

        p, v_world, theta, q = self._unpack_state(state)
        f, My = self._unpack_control_input(u)
        R = pitch_to_rotm(theta)

        W_f = f * R @ self.e2
        fx = W_f[0]
        fz = W_f[1]

        self.N_f = self.m * self.g - fz

        # Flight condition: normal force negative AND gear above threshold
        z_front, _ = self._compute_gear_heights(state)
        if self.N_f <= 0.0 and z_front > self.z_liftoff_threshold:
            sdot = self._flight_dynamics(state, u)
            self.is_contact = False
        # Front contact condition
        else:
            # Static friction assumption
            f_fric = -fx
            cth = np.cos(theta)
            sth = np.sin(theta)
            M_contact = self.m * self.g *( (self.h_g + self.z_off) * sth
                                           -(self.x_g - self.x_off) * cth )

            dthdt = q
            dqdt = 1.0/self.Jyy_f * (My + self.x_g*f + M_contact)

            dpdt = self.r_f * q * np.array([np.cos(theta - self.delta_f),
                                            -np.sin(theta - self.delta_f)])
            dvdt = (self.r_f * dqdt * np.array([np.cos(theta-self.delta_f),
                                               -np.sin(theta-self.delta_f)])
                    +self.r_f * (q**2) * np.array([-np.sin(theta-self.delta_f),
                                                   -np.cos(theta-self.delta_f)]))
            sdot = self.pack_state(dpdt, dvdt, dthdt, dqdt)
        return sdot

    def _rear_dynamics(self, state, u):
        # Unpack state and control input
        p, v_world, theta, q = self._unpack_state(state)
        f, My = self._unpack_control_input(u)
        R = pitch_to_rotm(theta)

        W_f = f * R @ self.e2
        fx = W_f[0]
        fz = W_f[1]

        self.N_b = self.m * self.g - fz

        # Flight condition: normal force negative AND gear above threshold
        _, z_rear = self._compute_gear_heights(state)
        if self.N_b <= 0.0 and z_rear > self.z_liftoff_threshold:
            sdot = self._flight_dynamics(state, u)
            self.is_contact = False
        # Rear contact condition
        else:
            # Static friction assumption
            f_fric = -fx
            cth = np.cos(theta)
            sth = np.sin(theta)
            M_contact = self.m * self.g * ((self.h_g + self.z_off) * sth
                                           + (self.x_g + self.x_off) * cth)

            dthdt = q
            dqdt = 1.0 / self.Jyy_b * (My - self.x_g*f + M_contact)

            dpdt = self.r_b * q * np.array([-np.cos(theta + self.delta_b),
                                            np.sin(theta + self.delta_b)])
            dvdt = (self.r_b * dqdt * np.array([-np.cos(theta + self.delta_b),
                                                np.sin(theta + self.delta_b)])
                    + self.r_b * (q ** 2) * np.array([-np.sin(theta + self.delta_b),
                                                      -np.cos(theta + self.delta_b)]))

            sdot = self.pack_state(dpdt, dvdt, dthdt, dqdt)
        return sdot

    def get_state(self, state):
        """Coordinate transformation from CM to Body"""
        W_p_CM = state[0:2]
        W_v_CM = state[2:4]
        theta = state[4]
        q = state[5]

        R_Omega_x = q * np.array([[-np.sin(theta), np.cos(theta)],
                                  [-np.cos(theta), -np.sin(theta)]])

        R = pitch_to_rotm(theta)
        B_p_offset = np.array([self.x_off, self.z_off])
        W_p_offset = R @ B_p_offset
        W_p_B = W_p_CM - W_p_offset - np.array([0.0, self.h_g])
        W_v_B = W_v_CM - R_Omega_x @ B_p_offset
        B_v_B = R.T @ W_v_B

        return np.concatenate([W_p_B, B_v_B, [theta], [q]])

    def clamp_ground(self, state):
        """Clamp state to prevent ground penetration after ODE step.

        When in contact mode, ensures no landing gear goes below z=0.
        Adjusts CM height and zeros velocity components as needed.
        """
        z_front, z_rear = self._compute_gear_heights(state)
        min_z_gear = min(z_front, z_rear)

        if not self.is_contact:
            return state

        # In contact mode: prevent penetration
        if min_z_gear < 0.0:
            state[1] -= min_z_gear

        # If both gears are near ground, enforce full ground contact
        z_front, z_rear = self._compute_gear_heights(state)
        theta = state[4]
        # Height difference from pitch: x_g * sin(theta)
        h_diff = self.x_g * np.abs(np.sin(theta))
        if max(z_front, z_rear) < h_diff + 0.005:
            # Both gears effectively on ground
            # Zero out translational velocity and pitch rate
            state[2] = 0.0  # vx = 0
            state[3] = 0.0  # vz = 0
            state[4] = 0.0  # theta = 0
            state[5] = 0.0  # q = 0
            # Reset CM height for zero-pitch contact
            state[1] = self.h_g + self.z_off

        return state