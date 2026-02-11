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
        self.x_g = 0.256
        self.y_g = 0.288
        self.h_g = 0.258

        # Arm length, MOI, and initial angle for contact
        # Front landing gear contact
        self.r_f = np.sqrt((self.x_g/2.0 - self.x_off)**2
                           + (self.h_g + self.z_off)**2)

        self.Jyy_f = self.Jyy + self.m * self.r_f**2

        self.delta_f = np.arctan2(self.x_g/2.0 - self.x_off,
                                  self.h_g + self.z_off)

        self.r_b = np.sqrt((self.x_g/2.0 + self.x_off)**2
                           + (self.h_g + self.z_off)**2)

        self.Jyy_b = self.Jyy + self.m * self.r_b**2

        self.delta_b = np.arctan2(self.x_g/2.0 + self.x_off,
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

        sdot = self._flight_dynamics(state, u)

        # if self.is_contact == True:
        #     print('N_f: ', self.N_f, 'N_b: ', self.N_b)

        # Check Near zero pitch condition
        if np.abs(theta) <= 0.1*np.pi/180.0 and self.is_contact == True:
            # Compute normal force at near zero pitch
            self._compute_Normal_force(u)

            # Flight condition
            if self.N_f <= 0.0 and self.N_b <= 0.0:
                sdot = self._flight_dynamics(state, u)
                self.is_contact = False
            # Front contact condition
            elif self.N_b <= 0.0 and self.N_f > 0:
                sdot = self._front_dynamics(state, u)
            # Rear contact condition
            elif self.N_f <= 0.0 and self.N_b > 0:
                sdot = self._rear_dynamics(state, u)

            elif self.N_f > 0 and self.N_b > 0:
                sdot = np.array([0.0, 0.0,
                                 0.0, 0.0,
                                 0.0, 0.0])

        elif np.abs(theta) > 0.1*np.pi/180.0 and self.is_contact == True:
            if theta > 0.0:
                sdot = self._front_dynamics(state, u)
            elif theta < 0.0:
                sdot = self._rear_dynamics(state, u)

        elif self.is_contact == False:
            sdot = self._flight_dynamics(state, u)
            # print('No contact')

        return sdot


    def _compute_Normal_force(self, u):

        # Unpack control input
        f, My = self._unpack_control_input(u)

        # Front normal force
        self.N_f = (self.m * self.g * (0.5 + self.x_off/self.x_g)
                    - 0.5 * f + My/self.x_g)

        # Rear normal force
        self.N_b = (self.m * self.g * (0.5 - self.x_off/self.x_g)
                    - 0.5 * f - My/self.x_g)

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
        print('Front dynamics')
        # Unpack state and control input

        p, v_world, theta, q = self._unpack_state(state)
        f, My = self._unpack_control_input(u)
        R = pitch_to_rotm(theta)

        W_f = f * R @ self.e2
        fx = W_f[0]
        fz = W_f[1]

        self.N_f = self.m * self.g - fz

        # Flight condition
        if self.N_f <= 0.0:
            sdot = self._flight_dynamics(state, u)
            self.is_contact = False
        # Front contact condition
        else:
            # Static friction assumption
            f_fric = -fx
            cth = np.cos(theta)
            sth = np.sin(theta)
            M_contact = ((-self.N_f*cth + f_fric*sth)*(self.x_g/2.0 - self.x_off)
                         +(self.N_f*sth + f_fric*cth)*(self.h_g + self.z_off))

            dthdt = q
            dqdt = 1.0/self.Jyy_f * (My + self.x_off*f + M_contact)

            dpdt = self.r_f * q * np.array([np.cos(theta - self.delta_f),
                                            -np.sin(theta - self.delta_f)])
            dvdt = (self.r_f * dqdt * np.array([np.cos(theta-self.delta_f),
                                               -np.sin(theta-self.delta_f)])
                    +self.r_f * (q**2) * np.array([-np.sin(theta-self.delta_f),
                                                   -np.cos(theta-self.delta_f)]))
            sdot = self.pack_state(dpdt, dvdt, dthdt, dqdt)
        return sdot

    def _rear_dynamics(self, state, u):
        print('Rear dynamics')
        # Unpack state and control input
        p, v_world, theta, q = self._unpack_state(state)
        f, My = self._unpack_control_input(u)
        R = pitch_to_rotm(theta)

        W_f = f * R @ self.e2
        fx = W_f[0]
        fz = W_f[1]

        self.N_b = self.m * self.g - fz

        # Flight condition
        if self.N_b <= 0.0:
            sdot = self._flight_dynamics(state, u)
            self.is_contact = False
        # Rear contact condition
        else:
            # Static friction assumption
            f_fric = -fx
            cth = np.cos(theta)
            sth = np.sin(theta)
            M_contact = ((-self.N_b * cth + f_fric * sth) * (self.x_g / 2.0 + self.x_off)
                         + (self.N_b * sth - f_fric * cth) * (self.h_g + self.z_off))

            dthdt = q
            dqdt = 1.0 / self.Jyy_b * (My + self.x_off*f + M_contact)

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