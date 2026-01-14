import numpy as np
from custom_ode import custom_rk4
from utils.math_tool import quaternion_to_rotm
from utils.drone_converter import HexaConverter

class HGDO:
    def __init__(self, DynParam, DroneParam, RotorParam ,DobParam):

        # Get dynamic parameter
        self.m = DynParam['m']
        J_array = DynParam['MoiArray']
        self.J = np.diag(J_array)
        self.J_inv = np.diag([1.0/self.J[0][0],
                              1.0/self.J[1][1],
                              1.0/self.J[2][2]])

        drone_param = DroneParam
        rotor_param = RotorParam

        # Get HGDO parameter
        self.eps_tau = DobParam['eps_tau']
        self.eps_f = DobParam['eps_f']

        self.gamma_trans = np.zeros((3,))
        self.gamma_rot = np.zeros((3,))

        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.v_world = np.zeros((3,))
        self.w = np.zeros((3,))

        g = -9.81
        self.g_vec = np.array([0.0, 0.0, g])

        self.hexa_converter = HexaConverter(DroneParams = drone_param,
                                            RotorParams = rotor_param)

    def dob_estimate(self, t_prev, t_curr, rotor_speed, state):
        """

        :param state:
        :param rotor_speed:
        :return:
        """
        # Unpack state from state vector
        p, v, q, w = self._unpack_state(state)

        # Update quaternion
        self.q = q

        # Update linear and angular velocity
        self.v_world = v
        self.w = w

        # Get collective thrust and moment
        u = self.hexa_converter.compute_u(rotor_speed)

        # ODE update
        t_ode = [t_prev, t_curr]

        # Solve translational gamma dynamics
        self.gamma_trans = custom_rk4.do_step(self._gamma_trans_dynamics,
                                            self.gamma_trans, u, t_ode)

        # Solve rotational gamma dynamics
        self.gamma_rot = custom_rk4.do_step(self._gamma_rot_dynamics,
                                            self.gamma_rot, u, t_ode)

        # Get translational disturbance in world frame
        f_ext_w = self.m * (self.gamma_trans + 1.0/self.eps_f*self.v_world)
        # Transform translational disturbance from world to body
        R_b_w = quaternion_to_rotm(self.q)
        f_ext = R_b_w.T @ f_ext_w

        # Get rotational disturbance in body frame
        tau_ext = self.J @ (self.gamma_rot + 1.0/self.eps_tau*self.w)

        return np.concatenate([f_ext, tau_ext])


    def _gamma_trans_dynamics(self, t, gamma_trans, u):

        f_input = self._u_to_force_vector(u)
        R_b_w = quaternion_to_rotm(self.q)

        # acceleration input in World frame
        u_T = R_b_w @ f_input / self.m

        gamma_trans_dot = (-1.0/self.eps_f*(gamma_trans + 1.0/self.eps_tau*self.v_world)
                           +1.0/self.eps_f*(-u_T - self.g_vec))

        return gamma_trans_dot

    def _gamma_rot_dynamics(self, t, gamma_rot, u):

        Jw = self.J @ self.w
        inertial_effect = self.J_inv @ np.cross(self.w, Jw)

        gamma_rot_dot = (-1.0/self.eps_tau*(gamma_rot + 1.0/self.eps_tau*self.w)
                         + 1.0/self.eps_tau*(-u[1:] + inertial_effect))
        return gamma_rot_dot

    def _unpack_state(self, state):
        """
        Unpack state from state vector
        :param state: p_world, v_body, q_b_w, w_b
        :return: p_world, v_world, q_b_w, w_b
        """
        p = state[0:3]
        v_body = state[3:6]
        q = state[6:10]
        v_world = quaternion_to_rotm(q) @ v_body
        w = state[10:13]
        return p, v_world, q, w

    def _u_to_force_vector(self, u):
        T_collective = u[0]
        return np.array([0.0, 0.0, T_collective])