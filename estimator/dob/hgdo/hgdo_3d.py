import numpy as np
from utils.custom_ode import custom_rk4
from utils.math_tool import pitch_to_rotm
from utils.drone_converter import HexaConverter

class HGDO3D:
    def __init__(self, DynamicParams, DroneParams, RotorParams, DobParams):

        # Set dynamic parameters
        self.m = DynamicParams['m']
        self.Jyy = DynamicParams['MoiArray'][1]


        drone_param = DroneParams
        rotor_param = RotorParams

        # Set HGDO parameter
        self.eps_tau = DobParams['eps_tau']
        self.eps_f = DobParams['eps_f']

        self.gamma_trans = np.zeros((2,))
        self.gamma_rot = 0.0

        self.th = 0.0
        self.v_world = np.zeros((2,))
        self.q = 0.0

        g = 9.81
        self.g_vec = np.array([0.0, -g])

        self.hexa_converter = HexaConverter(DroneParams = drone_param,
                                            RotorParams = rotor_param,
                                            Dim = 3)

    def dob_estimate(self, t_prev, t_curr, rotor_speed, state):
        """
        DOB estimate
        :param t_prev:
        :param t_curr:
        :param rotor_speed:
        :param state:
        :return:
        """
        p, v_world, th, q = self._unpack_state(state)

        # Update pitch
        self.th = th

        # Update linear and angular velocity
        self.v_world = v_world
        self.q = q

        # Get total thrust and moment
        u = self.hexa_converter.compute_u(rotor_speed)

        t_ode = [t_prev, t_curr]

        self.gamma_trans = custom_rk4.do_step(self._gamma_trans_dynamics,
                                              self.gamma_trans, u, t_ode)

        self.gamma_rot = custom_rk4.do_step(self._gamma_rot_dynamics,
                                              self.gamma_rot, u, t_ode)

        # Get translational disturbance
        f_ext_w = self.m * (self.gamma_trans + 1.0/self.eps_f*self.v_world)
        R_WB = pitch_to_rotm(th)
        f_ext = R_WB.T @ f_ext_w

        tau_ext = [self.Jyy * (self.gamma_rot + 1.0/self.eps_tau*self.q)]

        return np.concatenate([f_ext, tau_ext])


    def _gamma_trans_dynamics(self, t, gamma_trans, u):

        f_input = np.array([0.0, u[0]])
        R_WB = pitch_to_rotm(self.th)

        # Acceleration input in world frame
        u_T = R_WB @ f_input / self.m

        gamma_trans_dot = (-1.0/self.eps_f*(gamma_trans + 1.0/self.eps_f*self.v_world)
                           +1.0/self.eps_f*(-u_T - self.g_vec))

        return gamma_trans_dot

    def _gamma_rot_dynamics(self, t, gamma_rot, u):

        u_M = 1.0/self.Jyy * u[1]

        gamma_rot_dot = (-1.0/self.eps_tau*(gamma_rot + 1.0/self.eps_tau*self.q)
                     + 1.0/self.eps_tau*(-u_M))
        return gamma_rot_dot

    def _unpack_state(self, state):
        """Unpack state from state vector"""
        p = state[0:2]
        v_body = state[2:4]
        th = state[4]
        R_WB = pitch_to_rotm(th)
        v_world = R_WB.dot(v_body)
        q = state[5]
        return p, v_world, th, q