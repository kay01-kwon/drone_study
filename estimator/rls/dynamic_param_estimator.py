import numpy as np
from utils.math_tool import quaternion_to_rotm
from utils.drone_converter import HexaConverter

class DynamicParamEstimator:
    def __init__(self, DynParam, DroneParam, RotorParam, RlsParam):

        # Set initial state
        self.m_est = DynParam['m']
        self.m_nom = DynParam['m']
        self.r_offset_est = np.zeros(2)

        # Set initial variance and signal noise variance for mass
        self.P_m = RlsParam['P_m']
        self.R_m = RlsParam['R_m']

        # Forgetting factor for mass (optional, default = 1.0 for no forgetting)
        self.lambda_m = RlsParam.get('lambda_m', 1.0)

        self.g_mag = 9.81

        # Set initial variance and signal noise variance for com offset
        self.P_com_x = RlsParam['P_com']
        self.P_com_y = RlsParam['P_com']
        self.R_com = RlsParam['R_com']

        # Forgetting factor for COM (optional, default = 1.0 for no forgetting)
        self.lambda_com = RlsParam.get('lambda_com', 1.0)

        self.hexa_converter = HexaConverter(DroneParams=DroneParam,
                                            RotorParams=RotorParam)

    def update(self, state, disturbance_estimate, rotor_speed):

        # Get altitude
        z = state[2]

        # Update mass
        q = state[6:10]
        R_b_w = quaternion_to_rotm(q)
        f_ext, tau_ext = self._unpack_disturbance_estimate(disturbance_estimate)
        f_ext_world = R_b_w @ f_ext
        m_signal = - (f_ext_world[2] - self.m_nom * self.g_mag)/self.g_mag

        if z >= 0.1:
            # RLS with forgetting factor for mass
            K_m = self.P_m / (self.P_m + self.R_m)
            self.m_est = self.m_est + K_m*(m_signal - self.m_est)
            self.P_m = (1 - K_m) * self.P_m / self.lambda_m  # Forgetting factor applied

        # Update r_com_offset
        u = self.hexa_converter.compute_u(rotor_speed)
        f_col = u[0]
        r_signal = np.array([tau_ext[1]/f_col,
                             -tau_ext[0]/f_col])

        # RLS with forgetting factor for COM offset
        K_com_x = self.P_com_x / (self.P_com_x + self.R_com)
        K_com_y = self.P_com_y / (self.P_com_y + self.R_com)

        self.r_offset_est[0] = (self.r_offset_est[0]
                                + K_com_x*(r_signal[0] - self.r_offset_est[0]))
        self.r_offset_est[1] = (self.r_offset_est[1]
                                + K_com_y*(r_signal[1] - self.r_offset_est[1]))

        # Forgetting factor applied
        self.P_com_x = (1 - K_com_x) * self.P_com_x / self.lambda_com
        self.P_com_y = (1 - K_com_y) * self.P_com_y / self.lambda_com

    def get_mass_estimate(self):
        return self.m_est

    def get_mass_variance(self):
        return self.P_m

    def get_com_estimate(self):
        return self.r_offset_est

    def get_com_variance(self):
        return np.array([self.P_com_x, self.P_com_y])

    def get_parameter_estimate(self):
        return np.array([self.m_est, self.r_offset_est[0], self.r_offset_est[1]])

    def get_parameter_variance(self):
        return np.array([self.P_m, self.P_com_x, self.P_com_y])

    def _unpack_disturbance_estimate(self, disturbance_estimate):
        f_ext = disturbance_estimate[0:3]
        tau_ext = disturbance_estimate[3:6]
        return f_ext, tau_ext