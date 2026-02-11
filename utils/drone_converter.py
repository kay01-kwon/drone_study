"""
Drone converter

- Support rotor_speed to u (T, M)
- Support u to rotor_speed
"""
import numpy as np


class HexaConverter:
    def __init__(self, DroneParams, RotorParams, Dim = 6):

        # Get drone parameters
        self.l = DroneParams['arm_length']
        self.C_T = DroneParams['motor_const']
        k_m = DroneParams['moment_const']

        # Get rotor speed constraints
        w_min = RotorParams['w_rotor_min']
        w_max = RotorParams['w_rotor_max']

        # Convert from rotor speed constraints into rotor thrust
        self.T_min = self.C_T * w_min**2
        self.T_max = self.C_T * w_max**2

        if Dim == 6:
            # Thrust [6] to u (f, M) [4]
            self.Kf = np.zeros((4,6))
            self._precompute_rotor_pos(k_m, Dim)

            # u [4] to rotor thrusts [6]
            self.Kinv = np.zeros((6,4))
            self.Kinv = np.linalg.pinv(self.Kf)
        elif Dim == 3:
            # Thrust [3] to u (f, My) [2]
            self.Kf = np.zeros((2, 3))
            self._precompute_rotor_pos(k_m, Dim)

            self.Kinv = np.zeros((3,2))
            self.Kinv = np.linalg.inv(self.Kf)


    def _precompute_rotor_pos(self, k_m, Dim = 6):
        cos_pi_3 = np.cos(np.pi/3)
        sin_pi_3 = np.sin(np.pi/3)
        if Dim == 6:
            y1 = self.l*cos_pi_3
            y2 = self.l
            y3 = self.l*cos_pi_3

            y4 = -self.l*cos_pi_3
            y5 = -self.l
            y6 = -self.l*cos_pi_3

            x1 = self.l*sin_pi_3
            x2 = 0
            x3 = -self.l*sin_pi_3

            x4 = -self.l*sin_pi_3
            x5 = 0
            x6 = self.l*sin_pi_3

            self.Kf = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [y1, y2, y3, y4, y5, y6],
            [-x1, -x2, -x3, -x4, -x5, -x6],
            [-k_m, k_m, -k_m, k_m, -k_m, k_m],
            ])
        elif Dim == 3:
            self.Kf = np.array([
                [2.0, 2.0, 2.0],
                [-2*self.l*sin_pi_3, 0.0, 2*self.l*sin_pi_3]
            ])

    def compute_u(self, w_cmd):
        """
        Compute u
        :param w_cmd: rotor speed
        :return: u [f, M]
        """
        w_cmd_sqrd = w_cmd**2
        T_rotor = self.C_T*w_cmd_sqrd
        u = self.Kf @ T_rotor
        return u

    def compute_des_rotor_speed(self, u):
        """
        Compute desired rotor speed
        :param u: u [f, M]
        :return: des_rotors_speed [6]
        """
        # Get rotor thrusts
        T_rotor = self.Kinv @ u
        T_rotor_clamped = self._clamp_thrusts(T_rotor)
        des_rotors_speed = np.sqrt(T_rotor_clamped/self.C_T)
        return des_rotors_speed

    def _clamp_thrusts(self, T_rotor):
        T_rotor_clamped = np.clip(T_rotor, self.T_min, self.T_max)
        return T_rotor_clamped

