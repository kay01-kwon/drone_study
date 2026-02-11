import numpy as np
from utils.math_tool import pitch_to_rotm

class PitchControl:
    def __init__(self, DynamicParams, GainParams, DobMode = False):
        """

        :param DynamicParams:
        :param GainParams:
        :param DobMode:
        """
        # Load dynamics parameter
        self.m = DynamicParams['m']
        self.Jyy = DynamicParams['MoiArray'][1]

        g = 9.81
        self.g_vec = np.array([0.0, 0.0, -g])

        self.DobMode = DobMode

        # Position gain load
        Kp_x = GainParams['KpPosArray'][0]
        Kp_z = GainParams['KpPosArray'][2]
        Kp_array = np.array([Kp_x, Kp_z])
        self.Kp_pos = np.diag(Kp_array)

        if self.DobMode is False:
            Ki_x = GainParams['KiPosArray'][0]
            Ki_z = GainParams['KiPosArray'][2]
            Ki_array = np.array([Ki_x, Ki_z])
            self.Ki_pos = np.diag(Ki_array)
            self.I_trans_limit = np.array([GainParams['ITransLimit'][0],
                                           GainParams['ITransLimit'][1]])
            self.I_trans = np.zeros((2,))

        Kd_x = GainParams['KdPosArray'][0]
        Kd_z = GainParams['KdPosArray'][2]
        Kd_array = np.array([Kd_x, Kd_z])
        self.Kd_pos = np.diag(Kd_array)

        # Attitude gain load
        self.Kp_pitch = GainParams['KpOriArray'][1]

        if self.DobMode is False:
            self.Ki_pitch = GainParams['KiOriArray'][1]
            self.I_pitch_limit = GainParams['IOriLimit'][1]
            self.I_pitch = 0.0

        self.Kd_pitch = GainParams['KdOriArray'][1]

        self.accel_max = GainParams['AccelMax']

        self.u = np.zeros((2,))

    def compute_u(self, state, ref, d = np.zeros((3,)), dt = 0.01):
        """
        Compute control input u (f, My)
        :param state: p (x,z), v(vx, vz), theta, q
        :param ref: p_des (x, z), v_des (vx, vz)
        :param d: DOB or not
        :param dt:
        :return:
        """

        # Unpack state, reference, and disturbance
        p, v_body, th, q = self._unpack_state(state)
        p_des, v_des = self._unpack_ref(ref)
        f_ext, tau_ext = self._unpack_disturbance(d)


        # Compute current rotation matrix
        R_WB = pitch_to_rotm(th)
        v_world = R_WB @ v_body

        # Compute position and velocity error
        p_err = p - p_des
        v_err = v_world - v_des

        if self.DobMode is False:
            self.I_trans += dt*p_err
            self.I_trans = np.clip(self.I_trans, -self.I_trans_limit, self.I_trans_limit)
            linear_accelCmd = (-self.Kp_pos @ p_err
                               -self.Kd_pos @ v_err
                               -self.Ki_pos @ self.I_trans)

        else:
            # Compute desired force vector (PD)
            linear_accelCmd = (-self.Kp_pos @ p_err
                               -self.Kd_pos @ v_err)

        # Clamp accel max
        linear_accelCmd = np.clip(linear_accelCmd, -self.accel_max, self.accel_max)

        # After clamping, compensate for gravity vectory
        linear_accelCmd = linear_accelCmd - self.g_vec

        # Convert into force from acceleration cmd
        if self.DobMode is False:
            f_des = self.m * linear_accelCmd
        else:
            f_des = self.m * linear_accelCmd - R_WB @ f_ext

        self.u[0] = f_des.dot(R_WB[:,1])

        b2 = linear_accelCmd/np.linalg.norm(linear_accelCmd)

        th_des = np.arctan2(b2[1], b2[0])

        th_err = th - th_des
        q_err = q

        if self.DobMode is False:
            self.I_pitch += dt*th_err
            # Anti wind for I_pitch
            self.I_pitch = np.clip(self.I_pitch, self.I_pitch_limit, self.I_pitch_limit)
            angular_accelCmd = (-self.Kp_pitch @ th_err
                                -self.Kd_pitch @ q_err
                                -self.Ki_pitch @ self.I_trans)
            M_control = self.Jyy * angular_accelCmd

        else:
            angular_accelCmd = (-self.Kp_pitch @ th_err
                                - self.Kd_pitch @ q_err)
            M_control = self.Jyy * angular_accelCmd - tau_ext

        self.u[1] = M_control
        return self.u

    def _unpack_state(self, state):
        """Unpack a state from state vector"""
        p = state[0:2]
        v_body = state[2:4]
        theta = state[4]
        q = state[5]
        return p, v_body, theta, q

    def _unpack_ref(self, ref):
        """Unpack a ref from ref vector"""
        p_des = ref[0:2]
        v_des = ref[2:4]
        return p_des, v_des

    def _unpack_disturbance(self, d):
        """Unpack a disturbance from disturbance vector"""
        f_ext = d[0:2]
        tau_ext = d[3]
        return f_ext, tau_ext
