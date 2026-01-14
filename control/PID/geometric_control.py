import numpy as np
from utils.math_tool import quaternion_to_rotm
from utils.math_tool import compute_angle_axis

class GeometricControl:
    def __init__(self, DynamicParams, GainParams, DobMode = False):
        """
        :param DynamicParams:
        :param GainParams:
        """
        # Load dynamics parameter
        self.m = DynamicParams['m']
        self.J = np.diag(DynamicParams['MoiArray'])

        g = -9.81
        self.g_vec = np.array([0.0, 0.0, g])

        self.DobMode = DobMode

        # Position gain load
        self.Kp_pos = np.diag(GainParams['KpPosArray'])
        self.Kd_pos = np.diag(GainParams['KdPosArray'])
        if self.DobMode is False:
            self.Ki_pos = np.diag(GainParams['KiPosArray'])
            self.I_trans_limit = GainParams['ITransLimit']

        # Attitude gain load
        self.Kp_att = np.diag(GainParams['KpOriArray'])
        self.Kd_att = np.diag(GainParams['KdOriArray'])
        if self.DobMode is False:
            self.Ki_att = np.diag(GainParams['KiOriArray'])
            self.I_att_limit = GainParams['IOriLimit']

        self.accel_max = GainParams['AccelMax']

        if self.DobMode is False:
            self.I_trans = np.zeros((3,))
            self.I_att = np.zeros((3,))

        self.u = np.zeros((4,))

    def compute_u(self, s, ref, disturbance = np.zeros((6,)), dt = 0.01):
        """
        Compute control input u (f, M) using geometric control
        :param s: p, v, q, w
        :param ref: p_des, v_des, yaw_des, yaw_des_dot
        :param disturbance: DOB or not
        :return: u (f, M)
        """

        # Unpack state, reference, and disturbance
        p, v_body, q, w = self._unpack_state(s)
        p_des, v_des, yaw_des, w_des = self._unpack_ref(ref)
        f_ext, tau_ext = self._unpack_disturbance(disturbance)

        # Compute current rotation matrix
        R = quaternion_to_rotm(q)
        v_world = R @ v_body
        # Compute position and velocity error
        p_err = p - p_des
        v_err = v_world - v_des

        if self.DobMode is False:
            self.I_trans += dt*p_err
            # Anti wind for I_trans
            self.I_trans = np.clip(self.I_trans, -self.I_trans_limit, self.I_trans_limit)
            linear_accelCmd = (-self.Kp_pos @ p_err
                     -self.Kd_pos @ v_err
                     -self.Ki_pos @ self.I_trans)
        else:
            # Compute desired force vector (PID)
            linear_accelCmd =(-self.Kp_pos @ p_err
                              - self.Kd_pos @ v_err)

        # Clamp accel cmd
        linear_accelCmd = self._accel_clamp(linear_accelCmd)

        # After clamping, Compensate gravity
        linear_accelCmd = linear_accelCmd - self.g_vec

        # Convert into force from accelCmd
        if self.DobMode is False:
            f_des = self.m * linear_accelCmd
        else:
            f_des = self.m * linear_accelCmd - R @ f_ext

        self.u[0] = f_des.dot(R[:,2])

        b3 = linear_accelCmd/np.linalg.norm(linear_accelCmd)

        # Lee at al. 2010 approach to compute b1 and b2

        c1 = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

        if np.linalg.norm(np.cross(c1, b3)) < 1e-6:
            c1 = np.array([1.0, 0.0, 0.0])

        b2 = np.cross(b3, c1)
        b2 = b2 / np.linalg.norm(b2)
        b1 = np.cross(b2, b3)

        # Construct desired rotation matrix
        R_des = np.column_stack((b1, b2, b3))

        e_R = compute_angle_axis(R_des, R)
        e_w = w - R.T @ R_des @ w_des

        if self.DobMode is False:
            self.I_att += dt*e_R
            # Anti wind for I_att
            self.I_att = np.clip(self.I_att, -self.I_att_limit, self.I_att_limit)
            angular_accelCmd = (-self.Kp_att @ e_R
                                -self.Kd_att @ e_w
                                -self.Ki_att @ self.I_att)
            M_control = self.J @ angular_accelCmd
        else:
            angular_accelCmd =(-self.Kp_att @ e_R
                               -self.Kd_att @ e_w)
            M_control = self.J @ angular_accelCmd - tau_ext

        self.u[1:] = M_control
        return self.u

    def pack_ref(self, p_des, v_des, yaw_des, yaw_des_dot):
        """Pack ref vector"""
        return np.concatenate([p_des, v_des, yaw_des, yaw_des_dot])

    def _unpack_state(self, s):
        """Unpack a state from state vector"""
        p = s[0:3]
        v = s[3:6]
        q = s[6:10]
        w = s[10:13]
        return p, v, q, w

    def _unpack_disturbance(self, disturbance):
        """Unpack a disturbance"""
        f_ext = disturbance[0:3]
        tau_ext = disturbance[3:6]
        return f_ext, tau_ext


    def _unpack_ref(self, ref):
        """Unpack a reference state from state vector"""
        p_des = ref[0:3]
        v_des = ref[3:6]
        yaw_des = ref[6]
        yaw_des_dot = ref[7]
        w_des = np.array([0.0, 0.0, yaw_des_dot])
        return p_des, v_des, yaw_des, w_des

    def _accel_clamp(self, accel_cmd):
        accel_cmd_clamp = np.clip(accel_cmd, -self.accel_max, self.accel_max)
        return accel_cmd_clamp