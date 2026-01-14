import numpy as np
from utils.math_tool import quaternion_to_rotm
from utils.custom_ode import custom_rk4

class StatePredictor:
    def __init__(self, DynParam, DobParam):
        """

        :param DynParam: m, MoiArray
        :param DobParam: As_array
        """

        # Set dynamic parameters
        self.m = DynParam['m']
        J_array = DynParam['MoiArray']
        self.J = np.diag(J_array)
        self.J_inv = np.diag([1.0/self.J[0][0],
                              1.0/self.J[1][1],
                              1.0/self.J[2][2]])

        self.As = np.diag(DobParam['As_array'])

        # Initialize estimated state and uncertainties
        self.z_hat = np.zeros(6)

        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.v_world = np.zeros(3)
        self.w = np.zeros(3)

        # sigma 0~3: matched uncertainties (thrust, moments)
        # sigma 4~5: unmatched uncertainties (lateral forces in world frame)
        self.sigma = np.zeros(6)

        g = -9.81
        self.g_vec = np.array([0.0, 0.0, g])

    def update(self, t_prev, t_curr, state, u_rot, sigma):
        """

        :param t_prev:
        :param t_curr:
        :param state:
        :param u_rot:
        :param sigma:
        :return: z_hat:
        """
        # Update state
        p, v_body, q, w = self._unpack_state(state)
        self.q = q
        R = quaternion_to_rotm(self.q)
        self.v_world = R @ v_body

        # Update uncertainties
        self.sigma = sigma

        t_ode = [t_prev, t_curr]

        self.z_hat = custom_rk4.do_step(self._dynamics,
                                        self.z_hat, u_rot, t_ode)

        return self.z_hat

    def _dynamics(self, t, z, u_rot):

        # Unpack matched and unmatched uncertainties
        sigma_m, sigma_un = self._unpack_sigma()
        # Get transformed v (World) and w
        z = self._get_z()

        # Allocate memory first
        f = np.zeros((6,))
        g = np.zeros((6,4))
        g_perp = np.zeros((6,2))

        T_rot = u_rot[0]
        M_rot = u_rot[1:]
        R = quaternion_to_rotm(self.q)
        Jw = self.J @ self.w
        coupling = np.cross(self.w, Jw)

        # Get Basis from rotation matrix
        e_z_b = R[:,2]

        # World frame basis for lateral forces
        e_x_w = np.array([1.0, 0.0, 0.0])
        e_y_w = np.array([0.0, 1.0, 0.0])

        # f(R_b_w) in world frame
        f[0:3] = self.g_vec + T_rot/self.m*e_z_b
        f[3:] = self.J_inv @ (M_rot - coupling)

        # Matched uncertainties: thrust (body z) and moments
        g[0:3,0] = 1.0/self.m * e_z_b
        g[3:6,1:4] = self.J_inv

        # Unmatched uncertainties: lateral forces in world frame
        g_perp[0:3,0] = 1.0/self.m * e_x_w
        g_perp[0:3,1] = 1.0/self.m * e_y_w

        z_tilde = self.z_hat - z

        z_hat_dot = (f + g @ sigma_m
                     + g_perp @ sigma_un
                     + self.As @ z_tilde)

        return z_hat_dot


    def _unpack_state(self, state):
        """Unpack state from state vector"""
        p = state[0:3]
        v_body = state[3:6]
        q = state[6:10]
        w = state[10:13]
        return p, v_body, q, w

    def _get_z(self):
        return np.concatenate([self.v_world, self.w])

    def _unpack_sigma(self):
        """Unpack matched and unmatched uncertainties"""
        sigma_m = self.sigma[0:4]
        sigma_un = self.sigma[4:6]
        return sigma_m, sigma_un


