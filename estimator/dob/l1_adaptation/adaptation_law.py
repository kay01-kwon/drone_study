import numpy as np
from utils.math_tool import quaternion_to_rotm

class AdaptationLaw:
    def __init__(self, DynParam, DobParam):

        # Set dynamic parameters
        self.m = DynParam['m']
        J_array = DynParam['MoiArray']
        self.J = np.diag(J_array)

        # Set hurwitz matrix
        self.As = np.diag(DobParam['As_array'])
        self.As_inv = np.zeros((6,6))

        for i in range(6):
            self.As_inv[i,i] = 1.0/self.As[i,i]

        # Initialize sigma hat as zeros
        self.sigma_hat = np.zeros(6)

    def update(self, t_prev, t_curr, z_tilde, state):

        dt = t_curr - t_prev

        mu = np.zeros(6)
        Phi_inv = np.zeros((6,6))

        for i in range(6):
            exp_As_dt = np.exp(self.As[i,i] * dt)
            Phi_ii = self.As_inv[i,i]*(exp_As_dt - 1.0)
            Phi_inv[i,i] = 1.0/Phi_ii
            mu[i] = exp_As_dt * z_tilde[i]

        # Get rotation matrix
        q = self._get_quaternion_from_state(state)
        R_b_w = quaternion_to_rotm(q)

        # Extract body frame basis from rotation matrix
        e_z_b = R_b_w[:,2]

        # World frame basis for lateral forces
        e_x_w = np.array([1.0, 0.0, 0.0])
        e_y_w = np.array([0.0, 1.0, 0.0])

        G_inv = np.zeros((6,6))

        # Matched uncertainties: thrust (body z) and moments
        G_inv[0,0:3] = self.m * e_z_b.T
        G_inv[1:4,3:6] = self.J

        # Unmatched uncertainties: lateral forces in world frame
        G_inv[4,0:3] = self.m * e_x_w.T
        G_inv[5,0:3] = self.m * e_y_w.T

        self.sigma_hat = -G_inv @ Phi_inv @ mu

        return self.sigma_hat


    def _get_quaternion_from_state(self,state):
        q = state[6:10]
        return q