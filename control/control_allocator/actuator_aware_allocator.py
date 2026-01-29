import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

class ActuatorAwareAllocator:
    def __init__(self, DroneParams, RotorParams ,AllocWeight):

        # 1. Drone Parameter
        self.l = DroneParams['arm_length']
        self.C_T = DroneParams['motor_const']
        self.k_m = DroneParams['moment_const']

        # 2. Actuator parameters
        # 2nd order dynamical parameters
        self.p1 = RotorParams['p1']     # Friction
        self.p2 = RotorParams['p2']     # Drag
        self.p3 = RotorParams['p3']     # Stiffness

        # 3. Saturation condition
        # Rotor speed constraint
        self.w_rotor_max = RotorParams['w_rotor_max']
        self.w_rotor_min = RotorParams['w_rotor_min']

        # Acceleration and jerk constraint
        self.alpha_max = RotorParams['alpha_max']
        self.j_max = RotorParams['j_max']

        # Lower and upper bound setup for Nonlinear constraint
        self.lb = np.concatenate([[-self.alpha_max]*6, [-self.j_max]*6])
        self.ub = np.concatenate([[self.alpha_max]*6, [self.j_max]*6])

        w_idle_speed = 2000
        w_idle_cmd = w_idle_speed * 8191.0 / 9800.0
        # It is used for warm start
        self.last_w_cmd = np.array([w_idle_cmd]*6)

        # 4. Weight parameters
        # Weight for collective thrust and moment along x, y, and z
        self.Q = np.diag(AllocWeight['Q'])
        # Regularization weight
        self.R = AllocWeight['R']*np.eye(6)

    def allocate(self, s_rotor_curr, u_des, dt):
        """

        :param s_rotor_curr: current rotor state (speed, acceleration) (12,)
        :param u_des: Desired control input (thrust, Moment)
        :return: Optimal rotor speed (6, )
        """
        # Unpack rotor state first
        w_rotor_curr, alpha_rotor_curr = self._unpack(s_rotor_curr)

        def objective(w_cmd):
            # 2nd order rotor dynamics
            j_vec = self._get_jerk_vec(s_rotor_curr, w_cmd)
            w_next = (w_rotor_curr
                      + alpha_rotor_curr * w_cmd * dt
                      + 0.5 * j_vec * dt**2)
            u_d_var = self._compute_wrench(w_next)

            l2_norm = (u_d_var - u_des).T @ self.Q @ (u_d_var - u_des)

            # Prevent too much acceleration
            reg = (w_cmd - w_rotor_curr).T @ self.R @ (w_cmd - w_rotor_curr)
            return l2_norm + reg

        def saturation_constraints(w_cmd):
            j_next = self._get_jerk_vec(s_rotor_curr, w_cmd)
            alpha_next = alpha_rotor_curr + j_next * dt

            return np.concatenate([alpha_next, j_next])

        nl_cons = NonlinearConstraint(saturation_constraints, self.lb, self.ub)

        res = minimize(objective,
                       x0 = self.last_w_cmd,
                       method='SLSQP',
                       bounds = [(self.w_rotor_min, self.w_rotor_max)]*6,
                       constraints = [nl_cons],
                       options = {'ftol': 1e-6,
                                  'disp': False})

        if res.success:
            self.last_w_cmd = res.x
            return res.x
        else:
            return self.last_w_cmd

    def _get_jerk_vec(self, s_rotor_curr, w_cmd):
        """
        Physical Jerk vector
        j_vec = -(p1 + p2*w)*alpha + p3*(w_cmd - w)
        :param s_rotor_curr:
        :param w_cmd:
        :return:
        """

        w = s_rotor_curr[0:6]
        alpha = s_rotor_curr[6:12]

        j_vec = (-(self.p1 + self.p2*w)*alpha
                 + self.p3*(w_cmd - w))
        return j_vec

    def _unpack(self, s_rotor):
        """
        Unpack rotor state into rotor speed and acceleration
        :return: w_rotor, alpha_rotor
        """
        w_rotor = s_rotor[0:6]
        alpha_rotor = s_rotor[6:12]
        return w_rotor, alpha_rotor

    def _compute_wrench(self, w_rotor):
        """
        Compute collective thrust and moment from six rotors' speed
        :param w_rotor:
        :return:
        """
        # Each rotor thrust
        T_rotor = self.C_T * (w_rotor**2)
        cp3, sp3 = np.cos(np.pi/3.0), np.sin(np.pi/3.0)

        # Compute collective thrust
        f = np.sum(T_rotor)
        # Compute moment along x, y, and z axis
        mx = self.l * ((T_rotor[0] + T_rotor[2])*cp3 + T_rotor[1]
                       -(T_rotor[3] + T_rotor[5])*cp3 - T_rotor[4])
        my = self.l * (-(T_rotor[0] + T_rotor[5])*sp3
                       +(T_rotor[2] + T_rotor[3])*sp3 )
        mz = self.k_m * (-T_rotor[0] + T_rotor[1] - T_rotor[2]
                         + T_rotor[3] - T_rotor[4] + T_rotor[5])

        return np.array([f, mx, my, mz])