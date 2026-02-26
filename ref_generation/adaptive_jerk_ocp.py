"""
Adaptive Jerk OCP for S550 3DOF

NMPC with adaptive jerk constraints derived from physical actuator limits:
- j_z: from rotor acceleration max (Ṫ = 2 × C_T × ω × α_max)
- j_x: from thrust × pitch rate max (T_hover × q_max / m)

Uses 2-step delay model: dT_cmd(k-2) vs dT_actual(k-1)

Pure CasADi implementation (no acados dependency).

Author: Claude
Date: 2026-02-26
"""

import numpy as np
import casadi as cs
from collections import deque


class AdaptiveJerkLimiter:
    """
    Computes and adapts jerk limits based on actuator tracking.

    Physical limits:
    - j_z_max = (2 × C_T × ω_hover × α_max × 6) / m
    - j_x_max = T_hover × q_max / m
    """

    def __init__(self, params):
        # Physical parameters
        self.m = params['m']
        self.C_T = params['motor_const']
        self.alpha_max = params['alpha_rotor_max']  # RPM/s
        self.q_max = params.get('q_max', 3.0)  # rad/s, max pitch rate

        # Rotor speed bounds
        self.w_min = params['w_rotor_min']
        self.w_max = params['w_rotor_max']

        # Compute hover rotor speed
        T_hover_per_rotor = self.m * 9.81 / 6.0
        self.w_hover = np.sqrt(T_hover_per_rotor / self.C_T)
        self.T_hover = self.m * 9.81

        # Compute physical jerk limits
        # j_z = dT/dt / m = (2 × C_T × ω × α × 6) / m
        # Factor 6 for hexarotor total thrust
        T_dot_max = 2.0 * self.C_T * self.w_hover * self.alpha_max * 6.0
        self.j_z_physical = T_dot_max / self.m

        # j_x = T × q_dot / m ≈ T_hover × q_max / m
        self.j_x_physical = self.T_hover * self.q_max / self.m

        print(f"Physical jerk limits:")
        print(f"  j_z_max = {self.j_z_physical:.2f} m/s³")
        print(f"  j_x_max = {self.j_x_physical:.2f} m/s³")

        # Adaptive parameters
        self.K_p = params.get('K_p', 0.2)
        self.deadband = params.get('deadband', 0.15)

        # Current adaptive limits (start at physical max)
        self.j_z = self.j_z_physical
        self.j_x = self.j_x_physical

        # Bounds for adaptation
        self.j_z_min = 0.3 * self.j_z_physical
        self.j_x_min = 0.3 * self.j_x_physical

        # History for delay model
        self.dT_cmd_history = deque(maxlen=3)
        self.dT_actual_history = deque(maxlen=2)
        self.T_prev = None

        # Statistics
        self.eta_history = deque(maxlen=100)

    def update(self, T_cmd, T_actual):
        """
        Update jerk limits based on actuator tracking.

        Compares dT_cmd(k-2) with dT_actual(k-1).

        :param T_cmd: Commanded total thrust (N)
        :param T_actual: Actual total thrust (N)
        :return: (j_z, j_x) current jerk limits
        """
        if self.T_prev is not None:
            dT_cmd = T_cmd - self.T_prev['cmd']
            dT_actual = T_actual - self.T_prev['actual']

            self.dT_cmd_history.append(dT_cmd)
            self.dT_actual_history.append(dT_actual)

            # Compare k-2 command with k-1 actual
            if len(self.dT_cmd_history) >= 3 and len(self.dT_actual_history) >= 2:
                dT_cmd_k2 = self.dT_cmd_history[-3]
                dT_actual_k1 = self.dT_actual_history[-2]

                dT_cmd_norm = abs(dT_cmd_k2)
                dT_actual_norm = abs(dT_actual_k1)

                # Only adapt if command was significant
                if dT_cmd_norm > 0.5:
                    sign_match = (dT_cmd_k2 * dT_actual_k1) >= 0

                    if sign_match:
                        eta = dT_actual_norm / dT_cmd_norm
                        eta = np.clip(eta, 0.0, 2.0)
                    else:
                        eta = 0.5

                    error = 1.0 - eta

                    if abs(error) > self.deadband:
                        if error > 0:  # Actuator struggling
                            dj_ratio = -self.K_p * error
                        else:  # Has headroom
                            dj_ratio = -0.3 * self.K_p * error

                        # Apply to both limits proportionally
                        self.j_z = np.clip(
                            self.j_z * (1.0 + dj_ratio),
                            self.j_z_min,
                            self.j_z_physical
                        )
                        self.j_x = np.clip(
                            self.j_x * (1.0 + dj_ratio),
                            self.j_x_min,
                            self.j_x_physical
                        )

                    self.eta_history.append(eta)

        self.T_prev = {'cmd': T_cmd, 'actual': T_actual}

        return self.j_z, self.j_x

    def get_limits(self):
        """Get current jerk limits."""
        return self.j_z, self.j_x

    def get_statistics(self):
        """Get adapter statistics."""
        if len(self.eta_history) == 0:
            return {'eta_mean': 1.0, 'j_z': self.j_z, 'j_x': self.j_x}
        return {
            'eta_mean': np.mean(self.eta_history),
            'j_z': self.j_z,
            'j_x': self.j_x
        }

    def reset(self):
        """Reset to physical limits."""
        self.j_z = self.j_z_physical
        self.j_x = self.j_x_physical
        self.dT_cmd_history.clear()
        self.dT_actual_history.clear()
        self.T_prev = None
        self.eta_history.clear()


class AdaptiveJerkOCP:
    """
    S550 3DOF NMPC with adaptive jerk constraints.

    Pure CasADi implementation with soft jerk penalties.
    State: [px, pz, vx, vz, th, q] (6 states)
    Control: [u1, u2, u3] (3 rotor group thrusts)
    """

    def __init__(self, DynParam=None, DroneParam=None, RotorParam=None, MpcParam=None):
        # Default parameters
        if DynParam is None:
            DynParam = {'m': 3.0, 'MoiArray': [0.065, 0.065, 0.10]}
        if DroneParam is None:
            DroneParam = {
                'arm_length': 0.265,
                'motor_const': 1.465e-7,
                'moment_const': 0.01569
            }
        if RotorParam is None:
            RotorParam = {
                'w_rotor_min': 2000,
                'w_rotor_max': 7200,
                'alpha_rotor_max': 15000,
                'num_rotors': 3
            }

        self.m = DynParam['m']
        self.Jyy = DynParam['MoiArray'][1]
        self.l = DroneParam['arm_length']
        self.C_T = DroneParam['motor_const']
        self.k_m = DroneParam['moment_const']

        # Initialize jerk limiter
        limiter_params = {
            'm': self.m,
            'motor_const': self.C_T,
            'alpha_rotor_max': RotorParam['alpha_rotor_max'],
            'w_rotor_min': RotorParam['w_rotor_min'],
            'w_rotor_max': RotorParam['w_rotor_max'],
            'q_max': MpcParam.get('q_max', 3.0) if MpcParam else 3.0,
            'K_p': MpcParam.get('K_p', 0.2) if MpcParam else 0.2,
            'deadband': MpcParam.get('deadband', 0.15) if MpcParam else 0.15,
        }
        self.jerk_limiter = AdaptiveJerkLimiter(limiter_params)

        # Rotor speed limits -> thrust limits
        w_max = RotorParam['w_rotor_max']
        w_min = RotorParam['w_rotor_min']
        self.u_max = self.C_T * w_max**2
        self.u_min = self.C_T * w_min**2

        # MPC parameters
        if MpcParam is None:
            self.T_horizon = 0.20
            self.N = 20
            self.Q = np.diag([10, 15, 2, 2, 1, 0.1])
            self.R = np.diag([0.01] * 3)
            self.R_jerk = 1.0
        else:
            self.T_horizon = MpcParam.get('t_horizon', 0.20)
            self.N = MpcParam.get('n_nodes', 20)
            self.Q = np.diag(MpcParam.get('QArray', [10, 15, 2, 2, 1, 0.1]))
            self.R = MpcParam.get('R', 0.01) * np.eye(3)
            self.R_jerk = MpcParam.get('R_jerk', 1.0)

        self.dt = self.T_horizon / self.N
        self.nx = 6
        self.nu = 3

        # Hover thrust per rotor group (needed before building solver)
        self.u_hover = self.m * 9.81 / 6.0

        # Build solver
        self._build_solver()

        # Warm start storage
        self.x_warm = None
        self.u_warm = None
        self.previous_u0 = None

    def _build_solver(self):
        """Build CasADi NMPC solver."""
        nx = self.nx
        nu = self.nu

        # State and control symbols
        x = cs.MX.sym('x', nx)
        u = cs.MX.sym('u', nu)

        # Dynamics
        f_expl = self._dynamics(x, u)

        # RK4 discretization
        k1 = f_expl
        k2 = self._dynamics(x + self.dt/2 * k1, u)
        k3 = self._dynamics(x + self.dt/2 * k2, u)
        k4 = self._dynamics(x + self.dt * k3, u)
        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        self.f_discrete = cs.Function('f_discrete', [x, u], [x_next])

        # Build NLP
        self.opti = cs.Opti()

        # Decision variables
        self.X = self.opti.variable(nx, self.N + 1)
        self.U = self.opti.variable(nu, self.N)

        # Parameters
        self.x0_param = self.opti.parameter(nx)
        self.x_ref_param = self.opti.parameter(nx)
        self.u_prev_param = self.opti.parameter(nu)
        self.j_z_param = self.opti.parameter()

        # Cost
        cost = 0

        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]

            # State error
            x_err = x_k - self.x_ref_param
            cost += cs.mtimes([x_err.T, self.Q, x_err])

            # Control cost (deviation from hover)
            u_err = u_k - self.u_hover
            cost += cs.mtimes([u_err.T, self.R, u_err])

            # Jerk cost (soft constraint)
            if k == 0:
                du = u_k - self.u_prev_param
            else:
                du = u_k - self.U[:, k-1]

            # Total thrust rate -> jerk
            dT = 2.0 * cs.sum1(du)  # Factor 2 for paired motors
            T_dot = dT / self.dt
            jerk_z = T_dot / self.m

            # Soft jerk constraint
            jerk_violation = cs.fmax(0, cs.fabs(jerk_z) - self.j_z_param)
            cost += self.R_jerk * jerk_violation**2

        # Terminal cost
        x_N = self.X[:, self.N]
        x_err_N = x_N - self.x_ref_param
        cost += 2.0 * cs.mtimes([x_err_N.T, self.Q, x_err_N])

        self.opti.minimize(cost)

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        for k in range(self.N):
            x_next = self.f_discrete(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)
            self.opti.subject_to(self.opti.bounded(self.u_min, self.U[:, k], self.u_max))

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4
        }
        self.opti.solver('ipopt', opts)

    def _dynamics(self, x, u):
        """
        3DOF drone dynamics.

        State: [px, pz, vx, vz, th, q]
        Control: [u1, u2, u3] (rotor group thrusts)
        """
        # Unpack
        vx = x[2]
        vz = x[3]
        th = x[4]
        q = x[5]

        # Total thrust (factor 2 for paired motors)
        f_col = 2.0 * (u[0] + u[1] + u[2])

        # Pitch moment: My = l * sqrt(3) * (-u1 + u3)
        # Matches drone_converter.py Kf matrix: 2*l*sin(60°) = l*sqrt(3)
        My = self.l * cs.sqrt(3.0) * (-u[0] + u[2])

        # Rotation matrix
        cth = cs.cos(th)
        sth = cs.sin(th)

        # Dynamics (ideal model without COM offset - disturbance handled by HGDO)
        dpxdt = vx
        dpzdt = vz
        dvxdt = -f_col / self.m * sth
        dvzdt = f_col / self.m * cth - 9.81
        dthdt = q
        dqdt = My / self.Jyy

        return cs.vertcat(dpxdt, dpzdt, dvxdt, dvzdt, dthdt, dqdt)

    def solve(self, state, ref, T_actual=None):
        """
        Solve OCP with adaptive jerk.

        :param state: [px, pz, vx_body, vz_body, th, q]
        :param ref: [px_des, pz_des, vx_des, vz_des]
        :param T_actual: Actual total thrust for adaptation
        :return: (status, w_cmd, info)
        """
        # Update jerk limits
        if self.previous_u0 is not None:
            T_cmd = 2.0 * np.sum(self.previous_u0)
            if T_actual is None:
                T_actual = T_cmd
            j_z, j_x = self.jerk_limiter.update(T_cmd, T_actual)
        else:
            j_z, j_x = self.jerk_limiter.get_limits()

        # Build full state reference
        x_ref = np.zeros(6)
        x_ref[0:4] = ref[0:4]
        x_ref[4] = 0.0  # th_des
        x_ref[5] = 0.0  # q_des

        # Transform state (body to world velocity)
        state_world = self._state_transform(state)

        # Set parameters
        self.opti.set_value(self.x0_param, state_world)
        self.opti.set_value(self.x_ref_param, x_ref)
        self.opti.set_value(self.j_z_param, j_z)

        if self.previous_u0 is not None:
            self.opti.set_value(self.u_prev_param, self.previous_u0)
        else:
            self.opti.set_value(self.u_prev_param, np.ones(3) * self.u_hover)

        # Warm start
        if self.x_warm is not None:
            self.opti.set_initial(self.X, self.x_warm)
            self.opti.set_initial(self.U, self.u_warm)
        else:
            for k in range(self.N + 1):
                self.opti.set_initial(self.X[:, k], state_world)
            for k in range(self.N):
                self.opti.set_initial(self.U[:, k], np.ones(3) * self.u_hover)

        # Solve
        try:
            sol = self.opti.solve()
            status = 0

            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)

            # Store for warm start
            self.x_warm = np.hstack([x_opt[:, 1:], x_opt[:, -1:]])
            self.u_warm = np.hstack([u_opt[:, 1:], u_opt[:, -1:]])

            u_first = u_opt[:, 0]

        except Exception as e:
            status = 1
            if self.previous_u0 is not None:
                u_first = self.previous_u0
            else:
                u_first = np.ones(3) * self.u_hover

        self.previous_u0 = u_first.copy()

        # Convert to rotor speed
        w_cmd = np.sqrt(np.clip(u_first, 0, None) / self.C_T)

        info = {
            'j_z_limit': j_z,
            'j_x_limit': j_x,
            'stats': self.jerk_limiter.get_statistics()
        }

        return status, w_cmd, info

    def _state_transform(self, state):
        """Transform body velocity to world velocity."""
        from utils.math_tool import pitch_to_rotm
        th = state[4]
        R = pitch_to_rotm(th)
        v_body = state[2:4]
        v_world = R @ v_body
        state_new = state.copy()
        state_new[2:4] = v_world
        return state_new

    def get_jerk_limiter(self):
        return self.jerk_limiter

    def reset(self):
        """Reset controller."""
        self.x_warm = None
        self.u_warm = None
        self.previous_u0 = None
        self.jerk_limiter.reset()
