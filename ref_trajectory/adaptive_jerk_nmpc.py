"""
Adaptive Jerk NMPC Controller

NMPC controller with adaptive jerk constraints based on actuator tracking.
Uses CasADi for optimization with soft jerk constraints.

Author: Claude
Date: 2026-02-26
"""

import numpy as np
import casadi as cs
from ref_trajectory.jerk_adapter import AdaptiveJerkAdapter


class AdaptiveJerkNMPC:
    """
    NMPC with adaptive jerk constraints.

    Features:
    - Standard NMPC with position/velocity tracking
    - Adaptive jerk limit based on actuator performance
    - Soft jerk constraints via penalty
    - ω × T rotation budget handling
    """

    def __init__(self, DynParam=None, DroneParam=None, MpcParam=None, AdapterParam=None):
        """
        Initialize NMPC with adaptive jerk.

        :param DynParam: Dynamic parameters (m, MoiArray)
        :param DroneParam: Drone parameters (arm_length, motor_const, moment_const)
        :param MpcParam: MPC parameters (t_horizon, n_nodes, Q, R, etc.)
        :param AdapterParam: Jerk adapter parameters
        """
        # Default parameters
        if DynParam is None:
            self.m = 3.0
            self.J = np.array([0.065, 0.065, 0.10])
        else:
            self.m = DynParam['m']
            self.J = DynParam['MoiArray']

        if DroneParam is None:
            self.l = 0.265
            self.C_T = 1.465e-7
            self.k_m = 0.01569
        else:
            self.l = DroneParam['arm_length']
            self.C_T = DroneParam['motor_const']
            self.k_m = DroneParam['moment_const']

        if MpcParam is None:
            self.T_horizon = 0.5
            self.N = 20
            self.dt = self.T_horizon / self.N
            self.Q_pos = np.diag([10.0, 10.0, 10.0])
            self.Q_vel = np.diag([2.0, 2.0, 2.0])
            self.Q_att = np.diag([1.0, 1.0, 1.0, 1.0])
            self.Q_omega = np.diag([0.1, 0.1, 0.1])
            self.R = np.diag([0.01] * 6)
            self.R_jerk = 10.0  # Jerk penalty weight
        else:
            self.T_horizon = MpcParam.get('t_horizon', 0.5)
            self.N = MpcParam.get('n_nodes', 20)
            self.dt = self.T_horizon / self.N
            self.Q_pos = np.diag(MpcParam.get('Q_pos', [10.0, 10.0, 10.0]))
            self.Q_vel = np.diag(MpcParam.get('Q_vel', [2.0, 2.0, 2.0]))
            self.Q_att = np.diag(MpcParam.get('Q_att', [1.0, 1.0, 1.0, 1.0]))
            self.Q_omega = np.diag(MpcParam.get('Q_omega', [0.1, 0.1, 0.1]))
            self.R = np.diag(MpcParam.get('R', [0.01] * 6))
            self.R_jerk = MpcParam.get('R_jerk', 10.0)

        # Thrust limits
        self.u_max = self.C_T * 7200.0**2
        self.u_min = self.C_T * 2000.0**2

        # Initialize jerk adapter
        if AdapterParam is None:
            AdapterParam = {'m': self.m}
        else:
            AdapterParam['m'] = self.m
        self.jerk_adapter = AdaptiveJerkAdapter(AdapterParam)

        # Previous control for jerk computation
        self.u_prev = None
        self.T_actual_prev = None

        # Build NMPC solver
        self._build_solver()

        # State for warm start
        self.x_warm = None
        self.u_warm = None

    def _build_solver(self):
        """Build CasADi NMPC solver."""
        # State: [p(3), v(3), q(4), w(3)] = 13 states
        nx = 13
        nu = 6  # 6 rotor thrusts

        # CasADi symbols
        x = cs.MX.sym('x', nx)
        u = cs.MX.sym('u', nu)

        # Unpack state
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        # Dynamics (same as existing model)
        f_expl = self._dynamics(p, v, q, w, u)

        # Discrete dynamics (RK4)
        k1 = f_expl
        k2 = self._dynamics_from_state(x + self.dt/2 * k1, u)
        k3 = self._dynamics_from_state(x + self.dt/2 * k2, u)
        k4 = self._dynamics_from_state(x + self.dt * k3, u)
        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Dynamics function
        self.f_discrete = cs.Function('f_discrete', [x, u], [x_next])

        # Build NLP
        # Decision variables: x_0, u_0, x_1, u_1, ..., x_N
        self.opti = cs.Opti()

        # State trajectory
        self.X = self.opti.variable(nx, self.N + 1)
        # Control trajectory
        self.U = self.opti.variable(nu, self.N)

        # Parameters
        self.x0_param = self.opti.parameter(nx)  # Initial state
        self.x_ref_param = self.opti.parameter(nx, self.N + 1)  # Reference trajectory
        self.u_prev_param = self.opti.parameter(nu)  # Previous control (for jerk)
        self.j_limit_param = self.opti.parameter()  # Adaptive jerk limit

        # Cost
        cost = 0

        for k in range(self.N):
            # State error
            x_k = self.X[:, k]
            x_ref_k = self.x_ref_param[:, k]

            p_err = x_k[0:3] - x_ref_k[0:3]
            v_err = x_k[3:6] - x_ref_k[3:6]
            q_err = x_k[6:10] - x_ref_k[6:10]
            w_err = x_k[10:13] - x_ref_k[10:13]

            # Quadratic cost
            cost += cs.mtimes([p_err.T, self.Q_pos, p_err])
            cost += cs.mtimes([v_err.T, self.Q_vel, v_err])
            cost += cs.mtimes([q_err.T, self.Q_att, q_err])
            cost += cs.mtimes([w_err.T, self.Q_omega, w_err])

            # Control cost (penalize deviation from hover, not absolute value)
            u_k = self.U[:, k]
            u_hover = self.m * 9.81 / 6.0  # Hover thrust per rotor
            u_err = u_k - u_hover
            cost += cs.mtimes([u_err.T, self.R, u_err])

            # Jerk cost (soft constraint)
            if k == 0:
                du = u_k - self.u_prev_param
            else:
                du = u_k - self.U[:, k-1]

            # Total thrust rate
            dT = cs.sum1(du)  # Sum of all thrust changes
            T_dot = dT / self.dt

            # Convert to jerk (acceleration rate)
            jerk = T_dot / self.m

            # Soft jerk constraint: penalty for exceeding limit
            jerk_violation = cs.fmax(0, cs.fabs(jerk) - self.j_limit_param)
            cost += self.R_jerk * jerk_violation**2

        # Terminal cost
        x_N = self.X[:, self.N]
        x_ref_N = self.x_ref_param[:, self.N]
        p_err_N = x_N[0:3] - x_ref_N[0:3]
        v_err_N = x_N[3:6] - x_ref_N[3:6]
        cost += 2.0 * cs.mtimes([p_err_N.T, self.Q_pos, p_err_N])
        cost += 2.0 * cs.mtimes([v_err_N.T, self.Q_vel, v_err_N])

        self.opti.minimize(cost)

        # Constraints
        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        # Dynamics constraints
        for k in range(self.N):
            x_next = self.f_discrete(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Control bounds
        for k in range(self.N):
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

    def _dynamics(self, p, v, q, w, u):
        """
        Compute state derivatives.

        :param p: Position (world frame)
        :param v: Velocity (world frame)
        :param q: Quaternion (body to world)
        :param w: Angular velocity (body frame)
        :param u: Rotor thrusts [u1, ..., u6]
        :return: State derivative
        """
        # Rotation matrix
        R = self._quaternion_to_rotm(q)

        # Total thrust
        f_col = cs.sum1(u)
        f_body = cs.vertcat(0, 0, f_col)

        # Linear dynamics
        dpdt = v
        g_vec = cs.vertcat(0, 0, -9.81)
        dvdt = cs.mtimes(R, f_body) / self.m + g_vec

        # Quaternion dynamics
        w_quat = cs.vertcat(0, w)
        dqdt = 0.5 * self._otimes(q, w_quat)

        # Angular dynamics
        Mx, My, Mz = self._control_alloc_moment(u)
        M = cs.vertcat(Mx, My, Mz)

        Jxx, Jyy, Jzz = self.J[0], self.J[1], self.J[2]
        J_inv = cs.diag(cs.vertcat(1/Jxx, 1/Jyy, 1/Jzz))

        inertial_effect = cs.vertcat(
            (Jzz - Jyy) / Jxx * w[1] * w[2],
            (Jxx - Jzz) / Jyy * w[0] * w[2],
            (Jyy - Jxx) / Jzz * w[0] * w[1]
        )
        dwdt = cs.mtimes(J_inv, M) - inertial_effect

        return cs.vertcat(dpdt, dvdt, dqdt, dwdt)

    def _dynamics_from_state(self, x, u):
        """Dynamics from full state vector."""
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]
        return self._dynamics(p, v, q, w, u)

    def _quaternion_to_rotm(self, q):
        """Convert quaternion to rotation matrix."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        R = cs.vertcat(
            cs.horzcat(1-2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            cs.horzcat(2*(qy*qx + qw*qz), 1-2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)),
            cs.horzcat(2*(qz*qx - qw*qy), 2*(qz*qy + qw*qx), 1-2*(qx*qx + qy*qy))
        )
        return R

    def _otimes(self, q1, q2):
        """Quaternion multiplication."""
        q1_w, q1_x, q1_y, q1_z = q1[0], q1[1], q1[2], q1[3]
        q1_L = cs.vertcat(
            cs.horzcat(q1_w, -q1_x, -q1_y, -q1_z),
            cs.horzcat(q1_x, q1_w, -q1_z, q1_y),
            cs.horzcat(q1_y, q1_z, q1_w, -q1_x),
            cs.horzcat(q1_z, -q1_y, q1_x, q1_w)
        )
        return cs.mtimes(q1_L, q2)

    def _control_alloc_moment(self, u):
        """Compute moments from rotor thrusts."""
        cos_pi_3 = np.cos(np.pi / 3)
        sin_pi_3 = np.sin(np.pi / 3)

        Mx = self.l * (
            (u[0] + u[2]) * cos_pi_3 + u[1]
            - (u[3] + u[5]) * cos_pi_3 - u[4]
        )
        My = self.l * (
            -(u[0] + u[5]) * sin_pi_3
            + (u[2] + u[3]) * sin_pi_3
        )
        Mz = self.k_m * (
            -u[0] + u[1] - u[2] + u[3] - u[4] + u[5]
        )
        return Mx, My, Mz

    def solve(self, state, ref_trajectory, T_actual=None):
        """
        Solve NMPC with adaptive jerk constraint.

        :param state: Current state [p, v, q, w] (13,)
        :param ref_trajectory: Reference trajectory (13, N+1) or (8,) for single ref
        :param T_actual: Actual total thrust from actuators (for jerk adaptation)
        :return: status, u_opt (rotor thrusts), info
        """
        # Ensure ref_trajectory is (13, N+1)
        if ref_trajectory.ndim == 1:
            # Single reference, replicate
            ref_traj = np.tile(ref_trajectory.reshape(-1, 1), (1, self.N + 1))
        elif ref_trajectory.shape[1] != self.N + 1:
            # Interpolate or pad
            ref_traj = np.zeros((13, self.N + 1))
            for i in range(13):
                ref_traj[i, :] = np.interp(
                    np.linspace(0, 1, self.N + 1),
                    np.linspace(0, 1, ref_trajectory.shape[1]),
                    ref_trajectory[i, :]
                )
        else:
            ref_traj = ref_trajectory

        # Update jerk adapter
        if self.u_prev is not None:
            T_cmd = np.sum(self.u_prev)
            if T_actual is None:
                T_actual = T_cmd  # Assume perfect tracking if not provided
            omega = state[10:13]
            j_limit, T_dot_available = self.jerk_adapter.update(T_cmd, T_actual, omega)
        else:
            j_limit = self.jerk_adapter.get_current_jerk_limit()

        # Set parameters
        self.opti.set_value(self.x0_param, state)
        self.opti.set_value(self.x_ref_param, ref_traj)
        self.opti.set_value(self.j_limit_param, j_limit)

        if self.u_prev is not None:
            self.opti.set_value(self.u_prev_param, self.u_prev)
        else:
            # Initialize with hover thrust
            u_hover = self.m * 9.81 / 6.0
            self.opti.set_value(self.u_prev_param, np.ones(6) * u_hover)

        # Warm start
        if self.x_warm is not None:
            self.opti.set_initial(self.X, self.x_warm)
            self.opti.set_initial(self.U, self.u_warm)
        else:
            # Initialize with hover
            for k in range(self.N + 1):
                self.opti.set_initial(self.X[:, k], state)
            u_hover = self.m * 9.81 / 6.0
            for k in range(self.N):
                self.opti.set_initial(self.U[:, k], np.ones(6) * u_hover)

        # Solve
        try:
            sol = self.opti.solve()
            status = 0

            # Extract solution
            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)

            # Store for warm start (shift)
            self.x_warm = np.hstack([x_opt[:, 1:], x_opt[:, -1:]])
            self.u_warm = np.hstack([u_opt[:, 1:], u_opt[:, -1:]])

            # Get first control
            u_first = u_opt[:, 0]

        except Exception as e:
            status = 1
            # Use previous or hover
            if self.u_prev is not None:
                u_first = self.u_prev
            else:
                u_first = np.ones(6) * self.m * 9.81 / 6.0

        # Store for next iteration
        self.u_prev = u_first.copy()

        # Info
        info = {
            'j_limit': j_limit,
            'adapter_stats': self.jerk_adapter.get_statistics()
        }

        return status, u_first, info

    def solve_for_trajectory(self, state, t_curr, u_prev_rotor_speed=None):
        """
        Solve NMPC for trajectory tracking (compatible with existing interface).

        :param state: Current state [p, v, q, w]
        :param t_curr: Current time
        :param u_prev_rotor_speed: Previous rotor speeds (for thrust computation)
        :return: status, rotor_speed_cmd
        """
        from ref_generation.ref_generator import get_reference

        # Build reference trajectory along horizon
        ref_traj = np.zeros((13, self.N + 1))
        for k in range(self.N + 1):
            t_ref = t_curr + k * self.dt
            ref = get_reference(t_ref)

            # Convert to NMPC format
            ref_traj[0:6, k] = ref[0:6]  # p, v
            ref_traj[6, k] = np.cos(ref[6] / 2.0)  # qw
            ref_traj[9, k] = np.sin(ref[6] / 2.0)  # qz (yaw)
            ref_traj[12, k] = ref[7]  # yaw rate

        # Compute actual thrust from previous rotor speeds
        T_actual = None
        if u_prev_rotor_speed is not None:
            T_actual = np.sum(self.C_T * u_prev_rotor_speed**2)

        # Transform state (body velocity to world)
        state_world = self._state_transform(state)

        # Solve
        status, u_opt, info = self.solve(state_world, ref_traj, T_actual)

        # Convert thrust to rotor speed
        rotor_speed = np.sqrt(np.clip(u_opt, 0, None) / self.C_T)

        return status, rotor_speed

    def _state_transform(self, state):
        """Transform state: body velocity to world velocity."""
        from utils.math_tool import quaternion_to_rotm

        state_new = state.copy()
        q = state[6:10]
        R = quaternion_to_rotm(q)
        v_body = state[3:6]
        v_world = R @ v_body
        state_new[3:6] = v_world
        return state_new

    def get_jerk_adapter(self):
        """Get jerk adapter for external access."""
        return self.jerk_adapter

    def reset(self):
        """Reset controller state."""
        self.u_prev = None
        self.T_actual_prev = None
        self.x_warm = None
        self.u_warm = None
        self.jerk_adapter.reset()
