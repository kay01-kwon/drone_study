"""
S550_3DOF_ocp_v2: NMPC with Angular Velocity and Roll/Pitch Feedback

Unlike S550_3DOF_ocp, this version:
1. Feedbacks angular velocity (q) and pitch (theta) trajectories
2. Uses dynamically changing max jerk for angular velocity based on DOB moment differentiation
3. Position/velocity trajectory uses Hehn trajectory with z-jerk based on rotor dynamics

Author: Geonwoo Kwon
Date: 2026-02-27
"""

from acados_template import AcadosOcp, AcadosOcpSolver
from control.nmpc.model.S550_3DOF_model import S550_3DOF_model
from utils.math_tool import pitch_to_rotm
from utils.low_pass_filter import LowPassFilter
from scipy.linalg import block_diag
import numpy as np


class ScaledTrajectory:
    """Time-scaled wrapper for HehnTrajectory."""
    def __init__(self, traj, time_scale):
        self._traj = traj
        self._s = time_scale

    def get_position(self, t):
        return self._traj.get_position(t / self._s)

    def get_velocity(self, t):
        return self._traj.get_velocity(t / self._s) / self._s

    def get_acceleration(self, t):
        return self._traj.get_acceleration(t / self._s) / (self._s ** 2)

    def get_jerk(self, t):
        return self._traj.get_jerk(t / self._s) / (self._s ** 3)

    @property
    def duration(self):
        return self._traj.duration * self._s


class AngularTrajectoryGenerator:
    """
    Angular trajectory generator using Hehn-style minimum-time profiles.
    Generates roll/pitch angle and angular velocity trajectories.
    """
    def __init__(self, j_max_default=50.0):
        self.j_max_default = j_max_default
        self.traj = None
        self.t_start = 0.0

    def generate(self, theta0, q0, theta_target, q_target, j_max=None):
        """
        Generate angular trajectory from current state to target.

        Args:
            theta0: Current angle [rad]
            q0: Current angular velocity [rad/s]
            theta_target: Target angle [rad]
            q_target: Target angular velocity [rad/s]
            j_max: Maximum angular jerk [rad/s^3]
        """
        from ref_generation.hehn_trajectory import AxisTrajectory

        if j_max is None:
            j_max = self.j_max_default

        # Error from target
        delta_theta = theta0 - theta_target
        delta_q = q0 - q_target

        # Acceleration limits for angular motion (conservative)
        a_max_ang = 10.0  # rad/s^2

        self.axis_traj = AxisTrajectory(
            w0=delta_theta, v0=delta_q, a0=0.0,
            a_lo=-a_max_ang, a_hi=a_max_ang, j_max=j_max
        )
        self.axis_traj.solve()
        self.theta_target = theta_target
        self.q_target = q_target
        self.t_start = 0.0

        return self

    def get_state(self, t):
        """
        Get angular state at time t.
        Returns: (theta_des, q_des)
        """
        if self.axis_traj is None:
            return 0.0, 0.0

        w, v, a, j = self.axis_traj.evaluate(t)
        theta_des = w + self.theta_target
        q_des = v + self.q_target

        return theta_des, q_des

    @property
    def duration(self):
        if self.axis_traj is None:
            return 0.0
        return self.axis_traj.tf


class S550_3DOF_ocp_v2:
    """
    NMPC OCP with angular velocity and roll/pitch feedback.

    Key features:
    - Position/velocity trajectory: Hehn trajectory with z-jerk = 3 * C_T * w_hover * alpha
    - Angular trajectory: feedback at landing, otherwise 0
    - Dynamic jerk limit from DOB moment differentiation
    """

    def __init__(self, DynParam=None, DroneParam=None, MpcParam=None, RotorParam=None):

        if DynParam is None:
            m = 3.0
            J = np.array([0.06, 0.06, 0.08])
            DynParam = {'m': m, 'MoiArray': J}

        if DroneParam is None:
            l = 0.265
            self.C_T = 1.465e-7
            k_m = 0.01569
            DroneParam = {'arm_length': l,
                          'motor_const': self.C_T,
                          'moment_const': k_m}
        else:
            self.C_T = DroneParam['motor_const']

        # Store parameters
        self.m = DynParam['m']
        self.Jyy = DynParam['MoiArray'][1]

        # Default rotor speed limits
        w_max = 7200.0
        w_min = 2000.0
        u_max = self.C_T * w_max**2
        u_min = self.C_T * w_min**2

        if MpcParam is None:
            t_horizon = 0.20
            n_nodes = 20
            Q = np.diag([1, 1,          # px, pz
                         0.5, 0.5,      # vx, vz
                         0.5,           # th
                         0.05])         # wy
            R = np.diag([0.01]*3)
        else:
            t_horizon = MpcParam['t_horizon']
            n_nodes = MpcParam['n_nodes']
            Q = np.diag(MpcParam['QArray'])
            R = MpcParam['R']*np.eye(3)

        self.ocp = AcadosOcp()

        # Instantiate model object
        model_obj = S550_3DOF_model(DynParam, DroneParam)
        acados_model = model_obj.export_acados_model()

        self.ocp.model = acados_model

        x0 = np.array([0.0, 0.0,      # x, z
                       0.0, 0.0,      # vx, vz
                       0.0,           # th
                       0.0])          # q

        # Dim info
        nx = acados_model.x.rows()
        nu = acados_model.u.rows()
        ny = nx + nu

        # 1. Cost setup
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        self.ocp.cost.Vx_e = np.eye(nx)

        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)

        self.ocp.cost.W = block_diag(Q, R)
        self.ocp.cost.W_e = Q

        self.ocp.cost.yref = np.concatenate((x0, np.zeros(nu)))
        self.ocp.cost.yref_e = x0

        # 2. Set ocp constraints
        self.ocp.constraints.x0 = x0
        self.ocp.constraints.lbu = np.array([u_min]*nu)
        self.ocp.constraints.ubu = np.array([u_max]*nu)
        self.ocp.constraints.idxbu = np.array([0, 1, 2])

        # 3. Set ocp solver
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 100
        self.ocp.solver_options.tf = t_horizon
        self.ocp.solver_options.N_horizon = n_nodes

        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '_v2.json'
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.solver_json)

        self.previous_states = None
        self.previous_u0 = None

        self.nx = nx
        self.nu = nu
        self.N = n_nodes
        self.T = t_horizon
        self.ref_nmpc = np.zeros((6,))

        # Hover thrust per group: mg/6 (hexarotor: 3 groups x 2 motors)
        self.u_hover = DynParam['m'] * 9.81 / 6.0

        # Hover rotor speed [rad/s]
        self.w_hover = np.sqrt(self.u_hover / self.C_T)

        # Trajectory generation
        self.traj_gen = None
        self.traj = None
        self.ang_traj_gen = AngularTrajectoryGenerator()
        self.target_2d = None
        self.time_scale = 1.0
        self.t_start = 0.0
        self.replan_interval = 0.01

        # Reference smoothing
        self.ref_smooth_alpha = 0.3
        self.p_des_smooth = None
        self.v_des_smooth = None

        # DOB moment differentiation for dynamic jerk limit
        self.tau_prev = 0.0
        self.tau_dot_lpf = LowPassFilter(cut_off_freq=10.0, dim=1)

        # Physical parameters for jerk computation
        # h_eff = 360 mm (effective height from contact point)
        self.h_eff = 0.360  # [m]
        self.x_g = abs(DynParam.get('com_offset', [0.0, 0.0, 0.0])[0])  # CoM offset in x
        self.g = 9.81

        # Compute J_p = m * (x_g^2 + h_eff^2)
        self.J_p = self.m * (self.x_g**2 + self.h_eff**2)

        # Rotor dynamics parameters
        # alpha = 10,000 RPM/s = 10000 * 2*pi/60 rad/s
        self.alpha_rotor = 10000.0 * 2.0 * np.pi / 60.0  # [rad/s^2]

        # z-axis max jerk: j_max_z = 3 * C_T * w_hover * alpha (for 3 rotor groups)
        # Actually: f_max = 6 * C_T * w_hover * alpha (total for hexarotor)
        self.f_max = 6.0 * self.C_T * self.w_hover * self.alpha_rotor
        self.j_max_z = self.f_max / self.m  # Convert force to acceleration jerk

        # Landing mode flag
        self.is_landing = False
        self.landing_start_time = None

    def setup_tracking(self, tracking_params, target_2d):
        """Initialize Hehn trajectory generator for tracking mode."""
        from ref_generation.hehn_trajectory import HehnTrajectoryGenerator, QuadParams

        qp = QuadParams(
            a_max=tracking_params['a_max'],
            a_min=tracking_params['a_min'],
            omega_xy_max=tracking_params['omega_xy_max']
        )
        self.traj_gen = HehnTrajectoryGenerator(qp=qp)
        self.target_2d = np.array(target_2d)
        self.target_3d = np.array([target_2d[0], 0.0, target_2d[1]])
        self.time_scale = tracking_params.get('time_scale', 1.0)
        replan_freq = tracking_params.get('replan_freq', 100.0)
        self.replan_interval = 1.0 / replan_freq

    def set_landing_mode(self, is_landing, t_now=None):
        """Set landing mode for angular velocity feedback."""
        self.is_landing = is_landing
        if is_landing and t_now is not None:
            self.landing_start_time = t_now

    def update_dob_moment(self, tau_est, dt):
        """
        Update DOB moment and compute moment rate via differentiation + LPF.

        Args:
            tau_est: Current DOB moment estimate [Nm]
            dt: Time step [s]

        Returns:
            tau_dot: Filtered moment rate [Nm/s]
        """
        if dt > 1e-6:
            tau_dot_raw = (tau_est - self.tau_prev) / dt
        else:
            tau_dot_raw = 0.0

        self.tau_prev = tau_est

        # Apply LPF to moment rate
        tau_dot_filtered = self.tau_dot_lpf.do_filter(np.array([tau_dot_raw]), dt)

        return tau_dot_filtered[0]

    def compute_angular_jerk_max(self, theta, q, tau_dot):
        """
        Compute maximum angular jerk based on DOB moment differentiation.

        Formula from image:
        M_dot_y_max = -x_g * f_max - m*g*h_eff*theta_dot*cos(theta) - m*g*x_g*theta_dot*sin(theta)
        omega_dot_y_max = (1/J_p) * |M_dot_y_max|

        Args:
            theta: Current pitch angle [rad]
            q: Current pitch rate [rad/s]
            tau_dot: Filtered moment rate from DOB [Nm/s]

        Returns:
            j_max_ang: Maximum angular jerk [rad/s^3]
        """
        # Compute M_dot_y_max from physical constraints
        M_dot_term1 = -self.x_g * self.f_max
        M_dot_term2 = -self.m * self.g * self.h_eff * q * np.cos(theta)
        M_dot_term3 = -self.m * self.g * self.x_g * q * np.sin(theta)

        M_dot_y_max = abs(M_dot_term1 + M_dot_term2 + M_dot_term3)

        # Also consider actual DOB moment rate
        M_dot_y_max = max(M_dot_y_max, abs(tau_dot))

        # Compute angular jerk limit
        j_max_ang = M_dot_y_max / self.J_p

        # Minimum jerk limit to avoid numerical issues
        j_max_ang = max(j_max_ang, 5.0)

        return j_max_ang

    def solve(self, state, ref, u_prev=None):
        """
        Solve OCP for regulation (fixed reference).
        """
        if u_prev is None:
            u_prev = np.array([self.u_hover]*self.nu)

        self.ref_nmpc[0:4] = ref[0:4]
        self.ref_nmpc[4] = 0.0   # th_des = 0
        self.ref_nmpc[5] = 0.0   # q_des = 0

        y_ref = np.concatenate((self.ref_nmpc, u_prev))
        y_ref_N = self.ref_nmpc

        state_transformed = self._state_transform(state)

        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        if self.previous_states is not None:
            for stage in range(self.N):
                if stage < self.N - 1:
                    prev_state = self.previous_states[stage + 1]
                    y_ref_warm = np.concatenate((prev_state, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
                else:
                    self.ocp_solver.set(stage, 'y_ref', y_ref)
            self.ocp_solver.set(self.N, 'y_ref', y_ref_N)
        else:
            for stage in range(self.N):
                self.ocp_solver.set(stage, 'y_ref', y_ref)
            self.ocp_solver.set(self.N, 'y_ref', y_ref_N)

        status = self.ocp_solver.solve()

        self.previous_states = []
        for stage in range(self.N + 1):
            self.previous_states.append(self.ocp_solver.get(stage, 'x').copy())

        u = self.ocp_solver.get(0, 'u')
        self.previous_u0 = u.copy()
        w_cmd = np.sqrt(u / self.C_T)

        return status, w_cmd

    def solve_for_trajectory(self, state, state_2d, t_now, tau_est=0.0, dt=0.01, u_prev=None):
        """
        Solve OCP for trajectory tracking with angular velocity feedback.

        Key difference from S550_3DOF_ocp:
        - At landing: feedback angular velocity trajectory
        - Otherwise: feedback th_des=0, q_des=0

        Args:
            state: [px, pz, vx_body, vz_body, th, q]
            state_2d: [px, pz, vx_world, vz_world] for trajectory generation
            t_now: Current simulation time
            tau_est: DOB moment estimate for dynamic jerk computation
            dt: Time step for moment differentiation
            u_prev: previous thrust [u1, u2, u3]

        Returns: (status, w_cmd, p_des, v_des, th_des, q_des)
        """
        if u_prev is None:
            u_prev = np.array([self.u_hover]*self.nu)

        u_ref = self.previous_u0 if self.previous_u0 is not None else u_prev

        # Update DOB moment rate
        tau_dot = self.update_dob_moment(tau_est, dt)

        # Compute dynamic angular jerk limit
        theta = state[4]
        q = state[5]
        j_max_ang = self.compute_angular_jerk_max(theta, q, tau_dot)

        # Replan position trajectory
        need_replan = self.traj is None
        if not need_replan and (t_now - self.t_start) >= self.replan_interval:
            need_replan = True

        if need_replan:
            self._generate_trajectory_simple(state_2d, t_now)

        t_rel = t_now - self.t_start
        dt_horizon = self.T / self.N

        state_transformed = self._state_transform(state)

        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        # Determine angular velocity reference based on landing mode
        th_des = 0.0
        q_des = 0.0

        if self.is_landing:
            # At landing: generate angular velocity trajectory
            # Generate trajectory from current state to upright (th=0, q=0)
            self.ang_traj_gen.generate(
                theta0=theta, q0=q,
                theta_target=0.0, q_target=0.0,
                j_max=j_max_ang
            )

        # Set reference for each stage
        for stage in range(self.N):
            t_ref = t_rel + stage * dt_horizon
            ref_stage = self._traj_to_ref(t_ref)

            # Angular reference
            if self.is_landing:
                th_stage, q_stage = self.ang_traj_gen.get_state(stage * dt_horizon)
                ref_stage[4] = th_stage
                ref_stage[5] = q_stage
            else:
                ref_stage[4] = 0.0  # th_des = 0
                ref_stage[5] = 0.0  # q_des = 0

            y_ref = np.concatenate((ref_stage, u_ref))
            self.ocp_solver.set(stage, 'y_ref', y_ref)

        # Terminal reference
        t_ref_N = t_rel + self.T
        ref_N = self._traj_to_ref(t_ref_N)
        if self.is_landing:
            th_N, q_N = self.ang_traj_gen.get_state(self.T)
            ref_N[4] = th_N
            ref_N[5] = q_N
        else:
            ref_N[4] = 0.0
            ref_N[5] = 0.0
        self.ocp_solver.set(self.N, 'y_ref', ref_N)

        status = self.ocp_solver.solve()

        # Get current references for logging
        p_des, v_des = self._get_reference(t_rel)
        if self.is_landing:
            th_des, q_des = self.ang_traj_gen.get_state(0.0)
        else:
            th_des, q_des = 0.0, 0.0

        # Store for warm start
        self.previous_states = []
        for stage in range(self.N + 1):
            self.previous_states.append(self.ocp_solver.get(stage, 'x').copy())

        u = self.ocp_solver.get(0, 'u')
        self.previous_u0 = u.copy()
        w_cmd = np.sqrt(u / self.C_T)

        return status, w_cmd, p_des, v_des, th_des, q_des

    def _generate_trajectory_simple(self, state_2d, t_now):
        """Generate Hehn trajectory with custom z-jerk limit."""
        pos0 = np.array([state_2d[0], 0.0, state_2d[1]])
        vel0 = np.array([state_2d[2], 0.0, state_2d[3]])
        acc0 = np.zeros(3)

        # Use custom j_max for z-axis based on rotor dynamics
        traj_raw = self.traj_gen.generate(
            pos0, vel0, acc0,
            target=self.target_3d,
            j_max=self.j_max_z
        )

        if self.time_scale != 1.0:
            self.traj = ScaledTrajectory(traj_raw, self.time_scale)
        else:
            self.traj = traj_raw

        self.t_start = t_now

    def _traj_to_ref(self, t):
        """Get NMPC reference from trajectory."""
        pos = self.traj.get_position(t)
        vel = self.traj.get_velocity(t)

        ref = np.zeros(6)
        ref[0] = pos[0]    # px_des
        ref[1] = pos[2]    # pz_des
        ref[2] = vel[0]    # vx_des
        ref[3] = vel[2]    # vz_des
        ref[4] = 0.0       # th_des (will be overwritten if landing)
        ref[5] = 0.0       # q_des (will be overwritten if landing)
        return ref

    def _get_reference(self, t_rel):
        """Get p_des, v_des for logging."""
        pos = self.traj.get_position(t_rel)
        vel = self.traj.get_velocity(t_rel)
        p_des = np.array([pos[0], pos[2]])
        v_des = np.array([vel[0], vel[2]])
        return p_des, v_des

    def _state_transform(self, state):
        """Transform body velocity to world velocity for NMPC."""
        th = state[4]
        R = pitch_to_rotm(th)
        v_body = state[2:4]
        v_world = R @ v_body
        state_new = state.copy()
        state_new[2:4] = v_world
        return state_new

    def get_json_file_name(self):
        return self.solver_json

    def get_jerk_limits(self):
        """Return current jerk limits for debugging."""
        return {
            'j_max_z': self.j_max_z,
            'J_p': self.J_p,
            'f_max': self.f_max,
            'w_hover': self.w_hover,
            'alpha_rotor': self.alpha_rotor
        }
