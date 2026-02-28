from acados_template import AcadosOcp, AcadosOcpSolver
from control.nmpc.model.S550_model import S550_model
from utils.math_tool import quaternion_to_rotm
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

class S550Ocp:
    def __init__(self, DynParam = None, DroneParam = None, MpcParam = None):
        '''
        Constructor
        :param DynParam: m, MoiArray (Jxx, Jyy, Jzz)
        :param DroneParam: arm_length, rotor_const, moment_const, T_max, T_min
        :param MpcParam: t_horizon, n_nodes, QArray, R
        '''
        if DynParam is None:
            m = 3.0
            J = np.array([0.06, 0.06, 0.08])
            DynParam = {'m': m, 'MoiArray': J}

        u_max = 1.4765e-7 * 7300.0**2
        u_min = 1.4765e-7 * 2000.0**2

        if DroneParam is None:
            l = 0.265
            self.C_T = 1.465e-7
            k_m = 0.01569
            DroneParam = {'arm_length':l,
                          'motor_const':self.C_T,
                          'moment_const':k_m}
        else:
            self.C_T = DroneParam['motor_const']
            u_max = self.C_T * 7300.0**2
            u_min = self.C_T * 2000.0**2

        if MpcParam is None:
            t_horizon = 0.20                # Time horizon
            n_nodes = 20                    # Number of nodes
            Q = np.diag([1, 1, 1,           # px, py, pz
                        0.5, 0.5, 0.5,      # vx, vy, vz
                        0.5, 0.5, 0.5, 0.5,   # qw, qx, qy, qz
                        0.05, 0.05, 0.05])  #wx, wy ,wz
            R = np.diag([0.01]*6)           # u1...u6

        else:
            t_horizon = MpcParam['t_horizon']
            n_nodes = MpcParam['n_nodes']
            Q = np.diag(MpcParam['QArray'])
            R = MpcParam['R']*np.eye(6)


        self.ocp = AcadosOcp()

        # Instantiate model object
        model_obj = S550_model(DynParam, DroneParam)
        acados_model = model_obj.export_acados_model()

        # Put acados model into ocp model
        self.ocp.model = acados_model

        x0 = np.array([
            0.0, 0.0, 0.0,          # px, py, pz
            0.0, 0.0, 0.0,          # vx, vy, vz
            1.0, 0.0, 0.0, 0.0,     # qw, qx, qy, qz
            0.0, 0.0, 0.0           # wx, wy, wz
        ])

        # Dim info
        nx = acados_model.x.rows()
        nu = acados_model.u.rows()
        ny = nx + nu

        # 1. Cost setup

        # 1.1 Declare type of cost
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # 1.2 Vx setup
        self.ocp.cost.Vx = np.zeros((ny,nx))
        self.ocp.cost.Vx[:nx,:nx] = np.eye(nx)
        self.ocp.cost.Vx_e = np.eye(nx)

        # 1.3 Vu setup
        self.ocp.cost.Vu = np.zeros((ny,nu))
        self.ocp.cost.Vu[-nu:,-nu:] = np.eye(nu)

        # 1.4 Weight setup
        self.ocp.cost.W = block_diag(Q, R)
        self.ocp.cost.W_e = Q

        # 1.5 Reference setup
        self.ocp.cost.yref = np.concatenate((x0, np.zeros(nu)))
        self.ocp.cost.yref_e = x0

        # 2. Set ocp constraints
        self.ocp.constraints.x0 = x0
        self.ocp.constraints.lbu = np.array([u_min]*nu)
        self.ocp.constraints.ubu = np.array([u_max]*nu)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # 3. Set ocp solver
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4       # RK4
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.print_level = 0                 # 0 : Do not print
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 100
        # self.ocp.solver_options.qp_solver_cond_N = n_nodes
        # self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.regularize_method = 'PROJECT'
        self.ocp.solver_options.tf = t_horizon
        self.ocp.solver_options.N_horizon = n_nodes

        # self.ocp_solver = AcadosOcpSolver(self.ocp)
        # generate json file and generate cython
        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        AcadosOcpSolver.generate(self.ocp, json_file = self.solver_json)
        AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        self.ocp_solver = AcadosOcpSolver.create_cython_solver(self.solver_json)
        # Store state trajectory for warm start
        self.previous_states = None
        self.previous_u0 = None

        self.nx = nx
        self.nu = nu
        self.N = n_nodes
        self.T = t_horizon

        self.ref_nmpc = np.zeros((13,))
        self.ref_nmpc[6] = 1.0

        # Hover thrust per rotor: mg/6
        self.u_hover = DynParam['m'] * 9.81 / 6.0

        # Trajectory generation (initialized via setup_tracking)
        self.traj_gen = None
        self.traj = None
        self.target_3d = None
        self.time_scale = 1.0
        self.t_start = 0.0
        self.replan_interval = 0.01  # Default: 10ms (100Hz)

    def setup_tracking(self, tracking_params, target_3d):
        """Initialize Hehn trajectory generator for tracking mode."""
        from ref_generation.hehn_trajectory import HehnTrajectoryGenerator, QuadParams

        qp = QuadParams(
            a_max=tracking_params['a_max'],
            a_min=tracking_params['a_min'],
            omega_xy_max=tracking_params['omega_xy_max']
        )
        self.traj_gen = HehnTrajectoryGenerator(qp=qp)
        self.target_3d = np.array(target_3d)
        self.time_scale = tracking_params.get('time_scale', 1.0)
        replan_freq = tracking_params.get('replan_freq', 100.0)
        self.replan_interval = 1.0 / replan_freq

    def solve(self, state, ref, u_prev=None):
        '''
        Solve OCP problem with warm start
        :param state: p (World), v (Body), q, w (Body)
        :param ref: p (World), v (World), yaw_des, dot_yaw_des
        :param u_prev: w_cmd1...w_cmd6 (Rotor speed)
        :return: status, w_cmd(w_cmd1...w_cmd6)
        '''

        if u_prev is None:
            u_prev = np.zeros((6,))

        self.ref_nmpc[0:6] = ref[0:6]
        self.ref_nmpc[6] = np.cos(ref[6]/2.0)
        self.ref_nmpc[9] = np.sin(ref[6]/2.0)
        self.ref_nmpc[12] = ref[7]

        y_ref = np.concatenate((self.ref_nmpc,u_prev))
        y_ref_N = self.ref_nmpc

        # Transform linear velocity from body to world
        state_transformed = self.state_transform(state)

        # Set constraint at the first stage
        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        # Warm start: Use previous state trajectory as reference
        if self.previous_states is not None:
            # Shift previous trajectory forward by one step
            for stage in range(self.ocp.solver_options.N_horizon):
                if stage < self.ocp.solver_options.N_horizon - 1:
                    # Use next state from previous trajectory
                    prev_state = self.previous_states[stage + 1]
                    y_ref_warm = np.concatenate((prev_state, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
                else:
                    # Last stage: use terminal reference
                    y_ref_warm = np.concatenate((self.ref_nmpc, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)

            # Set terminal reference
            self.ocp_solver.set(self.ocp.solver_options.N_horizon, 'y_ref', self.ref_nmpc)
        else:
            # First solve: use constant reference
            for stage in range(self.ocp.solver_options.N_horizon):
                self.ocp_solver.set(stage, 'y_ref', y_ref)

            # Set y ref at the terminal stage
            self.ocp_solver.set(self.ocp.solver_options.N_horizon, 'y_ref', y_ref_N)

        status = self.ocp_solver.solve()

        # Store current state trajectory for next iteration
        N = self.ocp.solver_options.N_horizon
        self.previous_states = []
        for stage in range(N + 1):
            x_stage = self.ocp_solver.get(stage, 'x')
            self.previous_states.append(x_stage.copy())

        # Get control input at the first stage
        u = self.ocp_solver.get(0, 'u')

        rotor_speed = np.sqrt(u/self.C_T)

        return status, rotor_speed

    def solve_for_trajectory(self, state, t_curr, u_prev=None):
        '''
        Solve OCP problem with trajectory tracking along prediction horizon
        :param state: p (World), v (Body), q, w (Body)
        :param t_curr: Current time
        :param u_prev: u1...u6 (Rotor thrust)
        :return: status, w_cmd(w_cmd1...w_cmd6)
        '''
        from ref_generation.ref_generator import get_reference

        if u_prev is None:
            u_prev = np.zeros((6,))

        # Get time step for prediction horizon
        N = self.ocp.solver_options.N_horizon
        T = self.ocp.solver_options.tf
        dt = T / N

        # Transform linear velocity from body to world
        state_transformed = self.state_transform(state)

        # Set constraint at the first stage
        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        # Set reference for each stage along the prediction horizon
        for stage in range(N):
            # Get reference at future time
            t_ref = t_curr + stage * dt
            ref = get_reference(t_ref)

            # Convert to NMPC reference format
            self.ref_nmpc[0:6] = ref[0:6]
            self.ref_nmpc[6] = np.cos(ref[6]/2.0)
            self.ref_nmpc[9] = np.sin(ref[6]/2.0)
            self.ref_nmpc[12] = ref[7]

            y_ref = np.concatenate((self.ref_nmpc, u_prev))
            self.ocp_solver.set(stage, 'y_ref', y_ref)

        # Set terminal reference
        t_ref_N = t_curr + T
        ref_N = get_reference(t_ref_N)
        ref_nmpc_N = np.zeros((13,))
        ref_nmpc_N[0:6] = ref_N[0:6]
        ref_nmpc_N[6] = np.cos(ref_N[6]/2.0)
        ref_nmpc_N[9] = np.sin(ref_N[6]/2.0)
        ref_nmpc_N[12] = ref_N[7]

        self.ocp_solver.set(N, 'y_ref', ref_nmpc_N)

        # Solve OCP
        status = self.ocp_solver.solve()

        # Store current state trajectory for warm start
        self.previous_states = []
        for stage in range(N + 1):
            x_stage = self.ocp_solver.get(stage, 'x')
            self.previous_states.append(x_stage.copy())

        # Get control input at the first stage
        u = self.ocp_solver.get(0, 'u')

        rotor_speed = np.sqrt(u/self.C_T)

        return status, rotor_speed

    def state_transform(self, state):
        """
        Transform linear velocity from body to world frame
        :param state: W_p, B_v, q_B_W, B_w
        :return: W_p, W_v, q_B_W, B_w
        """
        q = state[6:10]
        R_B_W = quaternion_to_rotm(q)
        v_body = state[3:6]
        v_world = R_B_W @ v_body
        state_new = state.copy()
        state_new[3:6] = v_world
        return state_new

    def solve_for_hehn_trajectory(self, state, state_3d, t_now, u_prev=None):
        """
        Solve OCP for trajectory tracking with Hehn trajectory generation.
        :param state: [p(3), v_body(3), q(4), w(3)] - 13 dim
        :param state_3d: [px, py, pz, vx_world, vy_world, vz_world] for trajectory
        :param t_now: Current simulation time
        :param u_prev: previous thrust [u1...u6]
        :return: (status, w_cmd, p_des, v_des)
        """
        if u_prev is None:
            u_prev = np.array([self.u_hover] * self.nu)

        # Use previous optimal control for reference if available
        u_ref = self.previous_u0 if self.previous_u0 is not None else u_prev

        # Simple replanning: replan every interval
        need_replan = self.traj is None
        if not need_replan and (t_now - self.t_start) >= self.replan_interval:
            need_replan = True

        if need_replan:
            self._generate_trajectory(state_3d, t_now)

        t_rel = t_now - self.t_start
        dt = self.T / self.N

        # Transform body velocity to world velocity
        state_transformed = self.state_transform(state)

        # Set initial state constraint
        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        # Set reference for each stage along prediction horizon
        for stage in range(self.N):
            t_ref = t_rel + stage * dt
            ref_stage = self._traj_to_ref(t_ref)
            y_ref = np.concatenate((ref_stage, u_ref))
            self.ocp_solver.set(stage, 'y_ref', y_ref)

        # Terminal reference
        t_ref_N = t_rel + self.T
        ref_N = self._traj_to_ref(t_ref_N)
        self.ocp_solver.set(self.N, 'y_ref', ref_N)

        status = self.ocp_solver.solve()

        # Get p_des, v_des at current time for logging
        p_des, v_des = self._get_reference(t_rel)

        # Store state trajectory and control for warm start
        self.previous_states = []
        for stage in range(self.N + 1):
            self.previous_states.append(self.ocp_solver.get(stage, 'x').copy())

        u = self.ocp_solver.get(0, 'u')
        self.previous_u0 = u.copy()
        w_cmd = np.sqrt(u / self.C_T)

        return status, w_cmd, p_des, v_des

    def _generate_trajectory(self, state_3d, t_now):
        """Generate Hehn trajectory from current state to target."""
        pos0 = np.array([state_3d[0], state_3d[1], state_3d[2]])
        vel0 = np.array([state_3d[3], state_3d[4], state_3d[5]])
        acc0 = np.zeros(3)

        traj_raw = self.traj_gen.generate(pos0, vel0, acc0, target=self.target_3d)

        if self.time_scale != 1.0:
            self.traj = ScaledTrajectory(traj_raw, self.time_scale)
        else:
            self.traj = traj_raw

        self.t_start = t_now

    def _traj_to_ref(self, t):
        """Get NMPC reference from trajectory."""
        pos = self.traj.get_position(t)
        vel = self.traj.get_velocity(t)

        ref = np.zeros(13)
        ref[0:3] = pos        # px, py, pz
        ref[3:6] = vel        # vx, vy, vz
        ref[6] = 1.0          # qw = 1 (identity quaternion, no rotation)
        ref[7:10] = 0.0       # qx, qy, qz = 0
        ref[10:13] = 0.0      # wx, wy, wz = 0
        return ref

    def _get_reference(self, t_rel):
        """Get p_des, v_des for logging at current time."""
        pos = self.traj.get_position(t_rel)
        vel = self.traj.get_velocity(t_rel)
        return pos, vel

    def get_json_file_name(self):
        return self.solver_json