from acados_template import AcadosOcp, AcadosOcpSolver
from control.nmpc.model.S550_3DOF_model import S550_3DOF_model
from utils.math_tool import pitch_to_rotm
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

    @property
    def duration(self):
        return self._traj.duration * self._s

class S550_3DOF_ocp:
    def __init__(self, DynParam = None,
                 DroneParam = None,
                 MpcParam = None):

        if DynParam is None:
            m = 3.0
            J = np.array([0.06, 0.06, 0.08])
            DynParam = {'m': m, 'MoiArray': J}

        if DroneParam is None:
            l = 0.265
            self.C_T = 1.465e-7
            k_m = 0.01569
            DroneParam = {'arm_length':l,
                          'motor_const':self.C_T,
                          'moment_const':k_m}
        else:
            self.C_T = DroneParam['motor_const']

        # Default rotor speed limits
        w_max = 7200.0
        w_min = 2000.0
        u_max = self.C_T * w_max**2
        u_min = self.C_T * w_min**2

        if MpcParam is None:
            t_horizon = 0.20                # Time horizon
            n_nodes = 20                    # Number of nodes
            Q = np.diag([1, 1,          # px, pz
                        0.5, 0.5,       # vx, vz
                        0.5,            # th
                        0.05])          # wy
            R = np.diag([0.01]*3)           # u1...u3

        else:
            t_horizon = MpcParam['t_horizon']
            n_nodes = MpcParam['n_nodes']
            Q = np.diag(MpcParam['QArray'])
            R = MpcParam['R']*np.eye(3)


        self.ocp = AcadosOcp()

        # Instantiate model object
        model_obj = S550_3DOF_model(DynParam, DroneParam)
        acados_model = model_obj.export_acados_model()

        # Put acados model into ocp model
        self.ocp.model = acados_model

        x0 = np.array(
            [0.0, 0.0,      # x, z
             0.0, 0.0,      # vx, vz
             0.0,           # th
             0.0]           # q
        )

        # Dim info
        nx = acados_model.x.rows()
        nu = acados_model.u.rows()
        ny = nx + nu

        # 1. Cost setup

        # 1.1 Declare type of cost
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # 1.2 Vx setup
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        self.ocp.cost.Vx_e = np.eye(nx)

        # 1.3 Vu setup
        self.ocp.cost.Vu = np.zeros((ny, nu))
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
        self.ocp.constraints.idxbu = np.array([0, 1, 2])

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
        self.ocp.solver_options.tf = t_horizon
        self.ocp.solver_options.N_horizon = n_nodes

        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.solver_json)
        # Store state trajectory for warm start
        self.previous_states = None

        self.nx = nx
        self.nu = nu
        self.N = n_nodes
        self.T = t_horizon
        self.ref_nmpc = np.zeros((6,))

        # Hover thrust per group: mg/6 (hexarotor: 3 groups × 2 motors)
        self.u_hover = DynParam['m'] * 9.81 / 6.0

        # Trajectory generation (initialized via setup_tracking)
        self.traj_gen = None
        self.traj = None
        self.target_2d = None
        self.time_scale = 1.0
        self.t_start = 0.0
        self.replan_interval = 0.01  # 100ms (100Hz)

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

    def solve(self, state, ref, u_prev=None):
        """
        Solve OCP for regulation (fixed reference).
        :param state: [px, pz, vx_body, vz_body, th, q]
        :param ref: [px_des, pz_des, vx_des, vz_des]
        :param u_prev: previous thrust [u1, u2, u3] (for warm start reference)
        :return: (status, w_cmd)
        """
        if u_prev is None:
            u_prev = np.array([self.u_hover]*self.nu)

        # Build state reference: position + velocity desired, th=0, q=0
        self.ref_nmpc[0:4] = ref[0:4]
        self.ref_nmpc[4] = 0.0   # th_des = 0
        self.ref_nmpc[5] = 0.0   # q_des = 0

        y_ref = np.concatenate((self.ref_nmpc, u_prev))
        y_ref_N = self.ref_nmpc

        # Transform body velocity to world velocity
        state_transformed = self._state_transform(state)

        # Set initial state constraint
        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        # Warm start with previous trajectory or constant reference
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

        # Store state trajectory for warm start
        self.previous_states = []
        for stage in range(self.N + 1):
            self.previous_states.append(self.ocp_solver.get(stage, 'x').copy())

        u = self.ocp_solver.get(0, 'u')
        w_cmd = np.sqrt(u / self.C_T)

        return status, w_cmd

    def solve_for_trajectory(self, state, state_2d, t_now, u_prev=None):
        """
        Solve OCP for trajectory tracking with internal trajectory generation.
        :param state: [px, pz, vx_body, vz_body, th, q]
        :param state_2d: [px, pz, vx_world, vz_world] for trajectory generation
        :param t_now: Current simulation time
        :param u_prev: previous thrust [u1, u2, u3]
        :return: (status, w_cmd, p_des, v_des)
        """
        if u_prev is None:
            u_prev = np.array([self.u_hover]*self.nu)

        # Replan if interval elapsed
        if (t_now - self.t_start) >= self.replan_interval or self.traj is None:
            self._generate_trajectory(state_2d, t_now)

        t_rel = t_now - self.t_start
        dt = self.T / self.N

        # Transform body velocity to world velocity
        state_transformed = self._state_transform(state)

        # Set initial state constraint
        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        # Set reference for each stage along prediction horizon
        for stage in range(self.N):
            t_ref = t_rel + stage * dt
            ref_stage = self._traj_to_ref(t_ref)
            y_ref = np.concatenate((ref_stage, u_prev))
            self.ocp_solver.set(stage, 'y_ref', y_ref)

        # Terminal reference
        t_ref_N = t_rel + self.T
        ref_N = self._traj_to_ref(t_ref_N)
        self.ocp_solver.set(self.N, 'y_ref', ref_N)

        status = self.ocp_solver.solve()

        # Get p_des, v_des at current time for logging
        p_des, v_des = self._get_reference(t_rel)

        # Store state trajectory for warm start
        self.previous_states = []
        for stage in range(self.N + 1):
            self.previous_states.append(self.ocp_solver.get(stage, 'x').copy())

        u = self.ocp_solver.get(0, 'u')
        w_cmd = np.sqrt(u / self.C_T)

        return status, w_cmd, p_des, v_des

    def _generate_trajectory(self, state_2d, t_now):
        """Generate Hehn trajectory from current position to target.

        Uses HehnTrajectory directly: pos0 -> target.
        get_position(t) returns desired position, get_velocity(t) returns
        desired velocity. No sign conversions needed.

        Note: acc0 is NOT passed to the trajectory generator because the
        model-based acceleration estimate (from rotor thrust) can be wildly
        wrong on the ground (idle rotors give acc ≈ -8.6 m/s² which is
        outside the Hehn constraint bounds and causes the trajectory to go
        in the wrong direction).
        """
        pos0 = np.array([state_2d[0], 0.0, state_2d[1]])
        vel0 = np.array([state_2d[2], 0.0, state_2d[3]])
        acc0 = np.zeros(3)

        traj_raw = self.traj_gen.generate(pos0, vel0, acc0, target=self.target_3d)

        if self.time_scale != 1.0:
            self.traj = ScaledTrajectory(traj_raw, self.time_scale)
        else:
            self.traj = traj_raw

        self.t_start = t_now

    def _traj_to_ref(self, t):
        """Get NMPC reference directly from trajectory.
        get_position(t) = desired position, get_velocity(t) = desired velocity.
        """
        pos = self.traj.get_position(t)
        vel = self.traj.get_velocity(t)

        ref = np.zeros(6)
        ref[0] = pos[0]    # px_des
        ref[1] = pos[2]    # pz_des (3D z-axis -> 2D z)
        ref[2] = vel[0]    # vx_des
        ref[3] = vel[2]    # vz_des
        ref[4] = 0.0       # th_des
        ref[5] = 0.0       # q_des
        return ref

    def _get_reference(self, t_rel):
        """Get p_des, v_des for logging at current time."""
        pos = self.traj.get_position(t_rel)
        vel = self.traj.get_velocity(t_rel)
        p_des = np.array([pos[0], pos[2]])
        v_des = np.array([vel[0], vel[2]])
        return p_des, v_des

    def _state_transform(self, state):
        """Transform body velocity to world velocity for NMPC.
        Input state:  [px, pz, vx_body, vz_body, th, q]
        Output state: [px, pz, vx_world, vz_world, th, q]
        """
        from utils.math_tool import pitch_to_rotm
        th = state[4]
        R = pitch_to_rotm(th)
        v_body = state[2:4]
        v_world = R @ v_body
        state_new = state.copy()
        state_new[2:4] = v_world
        return state_new

    def get_json_file_name(self):
        return self.solver_json