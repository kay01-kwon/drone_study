from acados_template import AcadosOcp, AcadosOcpSolver
from control.nmpc.model.S550_actuator_model import S550ActuatorModel
from utils.math_tool import quaternion_to_rotm
from scipy.linalg import block_diag
import casadi as cs
import numpy as np


class S550ActuatorOcp:
    def __init__(self, DynParam=None, DroneParam=None,
                 MpcParam=None, ActuatorParam=None):
        '''
        NMPC with 2nd-order actuator dynamics and saturation.

        State dim: 25 = [p(3), v(3), q(4), w(3), w_rot(6), alpha_rot(6)]
        Control dim: 6 = [w_cmd_1 ... w_cmd_6] (commanded rotor speeds)

        :param DynParam: m, MoiArray (Jxx, Jyy, Jzz)
        :param DroneParam: arm_length, motor_const, moment_const
        :param MpcParam: t_horizon, n_nodes, QArray, Q_w_rot, Q_alpha_rot, RArray
        :param ActuatorParam: p1, p2, p3, alpha_max, j_max
        '''
        if DynParam is None:
            m = 2.9
            J = np.array([0.06, 0.06, 0.08])
            DynParam = {'m': m, 'MoiArray': J}
        else:
            m = DynParam['m']

        if DroneParam is None:
            l = 0.265
            C_T = 1.465e-7
            k_m = 0.01569
            DroneParam = {'arm_length': l,
                          'motor_const': C_T,
                          'moment_const': k_m}

        self.C_T = DroneParam['motor_const']

        # Rotor speed limits
        self.w_rot_max = 7300.0   # rad/s
        self.w_rot_min = 2000.0   # rad/s

        # Rotor acceleration limit
        if ActuatorParam is not None:
            self.alpha_rot_max = ActuatorParam['alpha_max']
        else:
            self.alpha_rot_max = 15000.0

        if MpcParam is None:
            t_horizon = 0.20
            n_nodes = 20
            Q_rigid = np.diag([
                1.0, 1.0, 1.0,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5,
                0.05, 0.05, 0.05,
            ])
            Q_w_rot = 0.0
            Q_alpha_rot = 0.0
            R = np.diag([0.01] * 6)
        else:
            t_horizon = MpcParam['t_horizon']
            n_nodes = MpcParam['n_nodes']
            Q_rigid = np.diag(MpcParam['QArray'])   # 13x13
            Q_w_rot = MpcParam.get('Q_w_rot', 0.0)
            Q_alpha_rot = MpcParam.get('Q_alpha_rot', 0.0)
            R = MpcParam['RArray'][0] * np.eye(6)

        # Build full 25x25 Q from rigid(13) + w_rot(6) + alpha_rot(6)
        Q = block_diag(Q_rigid,
                        Q_w_rot * np.eye(6),
                        Q_alpha_rot * np.eye(6))

        self.ocp = AcadosOcp()

        # Build model
        model_obj = S550ActuatorModel(DynParam, DroneParam, ActuatorParam)
        acados_model = model_obj.export_acados_model()
        self.ocp.model = acados_model

        nx = acados_model.x.rows()   # 25
        nu = acados_model.u.rows()   # 6
        ny = nx + nu                 # 31
        self.nx = nx
        self.nu = nu

        # Parameter dimension
        self.n_params = acados_model.p.rows()
        self.p_default = np.array([m, 0.0, 0.0])

        # Hover rotor speed: mg = 6 * C_T * w_hover^2
        self.w_hover = np.sqrt(m * 9.81 / (6.0 * self.C_T))

        # Initial state (hover equilibrium)
        x0 = np.zeros(nx)
        x0[6] = 1.0                          # qw = 1
        x0[13:19] = self.w_hover              # w_rot at hover

        # Hover control input
        u_hover = np.ones(nu) * self.w_hover

        # ============================================================
        # 1. Cost setup (LINEAR_LS)
        # ============================================================
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        self.ocp.cost.Vx_e = np.eye(nx)

        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)

        self.ocp.cost.W = block_diag(Q, R)
        self.ocp.cost.W_e = Q

        # Use hover equilibrium as initial reference (u_ref = w_hover, NOT 0)
        self.ocp.cost.yref = np.concatenate((x0, u_hover))
        self.ocp.cost.yref_e = x0

        # ============================================================
        # 2. Constraints
        # ============================================================
        self.ocp.constraints.x0 = x0

        # Control input bounds (commanded rotor speed)
        self.ocp.constraints.lbu = np.array([self.w_rot_min] * nu)
        self.ocp.constraints.ubu = np.array([self.w_rot_max] * nu)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # Path constraints: h(x,u) = [w_rot(6), alpha_rot(6)]
        # w_rot_min  <= w_rot_i  <= w_rot_max
        # -alpha_max <= alpha_i  <= alpha_max
        model_x = acados_model.x
        h_expr = cs.vertcat(model_x[13:19], model_x[19:25])

        self.ocp.model.con_h_expr = h_expr

        nh = 12  # 6 w_rot + 6 alpha_rot
        self.ocp.constraints.lh = np.concatenate([
            np.array([self.w_rot_min] * 6),
            np.array([-self.alpha_rot_max] * 6)
        ])
        self.ocp.constraints.uh = np.concatenate([
            np.array([self.w_rot_max] * 6),
            np.array([self.alpha_rot_max] * 6)
        ])

        # Soft constraint slack penalties (L1 + L2)
        self.ocp.cost.zl = 100.0 * np.ones(nh)
        self.ocp.cost.zu = 100.0 * np.ones(nh)
        self.ocp.cost.Zl = 100.0 * np.ones(nh)
        self.ocp.cost.Zu = 100.0 * np.ones(nh)
        self.ocp.constraints.lsh = np.zeros(nh)
        self.ocp.constraints.ush = np.zeros(nh)
        self.ocp.constraints.idxsh = np.arange(nh)

        # Terminal path constraints (same)
        self.ocp.model.con_h_expr_e = h_expr
        self.ocp.constraints.lh_e = self.ocp.constraints.lh.copy()
        self.ocp.constraints.uh_e = self.ocp.constraints.uh.copy()

        self.ocp.cost.zl_e = 100.0 * np.ones(nh)
        self.ocp.cost.zu_e = 100.0 * np.ones(nh)
        self.ocp.cost.Zl_e = 100.0 * np.ones(nh)
        self.ocp.cost.Zu_e = 100.0 * np.ones(nh)
        self.ocp.constraints.lsh_e = np.zeros(nh)
        self.ocp.constraints.ush_e = np.zeros(nh)
        self.ocp.constraints.idxsh_e = np.arange(nh)

        # ============================================================
        # 3. Solver options
        # ============================================================
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.levenberg_marquardt = 1e-1
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4    # RK4
        self.ocp.solver_options.sim_method_num_steps = 4     # Multiple steps for fast actuator dynamics
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 100
        self.ocp.solver_options.globalization = 'FIXED_STEP'
        self.ocp.solver_options.tf = t_horizon
        self.ocp.solver_options.N_horizon = n_nodes

        # Set initial parameter values
        self.ocp.parameter_values = self.p_default

        # Build solver
        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        AcadosOcpSolver.generate(self.ocp, json_file=self.solver_json)
        AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        self.ocp_solver = AcadosOcpSolver.create_cython_solver(self.solver_json)

        # ============================================================
        # 4. Initialize solver trajectory at hover equilibrium
        # ============================================================
        N = n_nodes
        for stage in range(N):
            self.ocp_solver.set(stage, 'x', x0)
            self.ocp_solver.set(stage, 'u', u_hover)
        self.ocp_solver.set(N, 'x', x0)

        self.previous_states = None

        # Reference for the rigid-body part (25 dim)
        self.ref_nmpc = np.zeros(nx)
        self.ref_nmpc[6] = 1.0                # qw = 1
        self.ref_nmpc[13:19] = self.w_hover    # w_rot hover reference

        print(f"S550ActuatorOcp created (nx={nx}, nu={nu}, params={self.n_params})")

    def solve(self, state, ref, param_est=None, u_prev=None):
        '''
        Solve OCP.
        :param state: [p(3), v_body(3), q(4), w(3), w_rot(6), alpha_rot(6)]
        :param ref: [px, py, pz, vx, vy, vz, yaw_des, dot_yaw_des]
        :param param_est: [m_est, rx, ry]
        :param u_prev: previous commanded rotor speeds (6,)
        :return: (status, w_cmd)
        '''
        if u_prev is None:
            u_prev = np.ones(6) * self.w_hover

        if param_est is None:
            params = self.p_default
        else:
            params = param_est

        # Build reference
        self.ref_nmpc[0:6] = ref[0:6]
        self.ref_nmpc[6] = np.cos(ref[6] / 2.0)
        self.ref_nmpc[9] = np.sin(ref[6] / 2.0)
        self.ref_nmpc[12] = ref[7]
        self.ref_nmpc[13:19] = self.w_hover

        y_ref = np.concatenate((self.ref_nmpc, u_prev))
        y_ref_N = self.ref_nmpc

        # Transform body velocity to world
        state_transformed = self.state_transform(state)

        # Initial state constraint
        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        N = self.ocp.solver_options.N_horizon

        if self.previous_states is not None:
            for stage in range(N):
                self.ocp_solver.set(stage, 'p', params)
                if stage < N - 1:
                    prev_state = self.previous_states[stage + 1]
                    y_ref_warm = np.concatenate((prev_state, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
                else:
                    y_ref_warm = np.concatenate((self.ref_nmpc, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
            self.ocp_solver.set(N, 'p', params)
            self.ocp_solver.set(N, 'y_ref', self.ref_nmpc)
        else:
            for stage in range(N):
                self.ocp_solver.set(stage, 'y_ref', y_ref)
                self.ocp_solver.set(stage, 'p', params)
            self.ocp_solver.set(N, 'y_ref', y_ref_N)
            self.ocp_solver.set(N, 'p', params)

        status = self.ocp_solver.solve()

        # Store trajectory for warm start
        self.previous_states = []
        for stage in range(N + 1):
            x_stage = self.ocp_solver.get(stage, 'x')
            self.previous_states.append(x_stage.copy())

        # Return commanded rotor speed (control input)
        w_cmd = self.ocp_solver.get(0, 'u')

        return status, w_cmd

    def solve_for_trajectory(self, state, t_curr, param_est=None, u_prev=None):
        '''
        Solve OCP with trajectory tracking along prediction horizon.
        :param state: full 25-dim state
        :param t_curr: current time
        :param param_est: [m_est, rx, ry]
        :param u_prev: previous commanded rotor speeds (6,)
        :return: (status, w_cmd)
        '''
        from ref_generation.ref_generator import get_reference

        if u_prev is None:
            u_prev = np.ones(6) * self.w_hover

        if param_est is None:
            params = self.p_default
        else:
            params = param_est

        N = self.ocp.solver_options.N_horizon
        T = self.ocp.solver_options.tf
        dt = T / N

        state_transformed = self.state_transform(state)

        # Initial state constraint
        self.ocp_solver.set(0, 'lbx', state_transformed)
        self.ocp_solver.set(0, 'ubx', state_transformed)

        # Warm start: shift previous trajectory forward
        if self.previous_states is not None:
            for stage in range(N):
                self.ocp_solver.set(stage, 'p', params)

                t_ref = t_curr + stage * dt
                ref = get_reference(t_ref)

                ref_nmpc = np.zeros(self.nx)
                ref_nmpc[0:6] = ref[0:6]
                ref_nmpc[6] = np.cos(ref[6] / 2.0)
                ref_nmpc[9] = np.sin(ref[6] / 2.0)
                ref_nmpc[12] = ref[7]
                ref_nmpc[13:19] = self.w_hover

                if stage < N - 1:
                    prev_state = self.previous_states[stage + 1]
                    y_ref_warm = np.concatenate((prev_state, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)
                else:
                    y_ref_warm = np.concatenate((ref_nmpc, u_prev))
                    self.ocp_solver.set(stage, 'y_ref', y_ref_warm)

            # Terminal
            t_ref_N = t_curr + T
            ref_N = get_reference(t_ref_N)
            ref_nmpc_N = np.zeros(self.nx)
            ref_nmpc_N[0:6] = ref_N[0:6]
            ref_nmpc_N[6] = np.cos(ref_N[6] / 2.0)
            ref_nmpc_N[9] = np.sin(ref_N[6] / 2.0)
            ref_nmpc_N[12] = ref_N[7]
            ref_nmpc_N[13:19] = self.w_hover

            self.ocp_solver.set(N, 'y_ref', ref_nmpc_N)
            self.ocp_solver.set(N, 'p', params)
        else:
            # First solve: use constant reference
            for stage in range(N):
                self.ocp_solver.set(stage, 'p', params)

                t_ref = t_curr + stage * dt
                ref = get_reference(t_ref)

                ref_nmpc = np.zeros(self.nx)
                ref_nmpc[0:6] = ref[0:6]
                ref_nmpc[6] = np.cos(ref[6] / 2.0)
                ref_nmpc[9] = np.sin(ref[6] / 2.0)
                ref_nmpc[12] = ref[7]
                ref_nmpc[13:19] = self.w_hover

                y_ref = np.concatenate((ref_nmpc, u_prev))
                self.ocp_solver.set(stage, 'y_ref', y_ref)

            # Terminal reference
            t_ref_N = t_curr + T
            ref_N = get_reference(t_ref_N)
            ref_nmpc_N = np.zeros(self.nx)
            ref_nmpc_N[0:6] = ref_N[0:6]
            ref_nmpc_N[6] = np.cos(ref_N[6] / 2.0)
            ref_nmpc_N[9] = np.sin(ref_N[6] / 2.0)
            ref_nmpc_N[12] = ref_N[7]
            ref_nmpc_N[13:19] = self.w_hover

            self.ocp_solver.set(N, 'y_ref', ref_nmpc_N)
            self.ocp_solver.set(N, 'p', params)

        status = self.ocp_solver.solve()

        # Store trajectory for warm start (next call)
        self.previous_states = []
        for stage in range(N + 1):
            x_stage = self.ocp_solver.get(stage, 'x')
            self.previous_states.append(x_stage.copy())

        w_cmd = self.ocp_solver.get(0, 'u')

        return status, w_cmd

    def state_transform(self, state):
        """
        Transform linear velocity from body to world frame.
        """
        q = state[6:10]
        R_B_W = quaternion_to_rotm(q)
        v_body = state[3:6]
        v_world = R_B_W @ v_body
        state_new = state.copy()
        state_new[3:6] = v_world
        return state_new

    def get_json_file_name(self):
        return self.solver_json
