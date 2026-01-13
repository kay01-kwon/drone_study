from acados_template import AcadosOcp, AcadosOcpSolver
from drone_control.nmpc.model.S550_simple import S550_model
from utils.math_tool import quaternion_to_rotm
from scipy.linalg import block_diag
import numpy as np

class S550SimpleOcp:
    def __init__(self, DynParam = None, DroneParam = None, MpcParam = None):
        '''
        Constructor
        :param DynParam: m, MoiArray (Jxx, Jyy, Jzz)
        :param DroneParam: arm_length, rotor_const, moment_const, T_max, T_min
        :param MpcParam: t_horizon, n_nodes, QArray, RArray
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
                          'motor_const':C_T,
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
            Q = 2*np.diag(MpcParam['QArray'])
            R = 2*np.diag(MpcParam['R']*6)


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

        self.ref_nmpc = np.zeros((13,))
        self.ref_nmpc[6] = 1.0

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
        q = state[6:10]
        R_B_W = quaternion_to_rotm(q)
        v_world = R_B_W @ state[3:6]

        state_new = state
        state_new[3:6] = v_world

        # Set constraint at the first stage
        self.ocp_solver.set(0, 'lbx', state_new)
        self.ocp_solver.set(0, 'ubx', state_new)

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
        q = state[6:10]
        R_B_W = quaternion_to_rotm(q)
        v_world = R_B_W @ state[3:6]

        state_new = state
        state_new[3:6] = v_world

        # Set constraint at the first stage
        self.ocp_solver.set(0, 'lbx', state_new)
        self.ocp_solver.set(0, 'ubx', state_new)

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

    def get_json_file_name(self):
        return self.solver_json