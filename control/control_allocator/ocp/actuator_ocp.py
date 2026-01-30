from acados_template import AcadosOcp, AcadosOcpSolver
from control.control_allocator.model.actuator_model import ActuatorModel
import casadi as cs
import numpy as np
import os
import shutil


class ActuatorOCP:
    def __init__(self, DroneParams, RotorParams, AllocParams):
        """
        Actuator-aware control allocator using acados.

        Given wrench_des, find optimal jerk to minimize wrench tracking error.

        State: [w(6), alpha(6)] = 12D
        Control: j(6) = jerk
        Cost: ||wrench(w_next) - wrench_des||^2_Q + ||j||^2_R

        :param DroneParams: arm_length, motor_const, moment_const
        :param RotorParams: p, w_rotor_min/max, alpha_rotor_max, jerk_rotor_max
        :param AllocParams: Q (4D), R (scalar), dt
        """
        if RotorParams is None or DroneParams is None or AllocParams is None:
            raise ValueError('DroneParams, RotorParams, AllocParams must be provided')

        # Drone parameters for wrench computation
        self.l = DroneParams['arm_length']
        self.C_T = DroneParams['motor_const']
        self.k_m = DroneParams['moment_const']

        # Rotor constraints
        self.w_min = RotorParams['w_rotor_min']
        self.w_max = RotorParams['w_rotor_max']
        self.alpha_max = RotorParams['alpha_rotor_max']
        self.j_max = RotorParams['jerk_rotor_max']

        # 2nd-order dynamics parameters (for w_cmd back-calculation)
        self.p1 = RotorParams['p'][0]
        self.p2 = RotorParams['p'][1]
        self.p3 = RotorParams['p'][2]

        # Time step
        self.dt = AllocParams['dt']

        # Build acados model
        model_obj = ActuatorModel(RotorParams)
        acados_model = model_obj.export_acados_model()

        # Setup OCP
        self.ocp = AcadosOcp()
        self.ocp.model = acados_model

        nx = 12  # [w(6), alpha(6)]
        nu = 6   # j(6)
        N = 1    # Single step

        # Cost: NONLINEAR_LS
        # Stage cost: regularization on jerk
        # Terminal cost: wrench tracking

        # Stage cost (on jerk only)
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = acados_model.u  # y = j (6D)

        R = AllocParams['R'] * np.eye(nu)
        self.ocp.cost.W = R
        self.ocp.cost.yref = np.zeros(nu)

        # Terminal cost (wrench tracking)
        # w_next = w + alpha * dt (after integration)
        w = acados_model.x[:6]
        alpha = acados_model.x[6:]
        w_terminal = w + alpha * self.dt
        wrench_expr = self._compute_wrench_sym(w_terminal)

        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr_e = wrench_expr  # y_e = wrench (4D)

        Q = np.diag(AllocParams['Q'])
        self.ocp.cost.W_e = Q
        self.ocp.cost.yref_e = np.zeros(4)

        # Control input constraints (jerk bounds)
        self.ocp.constraints.lbu = -self.j_max * np.ones(nu)
        self.ocp.constraints.ubu = self.j_max * np.ones(nu)
        self.ocp.constraints.idxbu = np.arange(nu)

        # State constraints
        w_lb = self.w_min * np.ones(6)
        w_ub = self.w_max * np.ones(6)
        alpha_lb = -self.alpha_max * np.ones(6)
        alpha_ub = self.alpha_max * np.ones(6)

        x_lb = np.concatenate([w_lb, alpha_lb])
        x_ub = np.concatenate([w_ub, alpha_ub])

        # Initial state constraint using x0 (equality constraint)
        self.ocp.constraints.x0 = np.zeros(nx)

        # Terminal state constraints
        self.ocp.constraints.lbx_e = x_lb
        self.ocp.constraints.ubx_e = x_ub
        self.ocp.constraints.idxbx_e = np.arange(nx)

        # Solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 50
        self.ocp.solver_options.print_level = 0

        # Time settings
        self.ocp.solver_options.tf = self.dt
        self.ocp.solver_options.N_horizon = N

        # Clean up old generated code if exists
        code_export_dir = self.ocp.model.name + '_acados_ocp'
        if os.path.exists(code_export_dir):
            shutil.rmtree(code_export_dir)

        # Build solver
        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        if os.path.exists(self.solver_json):
            os.remove(self.solver_json)

        AcadosOcpSolver.generate(self.ocp, json_file=self.solver_json)
        AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        self.ocp_solver = AcadosOcpSolver.create_cython_solver(self.solver_json)

        # For warm start
        self.last_j = np.zeros(nu)

    def solve(self, wrench_des, s_rotor):
        """
        Solve for optimal jerk given desired wrench and current rotor state.

        :param wrench_des: Desired wrench [F, Mx, My, Mz] (4D)
        :param s_rotor: Current rotor state [w(6), alpha(6)] (12D)
        :return: (status, j_opt) - solver status and optimal jerk (6D)
        """
        # Set initial state (x0 equality constraint)
        self.ocp_solver.set(0, 'x', s_rotor)

        # Set terminal reference (wrench_des)
        self.ocp_solver.set(1, 'yref', wrench_des)

        # Warm start with last solution
        self.ocp_solver.set(0, 'u', self.last_j)

        # Solve
        status = self.ocp_solver.solve()

        # Get optimal jerk
        j_opt = self.ocp_solver.get(0, 'u')

        if status == 0:
            self.last_j = j_opt.copy()

        return status, j_opt

    def compute_w_cmd(self, s_rotor, j_opt):
        """
        Back-calculate w_cmd from optimal jerk using 2nd-order dynamics.

        j = -(p1 + p2*w)*alpha + p3*(w_cmd - w)
        => w_cmd = w + (j + (p1 + p2*w)*alpha) / p3

        :param s_rotor: Current rotor state [w(6), alpha(6)]
        :param j_opt: Optimal jerk (6D)
        :return: w_cmd (6D)
        """
        w = s_rotor[:6]
        alpha = s_rotor[6:12]

        w_cmd = w + (j_opt + (self.p1 + self.p2 * w) * alpha) / self.p3

        # Clamp to valid range
        w_cmd = np.clip(w_cmd, self.w_min, self.w_max)

        return w_cmd

    def _compute_wrench_sym(self, w):
        """
        Compute wrench symbolically using CasADi.

        :param w: Rotor speed (CasADi MX, 6D)
        :return: wrench [F, Mx, My, Mz] (CasADi MX, 4D)
        """
        T_rotor = self.C_T * (w ** 2)
        cp3, sp3 = np.cos(np.pi / 3), np.sin(np.pi / 3)

        f = cs.sum1(T_rotor)
        mx = self.l * ((T_rotor[0] + T_rotor[2]) * cp3 + T_rotor[1]
                       - (T_rotor[3] + T_rotor[5]) * cp3 - T_rotor[4])
        my = self.l * (-(T_rotor[0] + T_rotor[5]) * sp3
                       + (T_rotor[2] + T_rotor[3]) * sp3)
        mz = self.k_m * (-T_rotor[0] + T_rotor[1] - T_rotor[2]
                         + T_rotor[3] - T_rotor[4] + T_rotor[5])

        return cs.vertcat(f, mx, my, mz)

    def get_json_file_name(self):
        return self.solver_json
