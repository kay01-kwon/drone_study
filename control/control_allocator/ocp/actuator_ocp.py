import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from control.control_allocator.model.actuator_model import ActuatorModel
from scipy.linalg import block_diag
import casadi as cs
import numpy as np

from control.control_allocator.model.actuator_model import ActuatorModel


class ActuatorOCP:
    def __init__(self, DroneParams, RotorParams, AllocParams):
        """

        :param DroneParams:
        :param RotorParams:
        :param AllocParams:
        """

        if RotorParams is None:
            print('Please specify RotorParams')
            exit(1)

        if DroneParams is None:
            print('Please specify DroneParams')
            exit(1)

        if AllocParams is None:
            print('Please specify AllocParams')
            exit(1)

        # Drone parameters
        self.l = DroneParams['arm_length']
        self.C_T = DroneParams['motor_const']
        self.k_m = DroneParams['moment_const']

        # Instantiate model
        model_obj = ActuatorModel(RotorParams)
        acados_model = model_obj.export_acados_model()
        self.ocp = AcadosOcp(acados_model)

        # Predicted model for construction of cost function
        dt = AllocParams['dt']
        w_next = acados_model.x[:6] + acados_model.x[6:] * dt
        wrench = self._compute_wrench(w_next)

        # Objective function
        y_expr = wrench

        Q = np.diag([AllocParams['QArray']])
        R = AllocParams['R']*np.eye(6)

        # Set cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'

        self.ocp.cost_y_expr = y_expr
        self.ocp.cost.W = scipy.linalg.block_diag(Q, R)
        self.ocp.yref = np.zeros((10,))

        # Constraints for control input
        self.ocp.constraints.lbu = -np.array([RotorParams['jerk_rotor_max']])
        self.ocp.constraints.ubu = np.array([RotorParams['jerk_rotor_max']])
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # Bound of rotor speed
        w_lb = np.ones(6) * RotorParams['w_rotor_min']
        w_ub = np.ones(6) * RotorParams['w_rotor_max']

        # Bound of rotor acceleration
        alpha_lb = -np.ones(6) * RotorParams['alpha_rotor_max']
        alpha_ub = np.ones(6) * RotorParams['alpha_rotor_max']

        # Bound of rotor state (Integration of speed and acceleration)
        state_lb = np.concatenate((w_lb, alpha_lb))
        state_ub = np.concatenate((w_ub, alpha_ub))

        self.ocp.constraints.lb = state_lb
        self.ocp.constraints.ub = state_ub
        self.ocp.constraints.idxbx = np.arange(12)

        nx = acados_model.x.rows()
        nu = acados_model.u.rows()

        # Set ocp solver
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_config.nlp_solver_type = 'SQP_RTI'

        # generate json file and generate cython
        self.solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        AcadosOcpSolver.generate(self.ocp, json_file = self.solver_json)
        AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        self.ocp_solver = AcadosOcpSolver.create_cython_solver(self.solver_json)

    def solve_wrench(self, wrench_des, s_rotor):

        self.ocp_solver.set(0, 'lbx', wrench_des)
        self.ocp_solver.set(0, 'ubx', wrench_des)

    def _compute_wrench(self, w):
        """
        Compute collective thrust and momemt
        :param w: rotor speed
        :return: f, mx, my, mz
        """

        T_rotor = self.C_T * (w**2)
        cp3, sp3 = np.cos(np.pi/3), np.sin(np.pi/3)

        # Compute collective thrust
        f = cs.sum1(T_rotor)

        # Compute moment along x, y, and z axis
        mx = self.l * ((T_rotor[0] + T_rotor[2])*cp3 + T_rotor[1]
                       -(T_rotor[3] + T_rotor[5])*cp3 - T_rotor[4])
        my = self.l * (-(T_rotor[0] + T_rotor[5])*sp3
                       +(T_rotor[2] + T_rotor[3])*sp3 )
        mz = self.k_m * (-T_rotor[0] + T_rotor[1] - T_rotor[2]
                         + T_rotor[3] - T_rotor[4] + T_rotor[5])

        return cs.vertcat(f, mx, my, mz)
