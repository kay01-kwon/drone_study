from acados_template import AcadosModel
import casadi as cs
from control.nmpc.cs_utils import cs_math_tool
import numpy as np

class S550_3DOF_model:
    def __init__(self, DynParam, DroneParam):
        '''
        DynParam: m, Jyy
        DroneParam: arm_length, motor_const, moment_const
        '''

        self.model_name = 'S550_3DOF_model'
        self.model = AcadosModel()

        # Dynamic Parameter
        self.m = DynParam['m']
        self.Jyy = DynParam['MoiArray'][1]

        # Drone parameter
        self.l = DroneParam['arm_length']
        self.C_T = DroneParam['motor_const']
        self.k_m = DroneParam['moment_const']

        # State x (p, v, th, q) (dim: 6)
        self.p = cs.MX.sym('p', 2)      # x, z
        self.v = cs.MX.sym('v', 2)      # vx, vz
        self.th = cs.MX.sym('th', 1)    # pitch
        self.q = cs.MX.sym('q', 1)      # pitch rate
        self.x = cs.vertcat(self.p, self.v,
                            self.th, self.q)
        self.x_dim = 6

        # Desired rotor thrusts (dim: 3)
        self.u1 = cs.MX.sym('u1')
        self.u2 = cs.MX.sym('u2')
        self.u3 = cs.MX.sym('u3')
        self.u = cs.vertcat(self.u1, self.u2, self.u3)
        self.u_dim = 3

        # The time derivative of state
        self.dpdt = cs.MX.sym('dpdt', 2)
        self.dvdt = cs.MX.sym('dvdt', 2)
        self.dthdt = cs.MX.sym('dthdt', 1)
        self.dqdt = cs.MX.sym('dqdt', 1)
        self.xdot = cs.vertcat(self.dpdt, self.dvdt,
                               self.dthdt, self.dqdt)

    def export_acados_model(self) -> AcadosModel:
        '''
        Export acados model
        :return: acados model
        '''

        print('Exporting acados model')

        f_expl = cs.vertcat(self._p_dynamics(), self._v_dynamics(),
                            self._th_dynamics(), self._q_dynamics())

        f_impl = self.xdot - f_expl

        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u
        self.model.name = self.model_name

        return self.model

    def _p_dynamics(self):
        '''
        dpdt = v
        :return: linear velocity (World frame)
        '''
        return self.v

    def _v_dynamics(self):
        '''
        dvdt = R @ f/m + g_vec
        Hexarotor has 3 groups of 2 motors each, so total thrust = 2*(u1+u2+u3)
        :return:
        '''

        # Total thrust (factor of 2 for paired motors in hexarotor)
        f_col = 2.0 * (self.u1 + self.u2 + self.u3)

        # Force in 2D (thrust along body z-axis = e2)
        f = cs.vertcat(0.0, f_col)
        acc_input = f/self.m

        g_vec = cs.vertcat(0.0, -9.81)

        R = cs_math_tool.pitch_to_rotm(self.th)

        dvdt = cs.mtimes(R, acc_input) + g_vec

        return dvdt

    def _th_dynamics(self):
        dthdt = self.q
        return dthdt

    def _q_dynamics(self):

        My = self._control_alloc_moment()

        angual_acc_y = My/self.Jyy

        dqdt = angual_acc_y
        return dqdt

    def _control_alloc_moment(self):
        '''
        Thrust to moment (hexarotor 3-group)
        Each group has 2 motors; pitch moment arm = 2*l*sin(60°) = l*sqrt(3)
        :return: moment
        '''

        My = self.l * cs.sqrt(3.0) * (-self.u1 + self.u3)

        return My



