from acados_template import AcadosModel
import casadi as cs
import numpy as np
from control.nmpc.cs_utils import cs_math_tool

class S550ParamModel:
    def __init__(self, DynParam, DroneParam):
        '''
        DroneParam: m, arm_length, motor_const, moment_const
        '''

        self.model_name = 'S550_actuator_func'
        self.model = AcadosModel()

        # drone parameter
        # mass, MOI, arm length, motor const, moment const
        self.m = DynParam['m']
        self.J = DynParam['MoiArray']
        self.l = DroneParam['arm_length']
        self.C_T = DroneParam['motor_const']
        self.k_m = DroneParam['moment_const']

        # State x (p, v, q, w) (dim: 13)
        self.p = cs.MX.sym('p', 3)                  # Position (World)
        self.v = cs.MX.sym('v', 3)                  # Velocity (World)
        self.q = cs.MX.sym('q', 4)                  # Quaternion (Body to World) qw, qx, qy, qz
        self.w = cs.MX.sym('w', 3)                  # Angular velocity (Body)
        self.w_rot = cs.MX.sym('w_rot', 6)          # Rotor speed (RPM)
        self.alpha_rot = cs.MX.sym('alpha_rot', 6)  # Rotor acceleration (RPM/s)
        self.x = cs.vertcat(self.p, self.v, self.q, self.w, self.w_rot, self.alpha_rot)
        self.x_dim = 25

        # Desired rotor thrusts (dim: 6)
        self.u1 = cs.MX.sym('u1')
        self.u2 = cs.MX.sym('u2')
        self.u3 = cs.MX.sym('u3')
        self.u4 = cs.MX.sym('u4')
        self.u5 = cs.MX.sym('u5')
        self.u6 = cs.MX.sym('u6')
        self.u = cs.vertcat(self.u1, self.u2, self.u3,
                            self.u4, self.u5, self.u6)
        self.u_dim = 6

        # Pass parameter related to dm and com offset
        self.param = cs.MX.sym('p',3)

        # The time derivative of state
        self.dpdt = cs.MX.sym('dpdt', 3)
        self.dvdt = cs.MX.sym('dvdt', 3)
        self.dqdt = cs.MX.sym('dqdt', 4)
        self.dwdt = cs.MX.sym('dwdt', 3)
        self.dw_rot_dt = cs.MX.sym('dw_rot_dt', 6)
        self.dalpha_rot_dt = cs.MX.sym('dalpha_rot_dt', 6)
        self.xdot = cs.vertcat(self.dpdt,
                               self.dvdt,
                               self.dqdt,
                               self.dwdt,
                               self.dw_rot_dt,
                               self.dalpha_rot_dt)

    def export_acados_model(self) -> AcadosModel:
        '''
        Export acados model
        :return: acados model
        '''
        f_expl = cs.vertcat(self._p_dynamics(), self._v_dynamics(),
                            self._q_dynamics(), self._w_dynamics())

        f_impl = self.xdot - f_expl

        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u
        #********************************************
        self.model.p = self.param
        #********************************************
        self.model.name = self.model_name

        return self.model

    def _p_dynamics(self):
        '''
        dpdt = v
        :return: linear velocity ( World frame )
        '''
        return self.v

    def _v_dynamics(self):
        '''
        dvdt = R @ f/m + g_vec
        :return: dvdt ( World frame )
        '''

        # Get dynamic parameter
        m_est = self.param[0]
        delta_m = self.m / m_est - 1.0

        # Collective thrust
        f_col = self._compute_collective_thrust()

        # Force in 3 dim ( Body frame )
        f = cs.vertcat(0.0, 0.0, f_col)

        acc_input = f/self.m

        g_vec = cs.vertcat(0.0, 0.0, -9.81)

        R = cs_math_tool.quaternion_to_rotm(self.q)

        dvdt = cs.mtimes(R, acc_input) + g_vec
        dvdt_delta = delta_m*cs.mtimes(R, acc_input)
        dvdt_combined = dvdt + dvdt_delta
        return dvdt_combined

    def _q_dynamics(self):
        '''
        dqdt = 0.5*otimes(q,w_quat)
        :return:
        '''

        w_quat = cs.vertcat(0.0, self.w)

        dqdt = 0.5*cs_math_tool.otimes(self.q, w_quat)
        return dqdt

    def _w_dynamics(self):
        '''
        dwdt = J_inv @ M - cross(w,J@w)
        :return:
        '''

        Jxx = self.J[0]
        Jyy = self.J[1]
        Jzz = self.J[2]

        Mx, My, Mz = self._control_alloc_moment()

        m_vec = cs.vertcat(Mx/Jxx, My/Jyy, Mz/Jzz)


        # Get angular velocity
        w_x = self.w[0]
        w_y = self.w[1]
        w_z = self.w[2]

        # Disturbance due to com offset
        rx = self.param[1]
        ry = self.param[2]

        f_col = self._compute_collective_thrust()

        #******************************************************
        disturbance_com_offset = cs.vertcat(-ry*f_col/Jxx,
                                            rx*f_col/Jyy,
                                            0.0)
        #******************************************************

        # inertial effect = J_inv @ (w x (J*w))
        inertial_effect = cs.vertcat((Jzz-Jyy)/Jxx*w_y*w_z,
                                     (Jxx-Jzz)/Jyy*w_x*w_z,
                                     (Jyy-Jxx)/Jzz*w_x*w_y)

        dwdt = m_vec - inertial_effect + disturbance_com_offset
        return dwdt

    def _compute_collective_thrust(self):
        f_col = (self.u1 + self.u2 + self.u3
                 + self.u4 + self.u5 + self.u6)
        return f_col

    def _control_alloc_moment(self):
        '''
        Thrust to moment
        :return:
        '''
        cos_pi_3 = np.cos(np.pi/3)
        sin_pi_3 = np.sin(np.pi/3)

        Mx = self.l * (
        (self.u1 + self.u3)*cos_pi_3 + self.u2
        -(self.u4 + self.u6)*cos_pi_3 - self.u5
        )

        My = self.l * (
            -(self.u1 + self.u6)*sin_pi_3
            +(self.u3 + self.u4)*sin_pi_3
        )

        Mz = self.k_m * (
            -self.u1 + self.u2 - self.u3
            +self.u4 - self.u5 + self.u6
        )

        return Mx, My, Mz