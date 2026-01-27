from acados_template import AcadosModel
import casadi as cs
import numpy as np
from control.nmpc.cs_utils import cs_math_tool

class S550ActuatorModel:
    def __init__(self, DynParam, DroneParam, ActuatorParam=None):
        '''
        S550 model with linear 2nd-order actuator dynamics.
        Saturation is NOT modeled here — handled by path constraints in OCP.

        State: [p(3), v(3), q(4), w(3), w_rot(6), alpha_rot(6)] = 25 dim
        Input: [w_cmd(6)] = commanded rotor speeds

        Actuator dynamics (per rotor, linear):
            dw_rot/dt   = alpha_rot
            dalpha/dt   = -(p1 + p2*w_rot)*alpha_rot + p3*(w_cmd - w_rot)

        Constraints (enforced as path constraints in OCP, not here):
            w_rot_min <= w_rot <= w_rot_max
            -alpha_max <= alpha_rot <= alpha_max

        DroneParam: arm_length, motor_const, moment_const
        ActuatorParam: p1, p2, p3, alpha_max, j_max
        '''

        self.model_name = 'S550_actuator_func'
        self.model = AcadosModel()

        # Drone parameters
        self.m = DynParam['m']
        self.J = DynParam['MoiArray']
        self.l = DroneParam['arm_length']
        self.C_T = DroneParam['motor_const']
        self.k_m = DroneParam['moment_const']

        # Actuator parameters (2nd order model)
        if ActuatorParam is None:
            # Default setup
            ActuatorParam = {
                'p1': 30.0,
                'p2': 0.001,
                'p3': 900.0,
                'alpha_max': 15000.0,
                'j_max': 300000.0,
            }
        self.p1 = ActuatorParam['p1']
        self.p2 = ActuatorParam['p2']
        self.p3 = ActuatorParam['p3']

        # State x: [p, v, q, w, w_rot, alpha_rot] (dim: 25)
        self.p_pos = cs.MX.sym('p', 3)                 # Position (World)
        self.v = cs.MX.sym('v', 3)                     # Velocity (World)
        self.q = cs.MX.sym('q', 4)                     # Quaternion (Body to World)
        self.w = cs.MX.sym('w', 3)                     # Angular velocity (Body)
        self.w_rot = cs.MX.sym('w_rot', 6)             # Rotor speed (rad/s)
        self.alpha_rot = cs.MX.sym('alpha_rot', 6)     # Rotor acceleration (rad/s^2)
        self.x = cs.vertcat(self.p_pos, self.v, self.q, self.w,
                            self.w_rot, self.alpha_rot)
        self.x_dim = 25

        # Control input: commanded rotor speeds (dim: 6)
        self.u = cs.MX.sym('u_cmd', 6)
        self.u_dim = 6

        # Parameters: m_est, rx, ry
        self.param = cs.MX.sym('param', 3)

        # Time derivatives
        self.dpdt = cs.MX.sym('dpdt', 3)
        self.dvdt = cs.MX.sym('dvdt', 3)
        self.dqdt = cs.MX.sym('dqdt', 4)
        self.dwdt = cs.MX.sym('dwdt', 3)
        self.dw_rot_dt = cs.MX.sym('dw_rot_dt', 6)
        self.dalpha_rot_dt = cs.MX.sym('dalpha_rot_dt', 6)
        self.xdot = cs.vertcat(self.dpdt, self.dvdt, self.dqdt,
                               self.dwdt, self.dw_rot_dt,
                               self.dalpha_rot_dt)

    def export_acados_model(self) -> AcadosModel:
        '''
        Export acados model with linear actuator dynamics.
        '''
        # Linear actuator dynamics (no saturation)
        dw_rot, dalpha_rot = self._actuator_dynamics()

        f_expl = cs.vertcat(
            self._p_dynamics(),
            self._v_dynamics(),
            self._q_dynamics(),
            self._w_dynamics(),
            dw_rot,
            dalpha_rot
        )

        f_impl = self.xdot - f_expl

        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u
        self.model.p = self.param
        self.model.name = self.model_name

        return self.model

    def _actuator_dynamics(self):
        '''
        Linear 2nd-order actuator dynamics.

        dw_rot/dt = alpha_rot
        dalpha_rot/dt = -(p1 + p2*w_rot)*alpha_rot + p3*(w_cmd - w_rot)

        Returns (dw_rot, dalpha_rot) each of dim 6.
        '''
        jerk_list = []

        for i in range(6):
            w_i = self.w_rot[i]
            alpha_i = self.alpha_rot[i]
            w_cmd_i = self.u[i]

            # Linear jerk (no clamp, no saturation)
            j_i = -(self.p1 + self.p2 * w_i) * alpha_i \
                   + self.p3 * (w_cmd_i - w_i)

            jerk_list.append(j_i)

        dw_rot = self.alpha_rot           # dw/dt = alpha
        dalpha_rot = cs.vertcat(*jerk_list)  # dalpha/dt = jerk

        return dw_rot, dalpha_rot

    def _p_dynamics(self):
        return self.v

    def _v_dynamics(self):
        '''
        dvdt = R @ f/m + g_vec
        Thrust computed from actual rotor speed: T_i = C_T * w_rot_i^2
        '''
        m_est = self.param[0]
        delta_m = self.m / m_est - 1.0

        f_col = self._compute_collective_thrust()
        f = cs.vertcat(0.0, 0.0, f_col)
        acc_input = f / self.m

        g_vec = cs.vertcat(0.0, 0.0, -9.81)
        R = cs_math_tool.quaternion_to_rotm(self.q)

        dvdt = cs.mtimes(R, acc_input) + g_vec
        dvdt_delta = delta_m * cs.mtimes(R, acc_input)
        return dvdt + dvdt_delta

    def _q_dynamics(self):
        w_quat = cs.vertcat(0.0, self.w)
        return 0.5 * cs_math_tool.otimes(self.q, w_quat)

    def _w_dynamics(self):
        Jxx = self.J[0]
        Jyy = self.J[1]
        Jzz = self.J[2]

        Mx, My, Mz = self._control_alloc_moment()
        m_vec = cs.vertcat(Mx / Jxx, My / Jyy, Mz / Jzz)

        w_x = self.w[0]
        w_y = self.w[1]
        w_z = self.w[2]

        rx = self.param[1]
        ry = self.param[2]
        f_col = self._compute_collective_thrust()

        disturbance_com_offset = cs.vertcat(
            -ry * f_col / Jxx,
             rx * f_col / Jyy,
             0.0
        )

        inertial_effect = cs.vertcat(
            (Jzz - Jyy) / Jxx * w_y * w_z,
            (Jxx - Jzz) / Jyy * w_x * w_z,
            (Jyy - Jxx) / Jzz * w_x * w_y
        )

        return m_vec - inertial_effect + disturbance_com_offset

    def _compute_collective_thrust(self):
        '''
        Compute thrust from actual rotor speeds: T_i = C_T * w_rot_i^2
        '''
        thrust = 0.0
        for i in range(6):
            thrust = thrust + self.C_T * self.w_rot[i] ** 2
        return thrust

    def _control_alloc_moment(self):
        '''
        Moment allocation from actual rotor thrusts (computed from w_rot).
        '''
        cos_pi_3 = np.cos(np.pi / 3)
        sin_pi_3 = np.sin(np.pi / 3)

        # Individual thrusts from actual rotor speeds
        T = [self.C_T * self.w_rot[i] ** 2 for i in range(6)]

        Mx = self.l * (
            (T[0] + T[2]) * cos_pi_3 + T[1]
            - (T[3] + T[5]) * cos_pi_3 - T[4]
        )

        My = self.l * (
            -(T[0] + T[5]) * sin_pi_3
            + (T[2] + T[3]) * sin_pi_3
        )

        Mz = self.k_m * (
            -T[0] + T[1] - T[2]
            + T[3] - T[4] + T[5]
        )

        return Mx, My, Mz
