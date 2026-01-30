from acados_template import AcadosModel
import casadi as cs
import numpy as np

class ActuatorModel:
    def __init__(self, RotorParams):
        """
        2nd order actuator model
        """

        self.model_name = 'actuator_model'
        self.model = AcadosModel()

        # Actuator parameters
        # 2nd order dynamic coefficients
        self.p1 = RotorParams['p'][0]       # Friction
        self.p2 = RotorParams['p'][1]       # Drag
        self.p3 = RotorParams['p'][2]       # Stiffness

        # Rotor speed constraint
        self.w_max = RotorParams['w_rotor_max']
        self.w_min = RotorParams['w_rotor_min']

        # Acceleration and jerk constraint
        self.alpha_max = RotorParams['alpha_rotor_max']
        self.j_max = RotorParams['jerk_rotor_max']

        # Rotor State
        self.w = cs.MX.sym('w', 6)
        self.a = cs.MX.sym('a', 6)
        self.x = cs.vertcat(self.w, self.a)
        self.x_dim = 12

        # Control state
        self.j = cs.MX.sym('j', 6)
        self.u = self.j
        self.j_dim = 6

        # Dynamics: dwdt = a, dadt = j
        self.dwdt = cs.MX.sym('dwdt', 6)
        self.dadt = cs.MX.sym('dadt', 6)
        self.xdot = cs.vertcat(self.dwdt, self.dadt)

    def export_acados_model(self) -> AcadosModel:
        """

        :return:
        """
        f_expl = cs.vertcat(self.a, self.j)
        f_impl = self.xdot - f_expl

        self.model.name = self.model_name
        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u

        return self.model
