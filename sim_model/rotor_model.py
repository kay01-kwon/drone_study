"""
Rotor simulation model
- Second order rotor dynamics
- Saturation is considered
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class RotorState:
    w_rotor: np.ndarray
    alpha_rotor: np.ndarray

class RotorModel:
    """
    RotorModel
    :param: RotorParams
    """

    def __init__(self, RotorParams):

        # Second order parameter
        self.p1 = RotorParams['p'][0]   # Friction
        self.p2 = RotorParams['p'][1]   # Drag
        self.p3 = RotorParams['p'][2]   # Stiffness

        # Constraints
        self.w_rotor_min = RotorParams['w_rotor_min']
        self.w_rotor_max = RotorParams['w_rotor_max']
        self.alpha_rotor_max = RotorParams['alpha_rotor_max']
        self.jerk_rotor_max = RotorParams['jerk_rotor_max']

        # Number of rotors
        self.num_rotors = RotorParams['num_rotors']

    def pack_state(self, w_rotor, alpha_rotor):
        """Pack rotor state into vector"""
        return np.concatenate([w_rotor, alpha_rotor])

    def unpack_state(self, s_rotor):
        """Unpack rotor state into vector"""
        w_rotor = s_rotor[0:self.num_rotors]
        alpha_rotor = s_rotor[self.num_rotors:]
        return w_rotor, alpha_rotor

    def dynamics(self, t, s_rotor, w_cmd):
        """
        Dynamics function of six rotor
        :param t: time (Not use, but required for rk4 interface)
        :param s_rotor: state vector for rotor [12]
        :param w_cmd: command vector for rotor [6]
        :return: ds_rotordt state derivative [12]
        """

        w_rotor, alpha_rotor = self.unpack_state(s_rotor)

        jerk_phys = np.zeros((self.num_rotors,))
        w_dot_eff = np.zeros((self.num_rotors,))
        jerk_eff = np.zeros((self.num_rotors,))

        for i in range(self.num_rotors):
            # Clamp cmd value first
            w_cmd[i] = np.clip(w_cmd[i], self.w_rotor_min, self.w_rotor_max)

            # Compute jerk from cmd and rotor states
            jerk_phys[i] = (-(self.p1 + self.p2*w_rotor[i])*alpha_rotor[i]
            - self.p3*(w_rotor[i]-w_cmd[i]))
            # Clamp Jerk
            jerk_phys[i] = np.clip(jerk_phys[i],-self.jerk_rotor_max, self.jerk_rotor_max)

            # Clamp Acceleration --> Clamp Jerk again...
            if alpha_rotor[i] >= self.alpha_rotor_max and jerk_phys[i] > 0:
                w_dot_eff[i] = self.alpha_rotor_max
                jerk_eff[i] = 0.0
            elif alpha_rotor[i] <= -self.alpha_rotor_max and jerk_phys[i] < 0:
                w_dot_eff[i] = -self.alpha_rotor_max
                jerk_eff[i] = 0.0
            else:
                w_dot_eff[i] = alpha_rotor[i]
                jerk_eff[i] = jerk_phys[i]

        return self.pack_state(w_dot_eff, jerk_eff)