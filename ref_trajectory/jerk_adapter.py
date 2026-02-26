"""
Adaptive Jerk Limit Controller

P-control style adaptation of jerk limit based on actuator tracking performance.
Compares dT_cmd(k-2) with dT_actual(k-1) to estimate actuator capability.

Author: Claude
Date: 2026-02-26
"""

import numpy as np
from collections import deque


class AdaptiveJerkAdapter:
    """
    Adaptive jerk limit controller using P-control.

    Monitors actuator tracking performance:
    - eta = ||dT_actual(k-1)|| / ||dT_cmd(k-2)||
    - eta ~ 1.0: actuator keeping up
    - eta < 1.0: actuator saturating/lagging

    Adjusts j_limit using P-control with deadband.
    """

    def __init__(self, params=None):
        """
        Initialize adaptive jerk adapter.

        :param params: Dictionary with adapter parameters
        """
        if params is None:
            params = {}

        # Mass for converting jerk to thrust rate
        self.m = params.get('m', 3.0)  # kg
        self.g = 9.81  # m/s^2

        # Jerk limit bounds (m/s^3)
        self.j_limit_max = params.get('j_limit_max', 30.0)
        self.j_limit_min = params.get('j_limit_min', 5.0)
        self.j_limit = params.get('j_limit_init', 15.0)

        # P-control parameters
        self.K_p = params.get('K_p', 0.3)
        self.deadband = params.get('deadband', 0.1)

        # Rotation budget parameters
        self.omega_max_expected = params.get('omega_max_expected', 2.0)  # rad/s
        self.rotation_reserve_ratio = params.get('rotation_reserve_ratio', 0.3)

        # History buffers for delay model
        # dT_cmd(k-2) appears as dT_actual(k-1)
        self.dT_cmd_history = deque(maxlen=3)  # Store k, k-1, k-2
        self.dT_actual_history = deque(maxlen=2)  # Store k, k-1
        self.T_prev = None  # Previous total thrust (for computing dT)

        # Statistics
        self.eta_history = deque(maxlen=100)
        self.j_limit_history = deque(maxlen=100)

    def reset(self):
        """Reset adapter state."""
        self.dT_cmd_history.clear()
        self.dT_actual_history.clear()
        self.T_prev = None
        self.j_limit = (self.j_limit_max + self.j_limit_min) / 2.0
        self.eta_history.clear()
        self.j_limit_history.clear()

    def update(self, T_cmd, T_actual, omega):
        """
        Update jerk limit based on actuator tracking performance.

        :param T_cmd: Commanded total thrust (scalar, N)
        :param T_actual: Actual total thrust (scalar, N)
        :param omega: Current angular velocity [omega_x, omega_y, omega_z] (rad/s)
        :return: Updated jerk limit (m/s^3)
        """
        # Compute thrust changes
        if self.T_prev is not None:
            dT_cmd = T_cmd - self.T_prev['cmd']
            dT_actual = T_actual - self.T_prev['actual']

            # Store in history
            self.dT_cmd_history.append(dT_cmd)
            self.dT_actual_history.append(dT_actual)

            # Need at least 2 steps of history for comparison
            # Compare dT_cmd(k-2) with dT_actual(k-1)
            if len(self.dT_cmd_history) >= 3 and len(self.dT_actual_history) >= 2:
                dT_cmd_k2 = self.dT_cmd_history[-3]  # k-2
                dT_actual_k1 = self.dT_actual_history[-2]  # k-1

                # Compute eta (tracking ratio) with robustness
                dT_cmd_norm = abs(dT_cmd_k2)
                dT_actual_norm = abs(dT_actual_k1)

                # Only update if command change was significant (0.5 N threshold)
                if dT_cmd_norm > 0.5:
                    # Check sign alignment
                    sign_match = (dT_cmd_k2 * dT_actual_k1) >= 0

                    if sign_match:
                        eta = dT_actual_norm / dT_cmd_norm
                        eta = np.clip(eta, 0.0, 2.0)
                    else:
                        # Opposite direction - conservative estimate
                        eta = 0.5

                    # P-control adaptation with asymmetry
                    error = 1.0 - eta

                    if abs(error) > self.deadband:
                        if error > 0:  # eta < 1, decrease limit
                            dj = -self.K_p * error * self.j_limit_max
                        else:  # eta > 1, increase limit (slower)
                            dj = -0.3 * self.K_p * error * self.j_limit_max

                        self.j_limit = np.clip(
                            self.j_limit + dj,
                            self.j_limit_min,
                            self.j_limit_max
                        )

                    # Store statistics
                    self.eta_history.append(eta)
                    self.j_limit_history.append(self.j_limit)

        # Update previous values
        self.T_prev = {'cmd': T_cmd, 'actual': T_actual}

        # Compute available thrust rate budget
        T_dot_available = self.compute_available_thrust_rate(T_actual, omega)

        return self.j_limit, T_dot_available

    def compute_available_thrust_rate(self, T_current, omega):
        """
        Compute available thrust rate budget after accounting for rotation.

        |omega x T| contribution to effective thrust rate in inertial frame.
        For body-z aligned thrust: |omega_xy| * T

        :param T_current: Current total thrust (N)
        :param omega: Angular velocity [omega_x, omega_y, omega_z] (rad/s)
        :return: Available thrust rate (N/s)
        """
        # omega x T magnitude (T is along body z-axis)
        omega_xy = max(abs(omega[0]), abs(omega[1])) * np.sqrt(2)

        # Base reserve for rotation
        T_hover = self.m * self.g
        base_reserve = self.rotation_reserve_ratio * self.omega_max_expected * T_hover

        # Actual rotation contribution
        actual_rotation = omega_xy * T_current

        # Use the larger of base reserve or actual (with margin)
        rotation_term = max(base_reserve, actual_rotation * 1.2)

        # Total thrust rate budget from jerk limit
        T_dot_total = self.j_limit * self.m

        # Available for motor command
        T_dot_available = max(T_dot_total - rotation_term, 0.1 * T_dot_total)

        return T_dot_available

    def get_statistics(self):
        """Get adapter statistics."""
        if len(self.eta_history) == 0:
            return {
                'eta_mean': 1.0,
                'eta_std': 0.0,
                'j_limit_mean': self.j_limit,
                'j_limit_current': self.j_limit
            }

        return {
            'eta_mean': np.mean(self.eta_history),
            'eta_std': np.std(self.eta_history),
            'j_limit_mean': np.mean(self.j_limit_history),
            'j_limit_current': self.j_limit
        }

    def get_current_jerk_limit(self):
        """Get current jerk limit."""
        return self.j_limit
