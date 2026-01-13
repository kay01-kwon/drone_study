import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag


class MinimumSnapTrajectoryQP:
    """
    Minimum snap trajectory generator using Quadratic Programming.

    This implements the true minimum snap trajectory from:
    Mellinger, D., & Kumar, V. (2011). "Minimum snap trajectory generation
    and control for quadrotors"

    Uses 7th order polynomials and solves a QP to minimize the integral of
    squared snap subject to waypoint and continuity constraints.
    """

    def __init__(self, waypoints, times, yaw_angles=None):
        """
        Initialize QP-based minimum snap trajectory generator

        :param waypoints: List of 3D waypoints [[x0,y0,z0], [x1,y1,z1], ...]
        :param times: List of times to reach each waypoint [t0, t1, t2, ...]
        :param yaw_angles: List of yaw angles at each waypoint (optional)
        """
        self.waypoints = np.array(waypoints)
        self.times = np.array(times)
        self.num_waypoints = len(waypoints)
        self.num_segments = self.num_waypoints - 1

        if yaw_angles is None:
            self.yaw_angles = np.zeros(self.num_waypoints)
        else:
            self.yaw_angles = np.array(yaw_angles)

        # Polynomial order (7th order polynomial has 8 coefficients)
        self.poly_order = 7
        self.num_coeffs = self.poly_order + 1

        # Solve QP to get optimal polynomial coefficients
        self.coeffs = self._solve_qp()

    def _compute_Q_matrix(self, T):
        """
        Compute the cost matrix Q for minimizing snap over interval [0, T]

        For polynomial p(t) = c0 + c1*t + ... + c7*t^7
        Snap = d^4p/dt^4 = 24*c4 + 120*c5*t + 360*c6*t^2 + 840*c7*t^3

        Cost = ∫_0^T (snap)^2 dt
        """
        Q = np.zeros((self.num_coeffs, self.num_coeffs))

        # Only coefficients c4, c5, c6, c7 affect snap
        for i in range(4, self.num_coeffs):
            for j in range(4, self.num_coeffs):
                # Derivative coefficients for snap (4th derivative)
                k1 = i * (i-1) * (i-2) * (i-3)
                k2 = j * (j-1) * (j-2) * (j-3)

                # Integral of t^(i+j-8) from 0 to T
                power = i + j - 8 + 1
                Q[i, j] = k1 * k2 * (T ** power) / power

        return Q

    def _compute_A_matrix(self, t, derivative=0):
        """
        Compute constraint matrix for polynomial evaluation

        :param t: Time value
        :param derivative: 0=position, 1=velocity, 2=acceleration, 3=jerk
        :return: Row vector for evaluating derivative at time t
        """
        A = np.zeros(self.num_coeffs)

        for i in range(derivative, self.num_coeffs):
            # Compute derivative coefficient
            coeff = 1
            for k in range(derivative):
                coeff *= (i - k)

            # Evaluate at time t
            if i - derivative == 0:
                A[i] = coeff
            else:
                A[i] = coeff * (t ** (i - derivative))

        return A

    def _solve_qp(self):
        """
        Solve the QP problem to find optimal polynomial coefficients using scipy

        Decision variables: coefficients for all segments and all 3 axes
        Objective: minimize sum of squared snap over all segments
        Constraints: waypoint positions, continuity, boundary conditions
        """
        # Solve separately for each axis (x, y, z)
        coeffs_all_axes = []

        for axis in range(3):
            # Build cost matrix (Q) for all segments
            Q_blocks = []
            for seg in range(self.num_segments):
                T = self.times[seg + 1] - self.times[seg]
                Q_seg = self._compute_Q_matrix(T)
                Q_blocks.append(Q_seg)

            Q = block_diag(*Q_blocks)  # Block diagonal matrix

            # Objective function: 0.5 * x^T * Q * x
            def objective(x):
                return 0.5 * x.T @ Q @ x

            def objective_grad(x):
                return Q @ x

            # Build equality constraints
            A_eq = []
            b_eq = []

            # 1. Waypoint constraints
            for seg in range(self.num_segments):
                T = self.times[seg + 1] - self.times[seg]

                # Start of segment matches waypoint
                A_row = np.zeros(self.num_segments * self.num_coeffs)
                A_row[seg * self.num_coeffs:(seg + 1) * self.num_coeffs] = \
                    self._compute_A_matrix(0, derivative=0)
                A_eq.append(A_row)
                b_eq.append(self.waypoints[seg, axis])

                # End of segment matches next waypoint
                A_row = np.zeros(self.num_segments * self.num_coeffs)
                A_row[seg * self.num_coeffs:(seg + 1) * self.num_coeffs] = \
                    self._compute_A_matrix(T, derivative=0)
                A_eq.append(A_row)
                b_eq.append(self.waypoints[seg + 1, axis])

            # 2. Continuity constraints (velocity, acceleration, jerk)
            for seg in range(self.num_segments - 1):
                T = self.times[seg + 1] - self.times[seg]

                for deriv in [1, 2, 3]:  # velocity, acceleration, jerk
                    A_row = np.zeros(self.num_segments * self.num_coeffs)

                    # End of current segment
                    A_row[seg * self.num_coeffs:(seg + 1) * self.num_coeffs] = \
                        self._compute_A_matrix(T, derivative=deriv)

                    # Start of next segment (negative)
                    A_row[(seg + 1) * self.num_coeffs:(seg + 2) * self.num_coeffs] = \
                        -self._compute_A_matrix(0, derivative=deriv)

                    A_eq.append(A_row)
                    b_eq.append(0.0)

            # 3. Boundary conditions at start
            for deriv in [1, 2, 3]:  # velocity, acceleration, jerk = 0
                A_row = np.zeros(self.num_segments * self.num_coeffs)
                A_row[0:self.num_coeffs] = self._compute_A_matrix(0, derivative=deriv)
                A_eq.append(A_row)
                b_eq.append(0.0)

            # 4. Boundary conditions at end
            T_last = self.times[-1] - self.times[-2]
            for deriv in [1, 2, 3]:  # velocity, acceleration, jerk = 0
                A_row = np.zeros(self.num_segments * self.num_coeffs)
                A_row[-self.num_coeffs:] = self._compute_A_matrix(T_last, derivative=deriv)
                A_eq.append(A_row)
                b_eq.append(0.0)

            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)

            # Define constraint for scipy
            constraints = {'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq,
                          'jac': lambda x: A_eq}

            # Initial guess (zeros)
            x0 = np.zeros(self.num_segments * self.num_coeffs)

            # Solve using SLSQP (handles equality constraints well)
            result = minimize(objective, x0, method='SLSQP', jac=objective_grad,
                            constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})

            if not result.success:
                print(f"Warning: Optimization failed for axis {axis}: {result.message}")

            coeffs_all_axes.append(result.x)

        # Organize coefficients by segment
        coeffs = []
        for seg in range(self.num_segments):
            segment_coeffs = np.zeros((3, self.num_coeffs))
            for axis in range(3):
                start_idx = seg * self.num_coeffs
                end_idx = (seg + 1) * self.num_coeffs
                segment_coeffs[axis] = coeffs_all_axes[axis][start_idx:end_idx]

            coeffs.append({
                'coeffs': segment_coeffs,
                't0': self.times[seg],
                'tf': self.times[seg + 1],
                'T': self.times[seg + 1] - self.times[seg]
            })

        return coeffs

    def get_state(self, t):
        """
        Get desired position, velocity, acceleration at time t

        :param t: Current time
        :return: position (3,), velocity (3,), acceleration (3,), yaw, yaw_dot
        """
        # Find which segment we're in
        segment_idx = self._find_segment(t)

        if segment_idx < 0:
            # Before first waypoint
            pos = self.waypoints[0]
            vel = np.zeros(3)
            acc = np.zeros(3)
            yaw = self.yaw_angles[0]
            yaw_dot = 0.0
        elif segment_idx >= len(self.coeffs):
            # After last waypoint
            pos = self.waypoints[-1]
            vel = np.zeros(3)
            acc = np.zeros(3)
            yaw = self.yaw_angles[-1]
            yaw_dot = 0.0
        else:
            # In a segment
            segment = self.coeffs[segment_idx]
            tau = t - segment['t0']  # Local time in segment

            pos = np.zeros(3)
            vel = np.zeros(3)
            acc = np.zeros(3)

            for axis in range(3):
                c = segment['coeffs'][axis]

                # Position: p(t) = Σ c_i * t^i
                pos[axis] = sum(c[i] * (tau ** i) for i in range(self.num_coeffs))

                # Velocity: v(t) = Σ i * c_i * t^(i-1)
                vel[axis] = sum(i * c[i] * (tau ** (i-1)) for i in range(1, self.num_coeffs))

                # Acceleration: a(t) = Σ i*(i-1) * c_i * t^(i-2)
                acc[axis] = sum(i * (i-1) * c[i] * (tau ** (i-2)) for i in range(2, self.num_coeffs))

            # Interpolate yaw angle
            yaw0 = self.yaw_angles[segment_idx]
            yaw1 = self.yaw_angles[segment_idx + 1]
            alpha = tau / segment['T']
            yaw = yaw0 + alpha * (yaw1 - yaw0)
            yaw_dot = (yaw1 - yaw0) / segment['T']

        return pos, vel, acc, yaw, yaw_dot

    def _find_segment(self, t):
        """
        Find which segment the time t belongs to

        :param t: Current time
        :return: Segment index
        """
        if t < self.times[0]:
            return -1
        if t >= self.times[-1]:
            return len(self.coeffs)

        for i in range(len(self.coeffs)):
            if self.times[i] <= t < self.times[i + 1]:
                return i

        return len(self.coeffs)


def calculate_waypoint_times(waypoints, max_velocity=1.0, time_scale=1.5):
    """
    Automatically calculate arrival times for waypoints based on distance

    :param waypoints: List of 3D positions [[x0,y0,z0], [x1,y1,z1], ...]
    :param max_velocity: Maximum average velocity between waypoints [m/s]
    :param time_scale: Safety margin multiplier (>1.0 for smoother trajectories)
    :return: List of cumulative times [t0, t1, t2, ...]
    """
    waypoints = np.array(waypoints)
    num_waypoints = len(waypoints)
    times = [0.0]  # Start at t=0

    for i in range(1, num_waypoints):
        # Calculate Euclidean distance between consecutive waypoints
        distance = np.linalg.norm(waypoints[i] - waypoints[i-1])

        # Calculate time based on distance and max velocity
        # Add time_scale to ensure smooth deceleration/acceleration
        if distance < 1e-6:  # Very small distance
            segment_time = 1.0  # Minimum 1 second
        else:
            segment_time = (distance / max_velocity) * time_scale

        # Accumulate time
        times.append(times[-1] + segment_time)

    return times


def create_trajectory_from_waypoints(waypoints, times=None, max_velocity=1.0,
                                     time_scale=1.5, yaw_angles=None):
    """
    Create a QP-based minimum snap trajectory from a list of waypoints

    :param waypoints: List of 3D positions [[x0,y0,z0], [x1,y1,z1], ...]
    :param times: List of times to reach each waypoint (optional, auto-calculated if None)
    :param max_velocity: Maximum average velocity for auto time calculation [m/s]
    :param time_scale: Safety margin for auto time calculation (>1.0 for smoother)
    :param yaw_angles: List of yaw angles at each waypoint (optional)
    :return: MinimumSnapTrajectoryQP object
    """
    if times is None:
        # Automatically calculate times based on distance
        times = calculate_waypoint_times(waypoints, max_velocity, time_scale)

    return MinimumSnapTrajectoryQP(waypoints, times, yaw_angles)
