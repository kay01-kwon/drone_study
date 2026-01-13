import numpy as np

class MinimumSnapTrajectory:
    """
    Minimum snap trajectory generator using 5th order polynomials.
    For each segment between waypoints, generates smooth trajectory with
    zero velocity and acceleration at waypoints.
    """

    def __init__(self, waypoints, times, yaw_angles=None):
        """
        Initialize minimum snap trajectory generator

        :param waypoints: List of 3D waypoints [[x0,y0,z0], [x1,y1,z1], ...]
        :param times: List of times to reach each waypoint [t0, t1, t2, ...]
        :param yaw_angles: List of yaw angles at each waypoint (optional)
        """
        self.waypoints = np.array(waypoints)
        self.times = np.array(times)
        self.num_waypoints = len(waypoints)

        if yaw_angles is None:
            self.yaw_angles = np.zeros(self.num_waypoints)
        else:
            self.yaw_angles = np.array(yaw_angles)

        # Compute polynomial coefficients for each segment
        self.coeffs = self._compute_coefficients()

        # Compute yaw polynomial coefficients
        self.yaw_coeffs = self._compute_yaw_coefficients()

    def _compute_coefficients(self):
        """
        Compute 5th order polynomial coefficients for each segment
        Constraints: position, velocity, acceleration at start and end
        """
        num_segments = self.num_waypoints - 1
        coeffs = []

        for i in range(num_segments):
            p0 = self.waypoints[i]
            pf = self.waypoints[i + 1]
            t0 = self.times[i]
            tf = self.times[i + 1]
            T = tf - t0

            # For each axis (x, y, z), compute coefficients
            segment_coeffs = np.zeros((3, 6))  # 3 axes, 6 coefficients each

            for axis in range(3):
                # Boundary conditions:
                # p(0) = p0, p(T) = pf
                # v(0) = 0, v(T) = 0
                # a(0) = 0, a(T) = 0

                # Solve linear system: A * coeffs = b
                A = np.array([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [1, T, T**2, T**3, T**4, T**5],
                    [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                    [0, 0, 2, 6*T, 12*T**2, 20*T**3]
                ])

                b = np.array([
                    p0[axis],  # p(0) = p0
                    0,         # v(0) = 0
                    0,         # a(0) = 0
                    pf[axis],  # p(T) = pf
                    0,         # v(T) = 0
                    0          # a(T) = 0
                ])

                segment_coeffs[axis] = np.linalg.solve(A, b)

            coeffs.append({
                'coeffs': segment_coeffs,
                't0': t0,
                'tf': tf,
                'T': T
            })

        return coeffs

    def _compute_yaw_coefficients(self):
        """
        Compute 3rd order (cubic) polynomial coefficients for yaw trajectory

        Uses 3rd order polynomial because yaw is controlled by torque, not thrust.
        This provides smooth yaw motion with continuous acceleration (minimum jerk).

        4 boundary conditions uniquely determine 4 coefficients of cubic polynomial:
        - yaw(0) = yaw0, yaw(T) = yaw1
        - yaw_dot(0) = 0, yaw_dot(T) = 0

        This is the standard approach for smooth orientation interpolation.
        """
        num_segments = self.num_waypoints - 1
        yaw_coeffs = []

        for i in range(num_segments):
            yaw0 = self.yaw_angles[i]
            yaw1 = self.yaw_angles[i + 1]
            t0 = self.times[i]
            tf = self.times[i + 1]
            T = tf - t0

            # Handle angle wrapping: find shortest path
            dyaw = yaw1 - yaw0
            dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # Wrap to [-pi, pi]
            yaw1 = yaw0 + dyaw  # Adjusted target yaw for shortest path

            # Boundary conditions (4 constraints for cubic polynomial):
            # yaw(0) = yaw0, yaw(T) = yaw1
            # yaw_dot(0) = 0, yaw_dot(T) = 0

            A = np.array([
                [1, 0, 0, 0],              # yaw(0) = yaw0
                [0, 1, 0, 0],              # yaw_dot(0) = 0
                [1, T, T**2, T**3],        # yaw(T) = yaw1
                [0, 1, 2*T, 3*T**2]        # yaw_dot(T) = 0
            ])

            b = np.array([
                yaw0,  # yaw(0) = yaw0
                0,     # yaw_dot(0) = 0
                yaw1,  # yaw(T) = yaw1
                0      # yaw_dot(T) = 0
            ])

            coeffs = np.linalg.solve(A, b)

            yaw_coeffs.append({
                'coeffs': coeffs,
                't0': t0,
                'tf': tf,
                'T': T
            })

        return yaw_coeffs

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
                # Position: p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
                pos[axis] = c[0] + c[1]*tau + c[2]*tau**2 + c[3]*tau**3 + c[4]*tau**4 + c[5]*tau**5
                # Velocity: v(t) = c1 + 2*c2*t + 3*c3*t^2 + 4*c4*t^3 + 5*c5*t^4
                vel[axis] = c[1] + 2*c[2]*tau + 3*c[3]*tau**2 + 4*c[4]*tau**3 + 5*c[5]*tau**4
                # Acceleration: a(t) = 2*c2 + 6*c3*t + 12*c4*t^2 + 20*c5*t^3
                acc[axis] = 2*c[2] + 6*c[3]*tau + 12*c[4]*tau**2 + 20*c[5]*tau**3

            # Compute yaw using 3rd order (cubic) polynomial
            yaw_segment = self.yaw_coeffs[segment_idx]
            c = yaw_segment['coeffs']
            # Yaw: yaw(t) = c0 + c1*t + c2*t^2 + c3*t^3
            yaw = c[0] + c[1]*tau + c[2]*tau**2 + c[3]*tau**3
            # Yaw rate: yaw_dot(t) = c1 + 2*c2*t + 3*c3*t^2
            yaw_dot = c[1] + 2*c[2]*tau + 3*c[3]*tau**2

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


def calculate_waypoint_times(waypoints, max_velocity=1.0, time_scale=1.5,
                            yaw_angles=None, max_yaw_rate=1.0):
    """
    Automatically calculate arrival times for waypoints based on distance and yaw change

    :param waypoints: List of 3D positions [[x0,y0,z0], [x1,y1,z1], ...]
    :param max_velocity: Maximum average velocity between waypoints [m/s]
    :param time_scale: Safety margin multiplier (>1.0 for smoother trajectories)
    :param yaw_angles: List of yaw angles at each waypoint (optional) [rad]
    :param max_yaw_rate: Maximum yaw rate [rad/s]
    :return: List of cumulative times [t0, t1, t2, ...]
    """
    waypoints = np.array(waypoints)
    num_waypoints = len(waypoints)
    times = [0.0]  # Start at t=0

    if yaw_angles is not None:
        yaw_angles = np.array(yaw_angles)

    for i in range(1, num_waypoints):
        # Calculate Euclidean distance between consecutive waypoints
        distance = np.linalg.norm(waypoints[i] - waypoints[i-1])

        # Calculate time based on distance and max velocity
        # Add time_scale to ensure smooth deceleration/acceleration
        if distance < 1e-6:  # Very small distance
            time_position = 1.0  # Minimum 1 second
        else:
            time_position = (distance / max_velocity) * time_scale

        # Calculate time based on yaw change if yaw angles provided
        if yaw_angles is not None:
            dyaw = yaw_angles[i] - yaw_angles[i-1]
            # Wrap angle difference to [-pi, pi]
            dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
            time_yaw = (abs(dyaw) / max_yaw_rate) * time_scale
        else:
            time_yaw = 0.0

        # Use the maximum of position time and yaw time
        segment_time = max(time_position, time_yaw)

        # Accumulate time
        times.append(times[-1] + segment_time)

    return times


def create_trajectory_from_waypoints(waypoints, times=None, max_velocity=1.0,
                                     time_scale=1.5, yaw_angles=None, max_yaw_rate=1.0):
    """
    Create a minimum snap trajectory from a list of waypoints

    :param waypoints: List of 3D positions [[x0,y0,z0], [x1,y1,z1], ...]
    :param times: List of times to reach each waypoint (optional, auto-calculated if None)
    :param max_velocity: Maximum average velocity for auto time calculation [m/s]
    :param time_scale: Safety margin for auto time calculation (>1.0 for smoother)
    :param yaw_angles: List of yaw angles at each waypoint (optional) [rad]
    :param max_yaw_rate: Maximum yaw rate for auto time calculation [rad/s]
    :return: MinimumSnapTrajectory object
    """
    if times is None:
        # Automatically calculate times based on distance and yaw change
        times = calculate_waypoint_times(waypoints, max_velocity, time_scale,
                                        yaw_angles, max_yaw_rate)

    return MinimumSnapTrajectory(waypoints, times, yaw_angles)
