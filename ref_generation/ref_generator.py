import numpy as np
from utils.reference_packer import reference_packer
from ref_generation.minimum_snap import create_trajectory_from_waypoints

# Define waypoints for the trajectory
# Format: [x, y, z]
WAYPOINTS = [
    [0.0, 0.0, 0.0],      # Start at origin
    [0.0, 0.0, 1.0],      # Takeoff to 1m
    [0.5, 0.5, 1.5],      # Move to [0.5, 0.5, 1.5]
    [0.5, -0.5, 1.0],     # Move to [0.5, -0.5, 1.0]
    [0.0, 0.0, 1.0],      # Return to hover position
]

# Yaw angles at each waypoint (in radians)
# Test different yaw orientations at each waypoint
YAW_ANGLES = [
    0.0,           # Start facing forward (0°)
    0.0,           # Maintain orientation during takeoff
    0.0,       # Turn 9° at [0.5, 0.5, 1.5]
    0.0,      # Turn 9° at [0.5, -0.5, 1.0]
    0.0            # Return to forward orientation
]

# Create global trajectory object
_trajectory = None
_trajectory_params = None

def initialize_trajectory(trajectory_params):
    """
    Initialize the minimum snap trajectory with automatic time calculation

    :param trajectory_params: Dictionary with 'max_velocity', 'max_yaw_rate', and 'time_scale'
    """
    global _trajectory, _trajectory_params
    _trajectory_params = trajectory_params

    _trajectory = create_trajectory_from_waypoints(
        waypoints=WAYPOINTS,
        times=None,  # Let the function calculate times automatically
        max_velocity=trajectory_params['max_velocity'],
        time_scale=trajectory_params['time_scale'],
        yaw_angles=YAW_ANGLES,
        max_yaw_rate=trajectory_params['max_yaw_rate']
    )

    # Print calculated waypoint times for debugging
    print("\n========== Trajectory Generation ==========")
    print(f"Max velocity: {trajectory_params['max_velocity']:.2f} m/s")
    print(f"Max yaw rate: {trajectory_params['max_yaw_rate']:.2f} rad/s")
    print(f"Time scale: {trajectory_params['time_scale']:.2f}")
    print("Waypoints:")
    for i, wp in enumerate(WAYPOINTS):
        yaw_deg = np.degrees(YAW_ANGLES[i])
        print(f"  WP{i}: {wp} at t={_trajectory.times[i]:.2f}s, yaw={yaw_deg:.1f}°")
    print(f"Total trajectory duration: {_trajectory.times[-1]:.2f}s")
    print("==========================================\n")

def get_reference(t_curr, trajectory_params=None):
    """
    Get desired reference trajectory at current time using minimum snap trajectory
    :param t_curr: Current time
    :param trajectory_params: Dictionary with 'max_velocity', 'max_yaw_rate', and 'time_scale' (optional)
    :return: Reference state [p_des, v_des, yaw_des, yaw_des_dot]
    """
    global _trajectory

    # Initialize trajectory if not done yet
    if _trajectory is None:
        if trajectory_params is None:
            # Use default values if not provided
            trajectory_params = {'max_velocity': 1.0, 'max_yaw_rate': 1.0, 'time_scale': 1.5}
        initialize_trajectory(trajectory_params)

    # Get state from minimum snap trajectory
    p_des, v_des, a_des, yaw_des, yaw_des_dot = _trajectory.get_state(t_curr)

    ref = reference_packer(p_des, v_des, np.array([yaw_des]), np.array([yaw_des_dot]))
    return ref

def reset_trajectory():
    """Reset the global trajectory object"""
    global _trajectory, _trajectory_params
    _trajectory = None
    _trajectory_params = None

def unpack_ref(ref):
    p_des = ref[0:3]
    v_des = ref[3:6]
    yaw_des = ref[6]
    yaw_des_dot = ref[7]
    return p_des, v_des, yaw_des, yaw_des_dot
