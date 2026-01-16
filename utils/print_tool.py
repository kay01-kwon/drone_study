import numpy as np

def print_statistics(control_type, dob_type,
                     drone_state, rotor_state,
                     ref, dob_est, param_est,
                     true_dynamic_params):
    """
    Print final simulation statistics including parameter estimates
    """
    # Get time
    t_plot = drone_state['t']

    # Get drone state
    pos = drone_state['pos']
    vel = drone_state['vel']
    roll = drone_state['roll']
    pitch = drone_state['pitch']
    yaw = drone_state['yaw']

    # Get rotor state
    w_rotor = rotor_state['w_rotor']
    alpha_rotor = rotor_state['alpha_rotor']

    # Get reference
    pos_des = ref['pos_des']
    vel_des = ref['vel_des']
    yaw_des = ref['yaw_des']

    # Get disturbance estimate
    f_est = dob_est['f_est']
    tau_est = dob_est['tau_est']

    # Get dynamic parameter estimate
    m_est = param_est['m_est']
    com_x_est = param_est['com_x_est']
    com_y_est = param_est['com_y_est']

    print("\n========== Simulation Results ==========")
    print(f"Control Type: {control_type.upper()}")
    print(f"DOB Type: {dob_type.upper()}")
    print(f"Final position: [{pos[-1, 0]:.3f}, {pos[-1, 1]:.3f}, {pos[-1, 2]:.3f}] m")

    pos_error = np.linalg.norm(pos - pos_des, axis=1)
    print(f"Final position error: {pos_error[-1]:.4f} m")
    print(f"Mean position error: {np.mean(pos_error):.4f} m")
    print(f"Max position error: {np.max(pos_error):.4f} m")

    # Yaw error calculation
    yaw_error_rad = yaw - yaw_des
    yaw_error_rad = np.arctan2(np.sin(yaw_error_rad), np.cos(yaw_error_rad))
    yaw_error_deg = np.degrees(yaw_error_rad)

    print(f"Final yaw: {yaw[-1]:.2f}°, Desired: {np.degrees(yaw_des[-1]):.2f}°")
    print(f"Final yaw error: {abs(yaw_error_deg[-1]):.2f}°")
    print(f"Mean yaw error: {np.mean(np.abs(yaw_error_deg)):.2f}°")
    print(f"Max yaw error: {np.max(np.abs(yaw_error_deg)):.2f}°")
    print()

    true_mass = true_dynamic_params['m']
    true_com_offset = np.array([true_dynamic_params['com_offset'][0],
                                true_dynamic_params['com_offset'][1]])

    # Parameter estimation results
    print("========== Parameter Estimation ==========")
    print(f"True mass: {true_mass:.4f} kg")
    print(f"Final estimated mass: {m_est[-1]:.4f} kg")
    print(f"Mass estimation error: {abs(m_est[-1] - true_mass):.4f} kg ({abs(m_est[-1] - true_mass)/true_mass*100:.2f}%)")
    print(f"True COM offset: [{true_com_offset[0]:.4f}, {true_com_offset[1]:.4f}] m")
    print(f"Final estimated COM: [{com_x_est[-1]:.4f}, {com_y_est[-1]:.4f}, N/A] m")
    print(f"COM X estimation error: {abs(com_x_est[-1] - true_com_offset[0]):.4f} m")
    print(f"COM Y estimation error: {abs(com_y_est[-1] - true_com_offset[1]):.4f} m")

    print("========================================\n")