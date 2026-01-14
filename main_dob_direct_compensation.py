#!/usr/bin/env python3
"""
Unified simulation main file for drone control

Supports multiple control methods and disturbance observers:

Control types:
- nmpc: NMPC trajectory tracking
- pd: PD/Geometric control

DOB types:
- none: No disturbance observer
- hgdo: High Gain Disturbance Observer
- l1: L1 Adaptive Control

Examples:
    python3 main_dob_direct_compensation.py --control nmpc --dob none
    python3 main_dob_direct_compensation.py --control pd --dob hgdo
    python3 main_dob_direct_compensation.py --control pd --dob l1
"""

import numpy as np
import argparse

from sim_model.S550_model import S550_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.math_tool import quaternion_to_euler
from utils import yaml_loader
from utils.custom_ode import custom_rk4
from matplotlib import pyplot as plt


def state_initialize(w_rotor_idle, initial_offset=None):
    """
    Initialize drone state and rotor state

    Args:
        w_rotor_idle: Idle rotor speed
        initial_offset: Initial position offset (optional)
    """
    if initial_offset is not None:
        p = np.array(initial_offset)
    else:
        p = np.zeros((3,))

    v = np.zeros((3,))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros((3,))
    s_drone = np.concatenate([p, v, q, w])

    w_rotor = w_rotor_idle * np.ones((6,))
    alpha_rotor = np.zeros((6,))
    s_rotor = np.concatenate([w_rotor, alpha_rotor])

    return s_drone, s_rotor


def load_parameters(control_type, dob_type):
    """
    Load all necessary parameters from split config files

    IMPORTANT: Separates TRUE parameters (for simulator) from NOMINAL parameters (for control/DOB)
    - TRUE parameters: Used in simulator to represent the actual system
    - NOMINAL parameters: Used in controller and DOB (shared between them)
    This separation allows testing robustness to model uncertainty.

    Args:
        control_type: Type of controller ('nmpc' or 'pd')
        dob_type: Type of DOB ('none', 'hgdo', or 'l1')

    Returns:
        Dictionary containing all loaded parameters with keys:
        - true_*: True parameters for simulator
        - nominal_*: Nominal parameters for controller and DOB
        - sim_params, trajectory_params, control-specific params, dob_params
    """
    params = {}

    # Load TRUE parameters from simulator config (used for simulation model)
    config_sim = yaml_loader.load_yaml('config/simulator/simulator.yaml')
    params['true_dynamic_params'] = yaml_loader.get_dynamic_params(config_sim)
    params['true_drone_params'] = yaml_loader.get_drone_params(config_sim)
    params['true_rotor_params'] = yaml_loader.get_rotor_params(config_sim)
    params['sim_params'] = yaml_loader.get_sim_params(config_sim)

    # Load trajectory parameters (common for trajectory tracking)
    config_traj = yaml_loader.load_yaml('config/trajectory/trajectory.yaml')
    params['trajectory_params'] = yaml_loader.get_trajectory_params(config_traj)

    # Load NOMINAL parameters from control config (used for controller AND DOB)
    if control_type == 'nmpc':
        config_control = yaml_loader.load_yaml('config/control/nmpc/nmpc_params.yaml')
        params['nmpc_params'] = yaml_loader.get_nmpc_params(config_control)

        # Load nominal dynamic params from control config
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)

    elif control_type == 'pd':
        config_control = yaml_loader.load_yaml('config/control/pd/pd_params.yaml')
        params['gain_params'] = yaml_loader.get_pd_gain_params(config_control)

        # Load nominal dynamic params from control config
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)

    # Load DOB-specific parameters (DOB will use nominal params from control config)
    if dob_type == 'hgdo':
        config_dob = yaml_loader.load_yaml('config/estimator/dob/hgdo.yaml')
        params['dob_params'] = yaml_loader.get_hgdo_params(config_dob)
    elif dob_type == 'l1':
        config_dob = yaml_loader.load_yaml('config/estimator/dob/l1_adaptive.yaml')
        params['dob_params'] = yaml_loader.get_l1_adaptation_params(config_dob)

    return params


def setup_controller(control_type, dob_type, params):
    """
    Setup the appropriate controller and disturbance observer

    IMPORTANT: Both controller and DOB use NOMINAL parameters (not true parameters)
    This ensures they share the same system model.

    Args:
        control_type: Type of controller ('nmpc' or 'pd')
        dob_type: Type of DOB ('none', 'hgdo', or 'l1')
        params: Dictionary of parameters (must contain nominal_* parameters)

    Returns:
        controller: The control object (uses nominal parameters)
        dob: Disturbance observer (uses nominal parameters, or None)
    """
    dob = None

    # Setup controller using NOMINAL parameters
    if control_type == 'nmpc':
        from control.nmpc.ocp.S550_simple_ocp import S550SimpleOcp

        controller = S550SimpleOcp(DynParam=params['nominal_dynamic_params'],
                                   DroneParam=params['nominal_drone_params'],
                                   MpcParam=params['nmpc_params'])

    elif control_type == 'pd':
        from control.PID.geometric_control import GeometricControl

        # DobMode = True if dob_type is not 'none'
        use_dob = (dob_type != 'none')
        controller = GeometricControl(DynamicParams=params['nominal_dynamic_params'],
                                     GainParams=params['gain_params'],
                                     DobMode=use_dob)
    else:
        raise ValueError(f"Unknown control type: {control_type}")

    # Setup disturbance observer using NOMINAL parameters (same as controller)
    if dob_type == 'hgdo':
        from estimator.dob.hgdo.hgdo import HGDO

        dob = HGDO(DynParam=params['nominal_dynamic_params'],
                   DroneParam=params['nominal_drone_params'],
                   RotorParam=params['nominal_rotor_params'],
                   DobParam=params['dob_params'])

    elif dob_type == 'l1':
        from estimator.dob.l1_adaptation.l1_adaptation import L1Adaptation

        dob = L1Adaptation(DynParam=params['nominal_dynamic_params'],
                          DroneParam=params['nominal_drone_params'],
                          RotorParam=params['nominal_rotor_params'],
                          DobParam=params['dob_params'])

    elif dob_type == 'none':
        dob = None
    else:
        raise ValueError(f"Unknown DOB type: {dob_type}")

    return controller, dob


def plot_results(control_type, dob_type, t_plot, pos_hist, vel_hist, pos_des_hist,
                roll_hist, pitch_hist, yaw_hist, yaw_des_hist,
                w_rotor_hist, alpha_rotor_hist):
    """
    Plot simulation results
    """
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    # Plot Position X
    axs[0, 0].plot(t_plot, pos_hist[:, 0], 'b-', label='Actual', linewidth=2)
    axs[0, 0].plot(t_plot, pos_des_hist[:, 0], 'r--', label='Desired', linewidth=2)
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('X Position [m]')
    axs[0, 0].set_title('X Position Tracking')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Position Y
    axs[0, 1].plot(t_plot, pos_hist[:, 1], 'b-', label='Actual', linewidth=2)
    axs[0, 1].plot(t_plot, pos_des_hist[:, 1], 'r--', label='Desired', linewidth=2)
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Y Position [m]')
    axs[0, 1].set_title('Y Position Tracking')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Position Z
    axs[0, 2].plot(t_plot, pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    axs[0, 2].plot(t_plot, pos_des_hist[:, 2], 'r--', label='Desired', linewidth=2)
    axs[0, 2].set_xlabel('Time [s]')
    axs[0, 2].set_ylabel('Z Position [m]')
    axs[0, 2].set_title('Z Position Tracking')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Plot Velocity
    axs[1, 0].plot(t_plot, vel_hist[:, 0], 'r-', label='vx', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 1], 'g-', label='vy', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 2], 'b-', label='vz', linewidth=2)
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Velocity [m/s]')
    axs[1, 0].set_title('Velocity')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Attitude Angles
    axs[1, 1].plot(t_plot, roll_hist, 'r-', label='Roll', linewidth=2)
    axs[1, 1].plot(t_plot, pitch_hist, 'g-', label='Pitch', linewidth=2)
    axs[1, 1].plot(t_plot, yaw_hist, 'b-', label='Yaw', linewidth=2)
    axs[1, 1].plot(t_plot, np.degrees(yaw_des_hist), 'b--', label='Yaw Desired', linewidth=2)
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Angle [deg]')
    axs[1, 1].set_title('Attitude Angles')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 3D Trajectory
    ax_3d = fig.add_subplot(3, 3, 6, projection='3d')
    ax_3d.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    ax_3d.plot(pos_des_hist[:, 0], pos_des_hist[:, 1], pos_des_hist[:, 2],
              'r--', label='Desired', linewidth=2)
    ax_3d.scatter(pos_hist[0, 0], pos_hist[0, 1], pos_hist[0, 2],
                 c='g', marker='o', s=100, label='Start')
    ax_3d.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2],
                 c='orange', marker='x', s=100, label='End')
    ax_3d.set_xlabel('X [m]')
    ax_3d.set_ylabel('Y [m]')
    ax_3d.set_zlabel('Z [m]')
    ax_3d.set_title('3D Trajectory')
    ax_3d.legend()
    ax_3d.grid(True)

    # Plot Rotor Speeds
    for j in range(6):
        axs[2, 0].plot(t_plot, w_rotor_hist[:, j], label=f'Rotor {j+1}')
    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 0].set_ylabel('Rotor Speed [RPM]')
    axs[2, 0].set_title('Rotor Speeds')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Plot Rotor Accelerations
    for j in range(6):
        axs[2, 1].plot(t_plot, alpha_rotor_hist[:, j], label=f'Rotor {j+1}')
    axs[2, 1].set_xlabel('Time [s]')
    axs[2, 1].set_ylabel('Rotor Acceleration [RPM/s]')
    axs[2, 1].set_title('Rotor Accelerations')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # Position Error
    pos_error = np.linalg.norm(pos_hist - pos_des_hist, axis=1)
    axs[2, 2].plot(t_plot, pos_error, 'b-', linewidth=2)
    axs[2, 2].set_xlabel('Time [s]')
    axs[2, 2].set_ylabel('Position Error [m]')
    axs[2, 2].set_title('Position Tracking Error')
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    # Add title with control and dob info
    dob_str = dob_type.upper() if dob_type != 'none' else 'No DOB'
    fig.suptitle(f'Control: {control_type.upper()} | DOB: {dob_str}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_statistics(control_type, dob_type, pos_hist, vel_hist, pos_des_hist,
                    yaw_hist, yaw_des_hist):
    """
    Print final simulation statistics
    """
    print("\n========== Simulation Results ==========")
    print(f"Control Type: {control_type.upper()}")
    print(f"DOB Type: {dob_type.upper()}")
    print(f"Final position: [{pos_hist[-1, 0]:.3f}, {pos_hist[-1, 1]:.3f}, {pos_hist[-1, 2]:.3f}] m")

    pos_error = np.linalg.norm(pos_hist - pos_des_hist, axis=1)
    print(f"Final position error: {pos_error[-1]:.4f} m")
    print(f"Mean position error: {np.mean(pos_error):.4f} m")
    print(f"Max position error: {np.max(pos_error):.4f} m")
    print()

    # Yaw error calculation
    yaw_error_rad = yaw_hist - yaw_des_hist
    yaw_error_rad = np.arctan2(np.sin(yaw_error_rad), np.cos(yaw_error_rad))
    yaw_error_deg = np.degrees(yaw_error_rad)

    print(f"Final yaw: {yaw_hist[-1]:.2f}°, Desired: {np.degrees(yaw_des_hist[-1]):.2f}°")
    print(f"Final yaw error: {abs(yaw_error_deg[-1]):.2f}°")
    print(f"Mean yaw error: {np.mean(np.abs(yaw_error_deg)):.2f}°")
    print(f"Max yaw error: {np.max(np.abs(yaw_error_deg)):.2f}°")

    print("========================================\n")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Unified drone simulation with flexible control and DOB selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main_dob_direct_compensation.py --control nmpc --dob none
  python3 main_dob_direct_compensation.py --control pd --dob hgdo
  python3 main_dob_direct_compensation.py --control pd --dob l1
        """)

    parser.add_argument('--control', type=str, default='nmpc',
                       choices=['nmpc', 'pd'],
                       help='Control method: nmpc (NMPC) or pd (PD/Geometric)')
    parser.add_argument('--dob', type=str, default='none',
                       choices=['none', 'hgdo', 'l1'],
                       help='Disturbance observer: none, hgdo (High Gain DOB), or l1 (L1 Adaptive)')

    args = parser.parse_args()

    control_type = args.control
    dob_type = args.dob

    print(f"\n{'='*60}")
    print(f"Control: {control_type.upper()} | DOB: {dob_type.upper()}")
    print(f"{'='*60}\n")

    # Validate combination
    if control_type == 'nmpc' and dob_type != 'none':
        print(f"Warning: NMPC typically doesn't use external DOB. Using DOB: {dob_type}")

    # Load all parameters from split config files
    params = load_parameters(control_type, dob_type)

    print("Parameter Configuration:")
    print("- Simulator: Uses TRUE parameters from config/simulator/simulator.yaml")
    print(f"- Controller & DOB: Use NOMINAL parameters from config/control/{control_type}/")
    print("  (Controller and DOB share the same nominal model)\n")

    # Create simulation models using TRUE parameters (actual system)
    drone_sim_model = S550_Sim_Model(DynamicParams=params['true_dynamic_params'])
    rotor_sim_model = RotorModel(RotorParams=params['true_rotor_params'])
    hexa_converter = HexaConverter(DroneParams=params['true_drone_params'],
                                   RotorParams=params['true_rotor_params'])

    # Setup controller and DOB
    controller, dob = setup_controller(control_type, dob_type, params)

    # State initialization
    w_rotor_idle = params['sim_params']['w_rotor_idle']
    s_drone, s_rotor = state_initialize(w_rotor_idle)

    # Simulation parameters
    tf = params['sim_params']['tf']
    dt = params['sim_params']['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # Data storage
    pos_hist = []
    vel_hist = []
    pos_des_hist = []
    roll_hist = []
    pitch_hist = []
    yaw_hist = []
    yaw_des_hist = []
    w_rotor_hist = []
    alpha_rotor_hist = []

    # Main simulation loop
    from ref_generation.ref_generator import get_reference, unpack_ref

    for i in range(N-1):
        # Unpack state
        p, v, q, w = drone_sim_model.unpack_state(s_drone)
        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)
        roll, pitch, yaw = quaternion_to_euler(q)
        s_feedback = np.concatenate([p, v, q, w])

        # Get reference trajectory
        if i == 0:
            ref = get_reference(t_sim[i], params['trajectory_params'])
        else:
            ref = get_reference(t_sim[i])
        p_des, v_des, yaw_des, yaw_des_dot = unpack_ref(ref)

        # Store history
        pos_hist.append(p.copy())
        vel_hist.append(v.copy())
        pos_des_hist.append(p_des)
        roll_hist.append(np.rad2deg(roll))
        pitch_hist.append(np.rad2deg(pitch))
        yaw_hist.append(np.rad2deg(yaw))
        yaw_des_hist.append(yaw_des)
        w_rotor_hist.append(w_rotor.copy())
        alpha_rotor_hist.append(alpha_rotor.copy())

        if dob is not None and i > 1:
            disturbance_estimate = dob.dob_estimate(t_sim[i-1], t_sim[i],
                                                    s_rotor[:6], s_feedback)
        else:
            disturbance_estimate = np.zeros(6)

        # Compute control
        if control_type == 'nmpc':
            status, w_cmd = controller.solve_for_trajectory(s_feedback, t_sim[i])

            if dob is not None:
                u_mpc = hexa_converter.compute_u(w_cmd)
                u_match = np.array([disturbance_estimate[2],    # fz
                                   disturbance_estimate[3],     # Mx
                                   disturbance_estimate[4],     # My
                                   disturbance_estimate[5]])    # Mz
                u_comp = u_mpc - u_match
                w_cmd = hexa_converter.compute_des_rotor_speed(u_comp)

            if status != 0 and i % 100 == 0:
                print(f"Warning: NMPC solver status {status} at t={t_sim[i]:.2f}s")

        elif control_type == 'pd':
            u = controller.compute_u(s_feedback, ref, disturbance_estimate)

            w_cmd = hexa_converter.compute_des_rotor_speed(u)

        # Simulation step
        t_ode = [t_sim[i], t_sim[i+1]]

        # Simulate rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)
        # Hard clamp
        alpha_max = params['true_rotor_params']['alpha_rotor_max']
        s_rotor[6:] = np.clip(s_rotor[6:], -alpha_max, alpha_max)

        # Convert rotor speeds to control input
        u = hexa_converter.compute_u(s_rotor[0:6])

        # Simulate drone dynamics
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u, t_ode)

    # Post-processing
    pos_hist = np.array(pos_hist)
    vel_hist = np.array(vel_hist)
    pos_des_hist = np.array(pos_des_hist)
    roll_hist = np.array(roll_hist)
    pitch_hist = np.array(pitch_hist)
    yaw_hist = np.array(yaw_hist)
    yaw_des_hist = np.array(yaw_des_hist)
    w_rotor_hist = np.array(w_rotor_hist)
    alpha_rotor_hist = np.array(alpha_rotor_hist)
    t_plot = t_sim[:-1]

    # Plot results
    plot_results(control_type, dob_type, t_plot, pos_hist, vel_hist, pos_des_hist,
                roll_hist, pitch_hist, yaw_hist, yaw_des_hist,
                w_rotor_hist, alpha_rotor_hist)

    # Print statistics
    print_statistics(control_type, dob_type, pos_hist, vel_hist, pos_des_hist,
                    yaw_hist, yaw_des_hist)

    # Cleanup for NMPC
    if control_type == 'nmpc':
        from utils.acados_cleanup import cleanup_acados_files
        cleanup_acados_files(controller.get_json_file_name())


if __name__ == '__main__':
    main()
