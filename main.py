#!/usr/bin/env python3
"""
Unified simulation main file for drone control
Supports multiple control methods:
- nmpc: NMPC trajectory tracking
- nmpc_regulation: NMPC regulation (setpoint stabilization)
- pd_hgdo: PD control with HGDO (High Gain Disturbance Observer)
- pd_l1: PD control with L1 Adaptation
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
        initial_offset: Initial position offset for regulation tests (optional)
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


def setup_controller(control_type, dynamic_params, drone_params, rotor_params, config):
    """
    Setup the appropriate controller based on control_type

    Returns:
        controller: The control object
        dob: Disturbance observer (if applicable, otherwise None)
    """
    dob = None

    if control_type == 'nmpc' or control_type == 'nmpc_regulation':
        from control.nmpc.ocp.S550_simple_ocp import S550SimpleOcp
        from utils.acados_cleanup import cleanup_acados_files

        nmpc_params = yaml_loader.get_nmpc_params(config)
        controller = S550SimpleOcp(DynParam=dynamic_params,
                                   DroneParam=drone_params,
                                   MpcParam=nmpc_params)

    elif control_type == 'pd_hgdo':
        from control.PID.geometric_control import GeometricControl
        from estimator.dob.hgdo.hgdo import HGDO

        gain_params = yaml_loader.get_pd_gain_params(config)
        controller = GeometricControl(DynamicParams=dynamic_params,
                                     GainParams=gain_params,
                                     DobMode=True)

        config_dob = yaml_loader.load_yaml('config/hgdo.yaml')
        hgdo_params = yaml_loader.get_hgdo_params(config_dob)
        dob = HGDO(DynParam=dynamic_params,
                   DroneParam=drone_params,
                   RotorParam=rotor_params,
                   DobParam=hgdo_params)

    elif control_type == 'pd_l1':
        from control.PID.geometric_control import GeometricControl
        from estimator.dob.l1_adaptation.l1_adaptation import L1Adaptation

        gain_params = yaml_loader.get_pd_gain_params(config)
        controller = GeometricControl(DynamicParams=dynamic_params,
                                     GainParams=gain_params,
                                     DobMode=True)

        config_dob = yaml_loader.load_yaml('config/l1_adaptive.yaml')
        l1_params = yaml_loader.get_l1_adaptation_params(config_dob)
        dob = L1Adaptation(DynParam=dynamic_params,
                          DroneParam=drone_params,
                          RotorParam=rotor_params,
                          DobParam=l1_params)
    else:
        raise ValueError(f"Unknown control type: {control_type}")

    return controller, dob


def compute_control(control_type, controller, dob, s_feedback, ref, t_prev, t_curr,
                   s_rotor, i):
    """
    Compute control input based on controller type

    Returns:
        w_cmd: Commanded rotor speeds
        status: Solver status (for NMPC)
    """
    status = 0

    if control_type == 'nmpc':
        # NMPC trajectory tracking
        status, w_cmd = controller.solve_for_trajectory(s_feedback, t_curr)

    elif control_type == 'nmpc_regulation':
        # NMPC regulation (ref contains setpoint)
        status, w_cmd = controller.solve(s_feedback, ref)

    elif control_type in ['pd_hgdo', 'pd_l1']:
        # PD control with disturbance observer
        if i > 1:
            disturbance_estimate = dob.dob_estimate(t_prev, t_curr,
                                                    s_rotor[:6], s_feedback)
            u = controller.compute_u(s_feedback, ref, disturbance_estimate)
        else:
            u = controller.compute_u(s_feedback, ref)

        # Need hexa_converter to convert u to w_cmd
        # This will be handled in the main loop
        return u, status

    return w_cmd, status


def plot_results(control_type, t_plot, pos_hist, vel_hist, pos_des_hist,
                roll_hist, pitch_hist, yaw_hist, yaw_des_hist,
                w_rotor_hist, alpha_rotor_hist, p_setpoint=None, yaw_setpoint=None):
    """
    Plot simulation results
    """
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    is_regulation = (control_type == 'nmpc_regulation')

    # Plot Position X
    axs[0, 0].plot(t_plot, pos_hist[:, 0], 'b-', label='Actual', linewidth=2)
    if is_regulation:
        axs[0, 0].axhline(y=p_setpoint[0], color='r', linestyle='--', label='Setpoint', linewidth=2)
    else:
        axs[0, 0].plot(t_plot, pos_des_hist[:, 0], 'r--', label='Desired', linewidth=2)
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('X Position [m]')
    axs[0, 0].set_title('X Position Tracking')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Position Y
    axs[0, 1].plot(t_plot, pos_hist[:, 1], 'b-', label='Actual', linewidth=2)
    if is_regulation:
        axs[0, 1].axhline(y=p_setpoint[1], color='r', linestyle='--', label='Setpoint', linewidth=2)
    else:
        axs[0, 1].plot(t_plot, pos_des_hist[:, 1], 'r--', label='Desired', linewidth=2)
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Y Position [m]')
    axs[0, 1].set_title('Y Position Tracking')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Position Z
    axs[0, 2].plot(t_plot, pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    if is_regulation:
        axs[0, 2].axhline(y=p_setpoint[2], color='r', linestyle='--', label='Setpoint', linewidth=2)
    else:
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
    if is_regulation:
        axs[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Velocity [m/s]')
    axs[1, 0].set_title('Velocity')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Attitude Angles
    axs[1, 1].plot(t_plot, roll_hist, 'r-', label='Roll', linewidth=2)
    axs[1, 1].plot(t_plot, pitch_hist, 'g-', label='Pitch', linewidth=2)
    axs[1, 1].plot(t_plot, yaw_hist, 'b-', label='Yaw', linewidth=2)
    if is_regulation:
        axs[1, 1].axhline(y=np.rad2deg(yaw_setpoint), color='b', linestyle='--',
                         alpha=0.5, label='Yaw Setpoint')
    else:
        axs[1, 1].plot(t_plot, np.degrees(yaw_des_hist), 'b--', label='Yaw Desired', linewidth=2)
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Angle [deg]')
    axs[1, 1].set_title('Attitude Angles')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 3D Trajectory
    ax_3d = fig.add_subplot(3, 3, 6, projection='3d')
    ax_3d.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    if is_regulation:
        ax_3d.scatter(p_setpoint[0], p_setpoint[1], p_setpoint[2],
                     c='r', marker='*', s=200, label='Setpoint')
    else:
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
    if is_regulation:
        pos_error_per_axis = np.abs(pos_hist - p_setpoint)
        axs[2, 2].plot(t_plot, pos_error_per_axis[:, 0], 'r-', label='X error', linewidth=2)
        axs[2, 2].plot(t_plot, pos_error_per_axis[:, 1], 'g-', label='Y error', linewidth=2)
        axs[2, 2].plot(t_plot, pos_error_per_axis[:, 2], 'b-', label='Z error', linewidth=2)
        axs[2, 2].set_ylabel('Position Error per Axis [m]')
        axs[2, 2].set_title('Position Error by Axis')
    else:
        pos_error = np.linalg.norm(pos_hist - pos_des_hist, axis=1)
        axs[2, 2].plot(t_plot, pos_error, 'b-', linewidth=2)
        axs[2, 2].set_ylabel('Position Error [m]')
        axs[2, 2].set_title('Position Tracking Error')
    axs[2, 2].set_xlabel('Time [s]')
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    plt.tight_layout()
    plt.show()


def print_statistics(control_type, pos_hist, vel_hist, pos_des_hist, yaw_hist,
                    yaw_des_hist, p_setpoint=None, yaw_setpoint=None):
    """
    Print final simulation statistics
    """
    print("\n========== Simulation Results ==========")
    print(f"Control Type: {control_type.upper()}")
    print(f"Final position: [{pos_hist[-1, 0]:.3f}, {pos_hist[-1, 1]:.3f}, {pos_hist[-1, 2]:.3f}] m")

    if control_type == 'nmpc_regulation':
        pos_error = np.linalg.norm(pos_hist - p_setpoint, axis=1)
        print(f"Setpoint: [{p_setpoint[0]:.3f}, {p_setpoint[1]:.3f}, {p_setpoint[2]:.3f}] m")
        print(f"Final position error: {pos_error[-1]:.6f} m")
        print(f"Mean position error: {np.mean(pos_error):.6f} m")
        print(f"Max position error: {np.max(pos_error):.6f} m")
        print()
        print(f"Final velocity: [{vel_hist[-1, 0]:.6f}, {vel_hist[-1, 1]:.6f}, {vel_hist[-1, 2]:.6f}] m/s")
        print(f"Final velocity magnitude: {np.linalg.norm(vel_hist[-1]):.6f} m/s")
        print()
        print(f"Final yaw: {yaw_hist[-1]:.2f}°, Setpoint: {np.rad2deg(yaw_setpoint):.2f}°")
        print(f"Final yaw error: {abs(yaw_hist[-1] - np.rad2deg(yaw_setpoint)):.4f}°")
    else:
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
    parser = argparse.ArgumentParser(description='Unified drone simulation')
    parser.add_argument('--control', type=str, default='nmpc',
                       choices=['nmpc', 'nmpc_regulation', 'pd_hgdo', 'pd_l1'],
                       help='Control method to use')
    args = parser.parse_args()

    control_type = args.control
    print(f"\n{'='*50}")
    print(f"Running simulation with: {control_type.upper()}")
    print(f"{'='*50}\n")

    # Load appropriate config file
    if control_type == 'nmpc_regulation':
        config_file = 'config/nmpc_regulation_params.yaml'
    elif control_type == 'nmpc':
        config_file = 'config/nmpc_params.yaml'
    else:  # pd_hgdo or pd_l1
        config_file = 'config/pd_params.yaml'

    config = yaml_loader.load_yaml(config_file)
    dynamic_params = yaml_loader.get_dynamic_params(config)
    drone_params = yaml_loader.get_drone_params(config)
    rotor_params = yaml_loader.get_rotor_params(config)
    sim_params = yaml_loader.get_sim_params(config)

    # Create simulation models
    drone_sim_model = S550_Sim_Model(DynamicParams=dynamic_params)
    rotor_sim_model = RotorModel(RotorParams=rotor_params)
    hexa_converter = HexaConverter(DroneParams=drone_params,
                                   RotorParams=rotor_params)

    # Setup controller
    controller, dob = setup_controller(control_type, dynamic_params,
                                      drone_params, rotor_params, config)

    # State initialization
    w_rotor_idle = sim_params['w_rotor_idle']
    if control_type == 'nmpc_regulation':
        # Start with small offset for regulation test
        s_drone, s_rotor = state_initialize(w_rotor_idle, initial_offset=[0.1, 0.1, 0.0])

        # Get regulation parameters
        regulation_params = yaml_loader.get_regulation_params(config)
        p_setpoint = regulation_params['setpoint_position']
        yaw_setpoint = regulation_params['setpoint_yaw']

        # Pack setpoint into reference format
        from utils.reference_packer import reference_packer
        ref_setpoint = reference_packer(p_setpoint, np.zeros(3),
                                       np.array([yaw_setpoint]),
                                       np.array([0.0]))
    else:
        s_drone, s_rotor = state_initialize(w_rotor_idle)
        trajectory_params = yaml_loader.get_trajectory_params(config)

    # Simulation parameters
    tf = sim_params['tf']
    dt = sim_params['dt']
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
    for i in range(N-1):
        # Unpack state
        p, v, q, w = drone_sim_model.unpack_state(s_drone)
        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)
        roll, pitch, yaw = quaternion_to_euler(q)
        s_feedback = np.concatenate([p, v, q, w])

        # Get reference
        if control_type == 'nmpc_regulation':
            ref = ref_setpoint
            p_des = p_setpoint
            yaw_des = yaw_setpoint
        else:
            from ref_generation.ref_generator import get_reference, unpack_ref
            if i == 0:
                ref = get_reference(t_sim[i], trajectory_params)
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

        # Compute control
        if control_type in ['nmpc', 'nmpc_regulation']:
            if control_type == 'nmpc':
                status, w_cmd = controller.solve_for_trajectory(s_feedback, t_sim[i])
            else:
                status, w_cmd = controller.solve(s_feedback, ref)

            if status != 0 and i % 100 == 0:
                print(f"Warning: NMPC solver status {status} at t={t_sim[i]:.2f}s")
        else:
            # PD control with DOB
            if i > 1:
                disturbance_estimate = dob.dob_estimate(t_sim[i-1], t_sim[i],
                                                        s_rotor[:6], s_feedback)
                u = controller.compute_u(s_feedback, ref, disturbance_estimate)
            else:
                u = controller.compute_u(s_feedback, ref)

            w_cmd = hexa_converter.compute_des_rotor_speed(u)

        # Simulation step
        t_ode = [t_sim[i], t_sim[i+1]]

        # Simulate rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)

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
    if control_type == 'nmpc_regulation':
        plot_results(control_type, t_plot, pos_hist, vel_hist, pos_des_hist,
                    roll_hist, pitch_hist, yaw_hist, yaw_des_hist,
                    w_rotor_hist, alpha_rotor_hist,
                    p_setpoint=p_setpoint, yaw_setpoint=yaw_setpoint)
        print_statistics(control_type, pos_hist, vel_hist, pos_des_hist,
                        yaw_hist, yaw_des_hist,
                        p_setpoint=p_setpoint, yaw_setpoint=yaw_setpoint)
    else:
        plot_results(control_type, t_plot, pos_hist, vel_hist, pos_des_hist,
                    roll_hist, pitch_hist, yaw_hist, yaw_des_hist,
                    w_rotor_hist, alpha_rotor_hist)
        print_statistics(control_type, pos_hist, vel_hist, pos_des_hist,
                        yaw_hist, yaw_des_hist)

    # Cleanup for NMPC
    if control_type in ['nmpc', 'nmpc_regulation']:
        from utils.acados_cleanup import cleanup_acados_files
        cleanup_acados_files(controller.get_json_file_name())


if __name__ == '__main__':
    main()
