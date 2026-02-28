#!/usr/bin/env python3
"""
6DOF Drone Simulation with Front/Rear Contact Dynamics and Hehn Trajectory

Supports multiple control methods, disturbance observers and dynamic parameter estimator:

Control types:
- nmpc: NMPC with Hehn trajectory tracking
- pd: Geometric control

DOB types:
- none: No disturbance observer
- hgdo: High Gain Disturbance Observer
- l1: L1 Adaptation

Examples:
    python3 main_control.py --control nmpc --dob none --mode regulation
    python3 main_control.py --control nmpc --dob l1 --mode tracking
    python3 main_control.py --control pd --dob hgdo --mode tracking

Author: Geonwoo Kwon
Date: 2026-01-16
"""

import numpy as np
import argparse
import time

from estimator.rls.dynamic_param_estimator import DynamicParamEstimator
from utils import parameter_loader
from utils.plot_sim_result import plot_results

from utils.state_initializer import state_initialize
from sim_model.S550_model import S550_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.math_tool import quaternion_to_euler, quaternion_to_rotm
from utils.custom_ode import custom_rk4
from utils.print_tool import print_statistics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='6DOF Control simulation with Hehn trajectory and contact dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python3 main_control.py --control nmpc --dob none --mode regulation
        python3 main_control.py --control nmpc --dob hgdo --mode tracking
        python3 main_control.py --control nmpc --dob l1 --mode tracking
        python3 main_control.py --control pd --dob hgdo --mode tracking
        """
    )

    parser.add_argument('--control', type=str, default='nmpc',
                       choices=['nmpc', 'pd'],
                       help='Control method: nmpc (DOB Compensation), pd (Geometric)')

    parser.add_argument('--dob', type=str, default='l1',
                       choices=['none', 'hgdo', 'l1'],
                       help='Disturbance observer: none, hgdo (High Gain DOB), or l1 (L1 Adaptive)')

    parser.add_argument('--mode', type=str, default='tracking',
                        choices=['tracking', 'regulation'],
                        help='Control mode: tracking (Hehn trajectory) or regulation (fixed hover)')

    args = parser.parse_args()

    control_type = args.control
    dob_type = args.dob
    control_mode = args.mode

    print(f"\n{'='*60}")
    print(f"6DOF Simulation with Contact Dynamics")
    print(f"Control: {control_type}, DOB: {dob_type}, Mode: {control_mode}")
    print(f"{'='*60}")

    # Load all parameters from split config files
    params = parameter_loader.load_parameters(control_type, dob_type)

    print("\nParameter Configuration:")
    print("- Simulator: Uses TRUE parameters from config/simulator/simulator.yaml")
    print(f"- Controller & DOB: Use NOMINAL parameters from config/control/{control_type}/")
    print(f"- CoM offset: {params['true_dynamic_params']['com_offset']}")

    # Create simulation models using TRUE parameters (actual system)
    drone_sim_model = S550_Sim_Model(DynamicParams=params['true_dynamic_params'])
    # For hard clamp, store maximum rotor acceleration
    alpha_max = params['true_rotor_params']['alpha_rotor_max']
    num_rotors = params['true_rotor_params']['num_rotors']

    rotor_sim_model = RotorModel(RotorParams=params['true_rotor_params'])
    hexa_converter = HexaConverter(DroneParams=params['true_drone_params'],
                                   RotorParams=params['true_rotor_params'],
                                   Dim=6)

    # Setup controller, DOB and RLS
    controller, dob = parameter_loader.setup_controller(control_type, dob_type, params)
    dynamic_param_estimator = DynamicParamEstimator(DynParam=params['nominal_dynamic_params'],
                                                    DroneParam=params['nominal_drone_params'],
                                                    RotorParam=params['nominal_rotor_params'],
                                                    RlsParam=params['rls_params'])

    # Setup tracking mode for NMPC with Hehn trajectory
    if control_mode == 'tracking' and control_type == 'nmpc':
        target_3d = params['tracking_params']['target_position']
        controller.setup_tracking(params['tracking_params'], target_3d)
        replan_freq = params['tracking_params'].get('replan_freq', 100.0)
        print(f"\nTracking: Hehn trajectory, replan at {replan_freq:.0f}Hz")
        print(f"  Target: {target_3d}")

    # State initialization
    w_rotor_idle = params['sim_params']['w_rotor_idle']

    # Initialize at ground level (like Gazebo)
    initial_pos = [0.0, 0.0, 0.0]  # Ground level
    s_drone, s_rotor = state_initialize(w_rotor_idle,
                                        initial_offset=initial_pos,
                                        Dim=6)

    # Set initial height offset for coordinate transformation (like PX4 hg offset)
    # This makes z=0 at ground level (body origin), matching mocap/PX4 convention
    drone_sim_model.set_initial_height_offset(s_drone)

    # Simulation parameters
    tf = params['sim_params']['tf']
    dt = params['sim_params']['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    print(f"Simulation time: {tf:.1f} s, dt: {dt*1000:.1f} ms\n")

    # Data storage
    pos_hist = []
    vel_hist = []

    pos_des_hist = []
    vel_des_hist = []

    roll_hist = []
    pitch_hist = []
    yaw_hist = []

    yaw_des_hist = []
    yaw_des_dot_hist = []

    w_rotor_hist = []
    alpha_rotor_hist = []

    f_est_hist = []
    tau_est_hist = []

    nmpc_solve_times = []

    # Parameter estimate storage
    m_est_hist = []
    com_x_est_hist = []
    com_y_est_hist = []

    # For regulation mode or PD tracking, use existing ref_generator
    if control_mode == 'regulation' or control_type == 'pd':
        from ref_generation.ref_generator import get_reference, unpack_ref

    # Print control mode information
    if control_mode == 'regulation':
        if 'nmpc_regulation_params' in params:
            p_target = params['nmpc_regulation_params']['setpoint_position']
            yaw_target = params['nmpc_regulation_params']['setpoint_yaw']
        else:
            p_target = params['regulation_params']['setpoint_position']
            yaw_target = params['regulation_params']['setpoint_yaw']
        print(f"\nRegulation: target position {p_target}")
        print(f"            target yaw: {yaw_target:.2f} rad")

    # Main simulation loop
    for i in range(N-1):
        # Get transformed state (like mavros/local_position/odom)
        # Position: World frame (Body origin), Velocity: Body frame
        s_transformed = drone_sim_model.get_transformed_state(s_drone)

        # Unpack transformed state
        p = s_transformed[0:3]      # Position (World, Body origin, height offset applied)
        v = s_transformed[3:6]      # Velocity (Body frame)
        q = s_transformed[6:10]     # Quaternion
        w = s_transformed[10:13]    # Angular velocity (Body frame)

        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)
        roll, pitch, yaw = quaternion_to_euler(q)
        s_feedback = np.concatenate([p, v, q, w])

        # Convert body velocity to world velocity for trajectory generation
        R = quaternion_to_rotm(q)
        v_world = R @ v

        # Get reference (tracking or regulation)
        t_now = t_sim[i]

        if control_mode == 'tracking' and control_type == 'nmpc':
            # Hehn trajectory tracking - p_des, v_des come from solver
            yaw_des = 0.0
            yaw_des_dot = 0.0
        elif control_mode == 'tracking':
            # PD tracking with existing ref_generator
            if i == 0:
                ref = get_reference(t_now, params['trajectory_params'])
            else:
                ref = get_reference(t_now)
            p_des, v_des, yaw_des, yaw_des_dot = unpack_ref(ref)
        else:  # regulation
            if 'nmpc_regulation_params' in params:
                p_des = params['nmpc_regulation_params']['setpoint_position']
                yaw_des = params['nmpc_regulation_params']['setpoint_yaw']
            else:
                p_des = params['regulation_params']['setpoint_position']
                yaw_des = params['regulation_params']['setpoint_yaw']
            v_des = np.zeros(3)
            yaw_des_dot = 0.0

        # DOB estimate
        if dob is not None and i > 1:
            d_est = dob.dob_estimate(t_sim[i-1], t_sim[i],
                                     s_rotor[:6], s_feedback)
            # Update RLS only when DOB is active
            dynamic_param_estimator.update(s_feedback, d_est, s_rotor[:6])
        else:
            d_est = np.zeros(6)

        param_est = dynamic_param_estimator.get_parameter_estimate()

        f_est = d_est[0:3]
        tau_est = d_est[3:6]

        # Compute control
        if control_type == 'nmpc':
            t_solve_start = time.perf_counter()

            if control_mode == 'tracking':
                # Hehn trajectory tracking
                state_3d = np.array([p[0], p[1], p[2],
                                     v_world[0], v_world[1], v_world[2]])
                status, w_cmd, p_des, v_des = controller.solve_for_hehn_trajectory(
                    s_feedback, state_3d, t_now)
            else:
                # Regulation mode - build reference
                ref = np.zeros(8)
                ref[0:3] = p_des
                ref[3:6] = v_des
                ref[6] = yaw_des
                ref[7] = yaw_des_dot
                status, w_cmd = controller.solve(s_feedback, ref)

            t_solve_end = time.perf_counter()
            nmpc_solve_times.append(t_solve_end - t_solve_start)

            # DOB compensation
            if dob is not None:
                u_mpc = hexa_converter.compute_u(w_cmd)
                u_d_match = np.array([d_est[2],
                                      d_est[3],
                                      d_est[4],
                                      d_est[5]])
                u_comp = u_mpc - u_d_match
                w_cmd = hexa_converter.compute_des_rotor_speed(u_comp)

            if status != 0 and i % 100 == 0:
                print(f"Warning: NMPC solver status {status} at t = {t_now:.2f}s")

        elif control_type == 'pd':
            if control_mode == 'tracking':
                ref = np.concatenate([p_des, v_des, [yaw_des], [yaw_des_dot]])
            else:
                ref = np.concatenate([p_des, v_des, [yaw_des], [yaw_des_dot]])
            u = controller.compute_u(s_feedback, ref, d_est)
            w_cmd = hexa_converter.compute_des_rotor_speed(u)

        # Store history
        pos_hist.append(p.copy())
        vel_hist.append(v.copy())

        pos_des_hist.append(p_des.copy() if isinstance(p_des, np.ndarray) else np.array(p_des))
        vel_des_hist.append(v_des.copy() if isinstance(v_des, np.ndarray) else np.array(v_des))

        roll_hist.append(roll)
        pitch_hist.append(pitch)
        yaw_hist.append(yaw)

        yaw_des_hist.append(yaw_des)
        yaw_des_dot_hist.append(yaw_des_dot)

        w_rotor_hist.append(w_rotor.copy())
        alpha_rotor_hist.append(alpha_rotor.copy())

        f_est_hist.append(f_est.copy())
        tau_est_hist.append(tau_est.copy())

        m_est_hist.append(param_est[0])
        com_x_est_hist.append(param_est[1])
        com_y_est_hist.append(param_est[2])

        # Simulation step
        t_ode = [t_sim[i], t_sim[i+1]]

        # Simulate rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)

        # Hard clamp for rotor acceleration
        s_rotor[num_rotors:] = np.clip(s_rotor[num_rotors:], -alpha_max, alpha_max)
        u = hexa_converter.compute_u(s_rotor[:num_rotors])

        # Simulate drone dynamics
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u, t_ode)

        # Prevent ground penetration
        s_drone = drone_sim_model.clamp_ground(s_drone)

        # Print progress
        if i % 1000 == 0:
            print(f"t={t_now:.2f}s, z={p[2]:.3f}m, pitch={np.rad2deg(pitch):.2f}deg, "
                  f"w_rotor=[{w_rotor[0]:.0f}, {w_rotor[1]:.0f}, ...]")

    # Post-processing (Convert to numpy array)
    drone_state_data = {'t': t_sim[:-1],
                        'pos': np.array(pos_hist),
                        'vel': np.array(vel_hist),
                        'roll': np.array(roll_hist),
                        'pitch': np.array(pitch_hist),
                        'yaw': np.array(yaw_hist)}

    rotor_state_data = {'w_rotor': np.array(w_rotor_hist),
                        'alpha_rotor': np.array(alpha_rotor_hist)}

    ref_data = {'pos_des': np.array(pos_des_hist),
                'vel_des': np.array(vel_des_hist),
                'yaw_des': np.array(yaw_des_hist),
                'yaw_des_dot': np.array(yaw_des_dot_hist)}

    dob_data = {'f_est': np.array(f_est_hist),
                'tau_est': np.array(tau_est_hist)}

    param_est_data = {'m_est': np.array(m_est_hist),
                      'com_x_est': np.array(com_x_est_hist),
                      'com_y_est': np.array(com_y_est_hist)}

    # Print final statistics
    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"Final position: x={drone_state_data['pos'][-1, 0]:.3f}, "
          f"y={drone_state_data['pos'][-1, 1]:.3f}, "
          f"z={drone_state_data['pos'][-1, 2]:.3f} m")
    print(f"Final attitude: roll={np.rad2deg(drone_state_data['roll'][-1]):.2f}, "
          f"pitch={np.rad2deg(drone_state_data['pitch'][-1]):.2f}, "
          f"yaw={np.rad2deg(drone_state_data['yaw'][-1]):.2f} deg")

    # NMPC solve time statistics
    if control_type == 'nmpc' and len(nmpc_solve_times) > 0:
        solve_times_ms = np.array(nmpc_solve_times) * 1000
        print(f"\nNMPC Solve Time Statistics:")
        print(f"  Mean:   {np.mean(solve_times_ms):.3f} ms")
        print(f"  Std:    {np.std(solve_times_ms):.3f} ms")
        print(f"  Max:    {np.max(solve_times_ms):.3f} ms")
        print(f"  Min:    {np.min(solve_times_ms):.3f} ms")

        print(f"\n  Sim dt: {dt*1000:.1f} ms")
        realtime_ratio = (dt * 1000) / np.mean(solve_times_ms)
        print(f"  Real-time ratio: {realtime_ratio:.1f}x (>1 means real-time capable)")

    print(f"{'='*60}")

    plot_results(control_type, dob_type, drone_state_data,
                 rotor_state_data, ref_data, dob_data, param_est_data,
                 params['true_dynamic_params'])

    print_statistics(control_type, dob_type,
                     drone_state_data, rotor_state_data,
                     ref_data, dob_data, param_est_data,
                     params['true_dynamic_params'])

    # Cleanup for NMPC
    if control_type == 'nmpc':
        from utils.acados_cleanup import cleanup_acados_files
        cleanup_acados_files(controller.get_json_file_name())



if __name__ == '__main__':
    main()
