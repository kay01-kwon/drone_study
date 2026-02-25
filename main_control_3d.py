#!/usr/bin/env python3
"""
3DOF (x, z, theta) Drone Simulation with Contact Dynamics

Supports:
- PD control with pitch
- NMPC control (3DOF acados)
- HGDO disturbance observer
- Regulation and tracking modes
- Hehn trajectory for tracking

Examples:
    python3 main_control_3d.py --control pd --dob hgdo --mode regulation
    python3 main_control_3d.py --control nmpc --dob none --mode regulation
    python3 main_control_3d.py --control nmpc --dob hgdo --mode tracking

Author: Geonwoo Kwon
Date: 2026-02-11
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

from utils import yaml_loader
from utils.state_initializer import state_initialize
from sim_model.S550_3d_model import S550_3D_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.custom_ode import custom_rk4


def load_parameters_3d(control_type, dob_type):
    """Load parameters for 3DOF simulation"""
    params = {}

    # Load True parameters from simulator_3d config
    config_sim = yaml_loader.load_yaml('config/simulator/simulator_3d.yaml')
    params['true_dynamic_params'] = yaml_loader.get_dynamic_params(config_sim)
    params['true_drone_params'] = yaml_loader.get_drone_params(config_sim)
    params['true_rotor_params'] = yaml_loader.get_rotor_params(config_sim)
    params['sim_params'] = yaml_loader.get_sim_params(config_sim)

    # Load NOMINAL parameters from control config
    if control_type == 'nmpc':
        config_control = yaml_loader.load_yaml('config/control/nmpc/nmpc_3d_params.yaml')
        params['nmpc_params'] = yaml_loader.get_nmpc_params(config_control)
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)

        # Regulation params
        regulation = config_control.get('regulation', {})
        params['regulation_params'] = {
            'setpoint_position': np.array(regulation.get('setpoint_position', [0.0, 1.0]))
        }

        # Tracking params
        tracking = config_control.get('tracking', {})
        params['tracking_params'] = {
            'target_position': np.array(tracking.get('target_position', [2.0, 1.0])),
            'a_max': tracking.get('a_max', 12.0),
            'a_min': tracking.get('a_min', 3.0),
            'omega_xy_max': tracking.get('omega_xy_max', 3.0),
            'time_scale': tracking.get('time_scale', 5.0),
        }

    elif control_type == 'pd':
        config_control = yaml_loader.load_yaml('config/control/pd/pd_3d_params.yaml')
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)
        params['gain_params'] = yaml_loader.get_pd_gain_params(config_control)

        # Default regulation for PD
        params['regulation_params'] = {
            'setpoint_position': np.array([0.0, 1.0])
        }
        params['tracking_params'] = {
            'target_position': np.array([2.0, 1.0]),
            'a_max': 12.0,
            'a_min': 3.0,
            'omega_xy_max': 3.0,
            'time_scale': 5.0,
        }

    # Load DOB parameters
    if dob_type == 'hgdo':
        config_dob = yaml_loader.load_yaml('config/estimator/dob/hgdo_3d.yaml')
        params['dob_params'] = yaml_loader.get_hgdo_params(config_dob)

    return params


def setup_controller_3d(control_type, dob_type, params):
    """Setup controller and DOB for 3DOF"""
    dob = None

    if control_type == 'nmpc':
        from control.nmpc.ocp.S550_3DOF_ocp import S550_3DOF_ocp
        controller = S550_3DOF_ocp(
            DynParam=params['nominal_dynamic_params'],
            DroneParam=params['nominal_drone_params'],
            MpcParam=params['nmpc_params'])

    elif control_type == 'pd':
        from control.PID.pitch_control import PitchControl
        use_dob = (dob_type != 'none')
        controller = PitchControl(
            DynamicParams=params['nominal_dynamic_params'],
            GainParams=params['gain_params'],
            DobMode=use_dob)

    # Setup DOB
    if dob_type == 'hgdo':
        from estimator.dob.hgdo.hgdo_3d import HGDO3D
        dob = HGDO3D(DynamicParams=params['nominal_dynamic_params'],
                     DroneParams=params['nominal_drone_params'],
                     RotorParams=params['nominal_rotor_params'],
                     DobParams=params['dob_params'])

    return controller, dob


def estimate_acceleration(state, w_rotor, C_T, m):
    """
    Model-based acceleration estimation from current thrust and pitch.
    a_world = R @ (f_body / m) + g_vec

    :param state: [px, pz, vx, vz, th, q]
    :param w_rotor: rotor speeds [w1, w2, w3] in rad/s
    :param C_T: thrust coefficient
    :param m: mass
    :return: [ax, az] in world frame
    """
    from utils.math_tool import pitch_to_rotm

    theta = state[4]
    R = pitch_to_rotm(theta)

    # Total thrust from rotor speeds (hexarotor: 2 motors per group)
    f_total = 2.0 * C_T * np.sum(w_rotor**2)
    f_body = np.array([0.0, f_total])

    # World frame acceleration
    g_vec = np.array([0.0, -9.81])
    a_world = R @ (f_body / m) + g_vec

    return a_world  # [ax, az]


def plot_results_3d(t, drone_data, rotor_data, ref_data, dob_data, nmpc_solve_times=None):
    """Plot results for 3DOF simulation"""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    # Position X
    axes[0, 0].plot(t, drone_data['pos'][:, 0], 'b-', label='x')
    axes[0, 0].plot(t, ref_data['pos_des'][:, 0], 'r--', label='x_des')
    axes[0, 0].set_ylabel('X [m]')
    axes[0, 0].set_title('Position X')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Position Z
    axes[0, 1].plot(t, drone_data['pos'][:, 1], 'b-', label='z')
    axes[0, 1].plot(t, ref_data['pos_des'][:, 1], 'r--', label='z_des')
    axes[0, 1].set_ylabel('Z [m]')
    axes[0, 1].set_title('Position Z')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Velocity X
    axes[1, 0].plot(t, drone_data['vel'][:, 0], 'b-', label='vx')
    axes[1, 0].plot(t, ref_data['vel_des'][:, 0], 'r--', label='vx_des')
    axes[1, 0].set_ylabel('Vx [m/s]')
    axes[1, 0].set_title('Velocity X')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Velocity Z
    axes[1, 1].plot(t, drone_data['vel'][:, 1], 'b-', label='vz')
    axes[1, 1].plot(t, ref_data['vel_des'][:, 1], 'r--', label='vz_des')
    axes[1, 1].set_ylabel('Vz [m/s]')
    axes[1, 1].set_title('Velocity Z')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Pitch
    axes[2, 0].plot(t, np.rad2deg(drone_data['pitch']), 'b-', label='pitch')
    axes[2, 0].set_ylabel('Pitch [deg]')
    axes[2, 0].set_title('Pitch Angle')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Pitch rate
    axes[2, 1].plot(t, np.rad2deg(drone_data['pitch_rate']), 'b-', label='pitch_rate')
    axes[2, 1].set_ylabel('Pitch Rate [deg/s]')
    axes[2, 1].set_title('Pitch Rate')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    # Rotor speeds
    for i in range(rotor_data['w_rotor'].shape[1]):
        axes[3, 0].plot(t, rotor_data['w_rotor'][:, i], label=f'w{i+1}')
    axes[3, 0].set_ylabel('Rotor Speed [RPM]')
    axes[3, 0].set_xlabel('Time [s]')
    axes[3, 0].set_title('Rotor Speeds')
    axes[3, 0].legend()
    axes[3, 0].grid(True)

    # DOB estimates
    axes[3, 1].plot(t, dob_data['f_est'][:, 0], 'b-', label='f_ext_x')
    axes[3, 1].plot(t, dob_data['f_est'][:, 1], 'g-', label='f_ext_z')
    axes[3, 1].plot(t, dob_data['tau_est'], 'r-', label='tau_ext')
    axes[3, 1].set_ylabel('Disturbance')
    axes[3, 1].set_xlabel('Time [s]')
    axes[3, 1].set_title('DOB Estimates')
    axes[3, 1].legend()
    axes[3, 1].grid(True)

    plt.tight_layout()
    plt.savefig('results_3d.png', dpi=300)
    plt.show()

    # Disturbance moment comparison: estimated vs actual (separate figure)
    if 'tau_actual' in dob_data:
        fig_dist, axes_dist = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Top: disturbance moment comparison
        axes_dist[0].plot(t, dob_data['tau_actual'], 'b-', linewidth=1.5, label=r'$\tau_{actual}$ (actual disturbance)')
        axes_dist[0].plot(t, dob_data['tau_est'], 'r--', linewidth=1.5, label=r'$\tau_{est}$ (DOB estimate)')
        axes_dist[0].set_ylabel('Moment [Nm]')
        axes_dist[0].set_title('Pitch Disturbance Moment: Actual vs DOB Estimate')
        axes_dist[0].legend()
        axes_dist[0].grid(True, alpha=0.3)

        # Bottom: compensated control moment (u_comp My)
        if 'My_comp' in dob_data:
            axes_dist[1].plot(t, dob_data['My_comp'], 'g-', linewidth=1.5, label=r'$M_{y,comp}$ (commanded)')
            axes_dist[1].set_ylabel('Moment [Nm]')
            axes_dist[1].set_xlabel('Time [s]')
            axes_dist[1].set_title('Compensated Control Moment ($u_{comp}$ My)')
            axes_dist[1].legend()
            axes_dist[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('disturbance_moment_comparison.png', dpi=300)
        plt.show()

    # NMPC solve time histogram
    if nmpc_solve_times is not None and len(nmpc_solve_times) > 0:
        solve_times_ms = np.array(nmpc_solve_times) * 1000

        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))

        # Histogram with 0.5ms bins
        max_time = max(15.0, np.max(solve_times_ms) + 1)
        bins = np.arange(0, max_time, 0.5)
        ax_hist.hist(solve_times_ms, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        # Add vertical lines for statistics
        mean_time = np.mean(solve_times_ms)
        max_solve = np.max(solve_times_ms)
        ax_hist.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f} ms')
        ax_hist.axvline(max_solve, color='orange', linestyle='--', linewidth=2, label=f'Max: {max_solve:.2f} ms')
        ax_hist.axvline(10.0, color='green', linestyle=':', linewidth=2, label='Control loop: 10 ms')

        # Statistics text box
        stats_text = f'N = {len(solve_times_ms)}\nMean: {mean_time:.2f} ms\nMax: {max_solve:.2f} ms\nStd: {np.std(solve_times_ms):.2f} ms'
        ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_hist.set_xlabel('Solve Time [ms]')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('NMPC Solve Time Distribution')
        ax_hist.legend(loc='upper left')
        ax_hist.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('nmpc_solve_time_hist.png', dpi=300)
        plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='3DOF Control simulation with contact dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python3 main_control_3d.py --control pd --dob hgdo --mode regulation
        python3 main_control_3d.py --control nmpc --dob none --mode regulation
        python3 main_control_3d.py --control nmpc --dob hgdo --mode tracking
        """
    )

    parser.add_argument('--control', type=str, default='pd',
                        choices=['pd', 'nmpc'],
                        help='Control method: pd (PitchControl) or nmpc')

    parser.add_argument('--dob', type=str, default='hgdo',
                        choices=['none', 'hgdo'],
                        help='Disturbance observer: none or hgdo')

    parser.add_argument('--mode', type=str, default='regulation',
                        choices=['regulation', 'tracking'],
                        help='Control mode: regulation (fixed hover) or tracking (hehn trajectory)')

    args = parser.parse_args()
    control_type = args.control
    dob_type = args.dob
    control_mode = args.mode

    print(f"\n{'='*60}")
    print(f"3DOF Simulation (x, z, theta)")
    print(f"Control: {control_type}, DOB: {dob_type}, Mode: {control_mode}")
    print(f"{'='*60}")

    # Load parameters
    params = load_parameters_3d(control_type, dob_type)

    print("Parameter Configuration:")
    print("- Simulator: Uses TRUE parameters from config/simulator/simulator_3d.yaml")
    print(f"- Controller & DOB: Use NOMINAL parameters from config/control/{control_type}/")
    print(f"- CoM offset: {params['true_dynamic_params']['com_offset']}")

    # Create simulation models (TRUE parameters)
    drone_sim_model = S550_3D_Sim_Model(DynamicParam=params['true_dynamic_params'])
    rotor_sim_model = RotorModel(RotorParams=params['true_rotor_params'])
    hexa_converter = HexaConverter(DroneParams=params['true_drone_params'],
                                   RotorParams=params['true_rotor_params'],
                                   Dim=3)

    # Setup controller and DOB (NOMINAL parameters)
    controller, dob = setup_controller_3d(control_type, dob_type, params)

    # State initialization
    w_rotor_idle = params['sim_params']['w_rotor_idle']
    com_offset = params['true_dynamic_params']['com_offset']
    x_init = params['sim_params']['initial_pos'][0]
    z_init = params['sim_params']['initial_pos'][1]
    initial_pos = np.array([x_init + com_offset[0], z_init + com_offset[1]])
    s_drone, s_rotor = state_initialize(w_rotor_idle, initial_pos, Dim=3)

    # Simulation parameters
    tf = params['sim_params']['tf']
    dt = params['sim_params']['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # For hard clamp
    alpha_max = params['true_rotor_params']['alpha_rotor_max']
    num_rotors = params['true_rotor_params']['num_rotors']

    # Setup tracking mode for NMPC
    C_T = params['nominal_drone_params']['motor_const']
    m_nom = params['nominal_dynamic_params']['m']

    if control_mode == 'regulation':
        p_target = params['regulation_params']['setpoint_position']
        print(f"\nRegulation: target position x={p_target[0]:.2f}, z={p_target[1]:.2f} m")
    else:
        target_2d = params['tracking_params']['target_position']
        if control_type == 'nmpc':
            controller.setup_tracking(params['tracking_params'], target_2d)
        replan_freq = params['tracking_params'].get('replan_freq', 100.0)
        print(f"\nTracking: Hehn trajectory, replan at {replan_freq:.0f}Hz")
        print(f"  Target: {target_2d}")

    print(f"Simulation time: {tf:.1f} s, dt: {dt*1000:.1f} ms\n")

    # Data storage
    pos_hist = []
    vel_hist = []
    pitch_hist = []
    pitch_rate_hist = []
    pos_des_hist = []
    vel_des_hist = []
    w_rotor_hist = []
    alpha_rotor_hist = []
    f_est_hist = []
    tau_est_hist = []
    tau_actual_hist = []
    My_comp_hist = []
    nmpc_solve_times = []

    # Nominal Jyy for actual disturbance computation
    Jyy_nominal = params['nominal_dynamic_params']['MoiArray'][1]

    # Main simulation loop
    for i in range(N - 1):
        # Get body frame state from simulation
        s_body = drone_sim_model.get_state(s_drone)

        # Unpack state
        p = s_body[0:2]
        v_body = s_body[2:4]
        theta = s_body[4]
        q = s_body[5]

        # Convert to world frame velocity
        from utils.math_tool import pitch_to_rotm
        v_world = pitch_to_rotm(theta) @ v_body

        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)

        # Compute reference
        t_now = t_sim[i]
        if control_mode == 'regulation':
            p_des = params['regulation_params']['setpoint_position']
            v_des = np.array([0.0, 0.0])
            ref = np.concatenate([p_des, v_des])

        # DOB estimate
        if dob is not None and i > 1:
            d_est = dob.dob_estimate(t_sim[i-1], t_sim[i], w_rotor, s_body)
        else:
            d_est = np.zeros(3)

        f_est = d_est[0:2]
        tau_est = d_est[2]

        # Compute control
        My_comp = 0.0
        if control_type == 'nmpc':
            t_solve_start = time.perf_counter()
            if control_mode == 'tracking':
                # Tracking: NMPC generates trajectory and returns p_des, v_des
                state_2d = np.array([p[0], p[1], v_world[0], v_world[1]])
                status, w_cmd, p_des, v_des = controller.solve_for_trajectory(
                    s_body, state_2d, t_now)
            else:
                status, w_cmd = controller.solve(s_body, ref)
            t_solve_end = time.perf_counter()
            nmpc_solve_times.append(t_solve_end - t_solve_start)

            if dob is not None and t_now > 1.0:
                # DOB compensation: subtract disturbance from NMPC output
                # Wait until t > 1.0s for DOB to converge after liftoff
                u_mpc = hexa_converter.compute_u(w_cmd)
                u_comp = u_mpc.copy()
                u_comp[0] -= d_est[1]              # subtract f_ext_z from Fz
                u_comp[1] -= d_est[2]              # subtract tau_ext from My
                My_comp = u_comp[1]
                w_cmd = hexa_converter.compute_des_rotor_speed(u_comp)
            else:
                My_comp = hexa_converter.compute_u(w_cmd)[1]

            if status != 0 and i % 100 == 0:
                print(f"Warning: NMPC solver status {status} at t={t_now:.2f}s")

        elif control_type == 'pd':
            if control_mode == 'tracking':
                # PD tracking: just use target position
                p_des = params['tracking_params']['target_position']
                v_des = np.array([0.0, 0.0])
                ref = np.concatenate([p_des, v_des])
            u = controller.compute_u(s_body, ref, d_est, dt)
            My_comp = u[1]
            w_cmd = hexa_converter.compute_des_rotor_speed(u)

        # Store history
        pos_hist.append(p.copy())
        vel_hist.append(v_world.copy())
        pitch_hist.append(theta)
        pitch_rate_hist.append(q)
        pos_des_hist.append(p_des.copy())
        vel_des_hist.append(v_des.copy())
        w_rotor_hist.append(w_rotor.copy())
        alpha_rotor_hist.append(alpha_rotor.copy())
        f_est_hist.append(f_est.copy())
        tau_est_hist.append(tau_est)
        My_comp_hist.append(My_comp)

        # Simulation step
        t_ode = [t_sim[i], t_sim[i + 1]]

        # Simulate rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)

        # Hard clamp for rotor acceleration
        s_rotor[num_rotors:] = np.clip(s_rotor[num_rotors:], -alpha_max, alpha_max)

        # Compute actual control input from rotor speeds
        u_actual = hexa_converter.compute_u(s_rotor[:num_rotors])

        # Compute actual disturbance moment (from DOB's perspective)
        # DOB model: Jyy_nom * dqdt = My + tau_ext
        # Actual: dqdt comes from simulator (flight/contact dynamics)
        # tau_actual = Jyy_nom * dqdt_actual - My_actual
        sdot_actual = drone_sim_model.dynamics(t_now, s_drone, u_actual)
        dqdt_actual = sdot_actual[5]
        My_actual = u_actual[1]
        tau_actual = Jyy_nominal * dqdt_actual - My_actual
        tau_actual_hist.append(tau_actual)

        # Simulate drone dynamics
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u_actual, t_ode)

        # Prevent ground penetration
        s_drone = drone_sim_model.clamp_ground(s_drone)

        # Print progress
        if i % 1000 == 0:
            print(f"t={t_sim[i]:.2f}s, z={p[1]:.3f}m, pitch={np.rad2deg(theta):.2f}deg, "
                  f"w_rotor=[{w_rotor[0]:.0f}, {w_rotor[1]:.0f}, {w_rotor[2]:.0f}]")

    # Post-processing
    drone_data = {
        'pos': np.array(pos_hist),
        'vel': np.array(vel_hist),
        'pitch': np.array(pitch_hist),
        'pitch_rate': np.array(pitch_rate_hist)
    }

    rotor_data = {
        'w_rotor': np.array(w_rotor_hist),
        'alpha_rotor': np.array(alpha_rotor_hist)
    }

    ref_data = {
        'pos_des': np.array(pos_des_hist),
        'vel_des': np.array(vel_des_hist)
    }

    dob_data = {
        'f_est': np.array(f_est_hist),
        'tau_est': np.array(tau_est_hist),
        'tau_actual': np.array(tau_actual_hist),
        'My_comp': np.array(My_comp_hist)
    }

    # Print final statistics
    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"Final position: x={drone_data['pos'][-1, 0]:.3f}, z={drone_data['pos'][-1, 1]:.3f} m")
    print(f"Final pitch: {np.rad2deg(drone_data['pitch'][-1]):.2f} deg")

    # NMPC solve time statistics
    if control_type == 'nmpc' and len(nmpc_solve_times) > 0:
        solve_times_ms = np.array(nmpc_solve_times) * 1000
        print(f"\nNMPC Solve Time Statistics (All):")
        print(f"  Mean:   {np.mean(solve_times_ms):.3f} ms")
        print(f"  Std:    {np.std(solve_times_ms):.3f} ms")
        print(f"  Max:    {np.max(solve_times_ms):.3f} ms")
        print(f"  Min:    {np.min(solve_times_ms):.3f} ms")

        # Statistics excluding first 10 samples (initial trajectory)
        if len(solve_times_ms) > 10:
            solve_times_after = solve_times_ms[10:]
            print(f"\nNMPC Solve Time Statistics (Excluding first 10):")
            print(f"  Mean:   {np.mean(solve_times_after):.3f} ms")
            print(f"  Max:    {np.max(solve_times_after):.3f} ms")
            print(f"  Min:    {np.min(solve_times_after):.3f} ms")

        print(f"\n  Sim dt: {dt*1000:.1f} ms")
        realtime_ratio = (dt * 1000) / np.mean(solve_times_ms)
        print(f"  Real-time ratio: {realtime_ratio:.1f}x (>1 means real-time capable)")

    print(f"{'='*60}")

    # Plot results
    plot_results_3d(t_sim[:-1], drone_data, rotor_data, ref_data, dob_data,
                    nmpc_solve_times=nmpc_solve_times if control_type == 'nmpc' else None)

    # Cleanup for NMPC
    if control_type == 'nmpc':
        from utils.acados_cleanup import cleanup_acados_files
        cleanup_acados_files(controller.get_json_file_name())


if __name__ == '__main__':
    main()
