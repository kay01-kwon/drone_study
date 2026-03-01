#!/usr/bin/env python3
"""
3DOF (x, z, theta) Drone Simulation with Angular Velocity Feedback - Version 2

Key differences from main_control_3d.py:
1. Uses S550_3DOF_ocp_v2 with angular velocity and roll/pitch feedback
2. Position/velocity trajectory uses Hehn trajectory with z-jerk based on rotor dynamics:
   - j_max_z = 12 * C_T * w_hover * alpha (6 rotors × 2 from chain rule, alpha = 10,000 RPM/s)
3. Angular trajectory with altitude-based feedback:
   - z <= 0.01m (grounded): feedback angular velocity and pitch trajectory
   - z > 0.01m (airborne): feedback pitch=0, angular velocity=0
4. DOB moment differentiation with LPF for dynamic angular jerk limit:
   - M_dot_y from DOB moment differentiation
   - j_max_ang = M_dot_y / J_p
   - J_p = m * (xg^2 + h_eff^2), where h_eff = 0.360m

Examples:
    python3 main_control_3d_v2.py --control nmpc --dob hgdo --mode tracking
    python3 main_control_3d_v2.py --control nmpc --dob hgdo --mode tracking --landing

Author: Geonwoo Kwon
Date: 2026-02-27
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


def load_parameters_3d_v2(control_type, dob_type):
    """Load parameters for 3DOF v2 simulation"""
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
            'replan_freq': tracking.get('replan_freq', 100.0),
        }

    elif control_type == 'pd':
        config_control = yaml_loader.load_yaml('config/control/pd/pd_3d_params.yaml')
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)
        params['gain_params'] = yaml_loader.get_pd_gain_params(config_control)

        regulation = config_control.get('regulation', {})
        params['regulation_params'] = {
            'setpoint_position': np.array(regulation.get('setpoint_position', [0.0, 1.0]))
        }

        tracking = config_control.get('tracking', {})
        params['tracking_params'] = {
            'target_position': np.array(tracking.get('target_position', [0.0, 1.0])),
            'a_max': tracking.get('a_max', 12.0),
            'a_min': tracking.get('a_min', 3.0),
            'omega_xy_max': tracking.get('omega_xy_max', 3.0),
            'time_scale': tracking.get('time_scale', 5.0),
        }

    # Load DOB parameters
    if dob_type == 'hgdo':
        config_dob = yaml_loader.load_yaml('config/estimator/dob/hgdo_3d.yaml')
        params['dob_params'] = yaml_loader.get_hgdo_params(config_dob)

    return params


def setup_controller_3d_v2(control_type, dob_type, params):
    """Setup controller and DOB for 3DOF v2 (uses S550_3DOF_ocp_v2)"""
    dob = None

    if control_type == 'nmpc':
        from control.nmpc.ocp.S550_3DOF_ocp_v2 import S550_3DOF_ocp_v2
        controller = S550_3DOF_ocp_v2(
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


def detect_grounded(p, z_threshold=0.01):
    """
    Detect if the drone is grounded (altitude <= 0.01m).

    Args:
        p: Current position [x, z]
        z_threshold: Height threshold for grounded detection [m]

    Returns:
        bool: True if grounded (z <= 0.01m)
    """
    return p[1] <= z_threshold


def plot_results_3d_v2(t, drone_data, rotor_data, ref_data, dob_data,
                       ang_ref_data=None, jerk_data=None, nmpc_solve_times=None):
    """Plot results for 3DOF v2 simulation with angular reference"""
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

    # Pitch with reference
    axes[2, 0].plot(t, np.rad2deg(drone_data['pitch']), 'b-', label='pitch')
    if ang_ref_data is not None and 'th_des' in ang_ref_data:
        axes[2, 0].plot(t, np.rad2deg(ang_ref_data['th_des']), 'r--', label='pitch_des')
    axes[2, 0].set_ylabel('Pitch [deg]')
    axes[2, 0].set_title('Pitch Angle')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Pitch rate with reference
    axes[2, 1].plot(t, np.rad2deg(drone_data['pitch_rate']), 'b-', label='pitch_rate')
    if ang_ref_data is not None and 'q_des' in ang_ref_data:
        axes[2, 1].plot(t, np.rad2deg(ang_ref_data['q_des']), 'r--', label='pitch_rate_des')
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
    plt.savefig('results_3d_v2.png', dpi=300)
    plt.show()

    # Disturbance moment comparison
    if 'tau_actual' in dob_data:
        fig_dist, axes_dist = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes_dist[0].plot(t, dob_data['tau_actual'], 'b-', linewidth=1.5,
                          label=r'$\tau_{actual}$')
        axes_dist[0].plot(t, dob_data['tau_est'], 'r--', linewidth=1.5,
                          label=r'$\tau_{est}$')
        axes_dist[0].set_ylabel('Moment [Nm]')
        axes_dist[0].set_title('Pitch Disturbance Moment: Actual vs DOB Estimate')
        axes_dist[0].legend()
        axes_dist[0].grid(True, alpha=0.3)

        if 'tau_effective' in dob_data:
            axes_dist[1].plot(t, dob_data['tau_est'], 'r--', linewidth=1.0,
                              alpha=0.5, label=r'$\tau_{est}$ (raw)')
            axes_dist[1].plot(t, dob_data['tau_effective'], 'g-', linewidth=1.5,
                              label=r'$\tau_{effective}$ (pitch-latched)')
            axes_dist[1].set_ylabel('Moment [Nm]')
            axes_dist[1].set_title('Effective DOB Moment (Pitch Tolerance Latching)')
            axes_dist[1].legend()
            axes_dist[1].grid(True, alpha=0.3)

        if 'My_comp' in dob_data:
            axes_dist[2].plot(t, dob_data['My_comp'], 'b-', linewidth=1.5,
                              label=r'$M_{y,comp}$')
            axes_dist[2].set_ylabel('Moment [Nm]')
            axes_dist[2].set_xlabel('Time [s]')
            axes_dist[2].set_title('Compensated Control Moment')
            axes_dist[2].legend()
            axes_dist[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('disturbance_moment_comparison_v2.png', dpi=300)
        plt.show()

    # Dynamic jerk limit plot
    if ang_ref_data is not None and 'j_max_ang' in ang_ref_data:
        fig_jerk, ax_jerk = plt.subplots(figsize=(10, 4))
        ax_jerk.plot(t, ang_ref_data['j_max_ang'], 'b-', linewidth=1.5)
        ax_jerk.set_ylabel('Angular Jerk Limit [rad/s³]')
        ax_jerk.set_xlabel('Time [s]')
        ax_jerk.set_title('Dynamic Angular Jerk Limit from DOB')
        ax_jerk.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('angular_jerk_limit_v2.png', dpi=300)
        plt.show()

    # Rotor acceleration plot
    if 'alpha_rotor' in rotor_data:
        fig_alpha, axes_alpha = plt.subplots(rotor_data['alpha_rotor'].shape[1], 1,
                                              figsize=(10, 8), sharex=True)
        if rotor_data['alpha_rotor'].shape[1] == 1:
            axes_alpha = [axes_alpha]

        for i in range(rotor_data['alpha_rotor'].shape[1]):
            axes_alpha[i].plot(t, rotor_data['alpha_rotor'][:, i], 'r-', linewidth=1.5,
                               label=rf'$\alpha_{{{i+1}}}$')
            axes_alpha[i].set_ylabel(f'$\\alpha_{{{i+1}}}$ [RPM/s]')
            axes_alpha[i].legend(loc='upper right')
            axes_alpha[i].grid(True, alpha=0.3)

        axes_alpha[-1].set_xlabel('Time [s]')
        fig_alpha.suptitle('Rotor Angular Acceleration', fontsize=14)
        plt.tight_layout()
        plt.savefig('rotor_acceleration_v2.png', dpi=300)
        plt.show()

    # Jerk comparison plot: drone jerk vs Hehn trajectory jerk
    if jerk_data is not None:
        fig_jerk, axes_jerk = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Jerk X
        axes_jerk[0].plot(t, jerk_data['jerk_drone'][:, 0], 'b-', linewidth=1.0,
                          label='Drone $j_x$')
        axes_jerk[0].plot(t, jerk_data['jerk_henh'][:, 0], 'r--', linewidth=1.5,
                          label='Hehn $j_x$')
        axes_jerk[0].set_ylabel(r'Jerk X [m/s$^3$]')
        axes_jerk[0].set_title('Jerk X: Drone vs Hehn Trajectory')
        axes_jerk[0].legend()
        axes_jerk[0].grid(True, alpha=0.3)

        # Jerk Z
        axes_jerk[1].plot(t, jerk_data['jerk_drone'][:, 1], 'b-', linewidth=1.0,
                          label='Drone $j_z$')
        axes_jerk[1].plot(t, jerk_data['jerk_henh'][:, 1], 'r--', linewidth=1.5,
                          label='Hehn $j_z$')
        axes_jerk[1].set_ylabel(r'Jerk Z [m/s$^3$]')
        axes_jerk[1].set_xlabel('Time [s]')
        axes_jerk[1].set_title('Jerk Z: Drone vs Hehn Trajectory')
        axes_jerk[1].legend()
        axes_jerk[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('jerk_comparison_3d_v2.png', dpi=300)
        plt.show()

    # NMPC solve time histogram
    if nmpc_solve_times is not None and len(nmpc_solve_times) > 0:
        solve_times_ms = np.array(nmpc_solve_times) * 1000

        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))

        max_time = max(15.0, np.max(solve_times_ms) + 1)
        bins = np.arange(0, max_time, 0.5)
        ax_hist.hist(solve_times_ms, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        mean_time = np.mean(solve_times_ms)
        max_solve = np.max(solve_times_ms)
        ax_hist.axvline(mean_time, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_time:.2f} ms')
        ax_hist.axvline(max_solve, color='orange', linestyle='--', linewidth=2,
                        label=f'Max: {max_solve:.2f} ms')
        ax_hist.axvline(10.0, color='green', linestyle=':', linewidth=2,
                        label='Control loop: 10 ms')

        stats_text = (f'N = {len(solve_times_ms)}\n'
                      f'Mean: {mean_time:.2f} ms\n'
                      f'Max: {max_solve:.2f} ms\n'
                      f'Std: {np.std(solve_times_ms):.2f} ms')
        ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_hist.set_xlabel('Solve Time [ms]')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('NMPC Solve Time Distribution (v2)')
        ax_hist.legend(loc='upper left')
        ax_hist.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('nmpc_solve_time_hist_v2.png', dpi=300)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='3DOF Control v2 simulation with angular velocity feedback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python3 main_control_3d_v2.py --control nmpc --dob hgdo --mode tracking
        python3 main_control_3d_v2.py --control nmpc --dob hgdo --mode tracking --landing
        """
    )

    parser.add_argument('--control', type=str, default='nmpc',
                        choices=['pd', 'nmpc'],
                        help='Control method: pd or nmpc')

    parser.add_argument('--dob', type=str, default='hgdo',
                        choices=['none', 'hgdo'],
                        help='Disturbance observer: none or hgdo')

    parser.add_argument('--mode', type=str, default='tracking',
                        choices=['regulation', 'tracking'],
                        help='Control mode: regulation or tracking')

    parser.add_argument('--landing', action='store_true',
                        help='Enable landing mode with angular velocity feedback')

    args = parser.parse_args()
    control_type = args.control
    dob_type = args.dob
    control_mode = args.mode
    enable_landing = args.landing

    print(f"\n{'='*60}")
    print(f"3DOF Simulation V2 (x, z, theta) with Angular Feedback")
    print(f"Control: {control_type}, DOB: {dob_type}, Mode: {control_mode}")
    print(f"Landing mode: {'Enabled' if enable_landing else 'Disabled'}")
    print(f"{'='*60}")

    # Load parameters
    params = load_parameters_3d_v2(control_type, dob_type)

    print("\nParameter Configuration:")
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
    controller, dob = setup_controller_3d_v2(control_type, dob_type, params)

    # Print jerk limits for NMPC v2
    if control_type == 'nmpc':
        jerk_info = controller.get_jerk_limits()
        print(f"\nJerk Limit Configuration (v2):")
        print(f"  j_max_z: {jerk_info['j_max_z']:.2f} m/s³")
        print(f"  h_eff: {jerk_info['h_eff']:.3f} m")
        print(f"  xg: {jerk_info['xg']:.3f} m")
        print(f"  J_p = m*(xg²+h_eff²): {jerk_info['J_p']:.4f} kg·m²")
        print(f"  f_max: {jerk_info['f_max']:.2f} N")
        print(f"  w_hover: {jerk_info['w_hover']:.2f} RPM")
        print(f"  alpha_rotor: {jerk_info['alpha_rotor']:.2f} RPM/s")
        print(f"  Grounded threshold: 0.01 m")

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

    # Setup tracking mode
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
        print(f"\nTracking: Hehn trajectory with dynamic jerk, replan at {replan_freq:.0f}Hz")
        print(f"  Target: {target_2d}")

    print(f"Simulation time: {tf:.1f} s, dt: {dt*1000:.1f} ms\n")

    # Data storage
    pos_hist = []
    vel_hist = []
    pitch_hist = []
    pitch_rate_hist = []
    pos_des_hist = []
    vel_des_hist = []
    th_des_hist = []
    q_des_hist = []
    j_max_ang_hist = []
    w_rotor_hist = []
    alpha_rotor_hist = []
    f_est_hist = []
    tau_est_hist = []
    tau_actual_hist = []
    My_comp_hist = []
    nmpc_solve_times = []
    acc_hist = []
    jerk_henh_x_hist = []
    jerk_henh_z_hist = []
    tau_effective_hist = []
    grounded_moment_active_hist = []

    # True Jyy for actual physical disturbance computation
    Jyy_true = params['true_dynamic_params']['MoiArray'][1]

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
            th_des = 0.0
            q_des = 0.0
            ref = np.concatenate([p_des, v_des])

        # DOB estimate
        if dob is not None and i > 1:
            d_est = dob.dob_estimate(t_sim[i-1], t_sim[i], w_rotor, s_body)
        else:
            d_est = np.zeros(3)

        f_est = d_est[0:2]
        tau_est = d_est[2]

        # Detect grounded state (z <= 0.01m)
        # Note: Controller internally checks altitude for angular trajectory feedback
        if enable_landing and control_type == 'nmpc':
            is_grounded = detect_grounded(p)
            controller.set_landing_mode(is_grounded, t_now)

        # Compute control
        My_comp = 0.0
        j_max_ang = 0.0

        if control_type == 'nmpc':
            t_solve_start = time.perf_counter()

            if control_mode == 'tracking':
                state_2d = np.array([p[0], p[1], v_world[0], v_world[1]])
                # Pass DOB moment estimate for dynamic jerk computation
                status, w_cmd, p_des, v_des, th_des, q_des = controller.solve_for_trajectory(
                    s_body, state_2d, t_now, tau_est=tau_est, dt=dt)

                # Get current angular jerk limit for logging
                j_max_ang = controller.compute_angular_jerk_max(theta, q, 0.0)
            else:
                status, w_cmd = controller.solve(s_body, ref)
                th_des = 0.0
                q_des = 0.0

            t_solve_end = time.perf_counter()
            nmpc_solve_times.append(t_solve_end - t_solve_start)

            # DOB compensation with pitch tolerance-based moment latching
            # - Grounded & |pitch| > 5 deg: accumulate max DOB moment
            # - Grounded & |pitch| < 5 deg: use stored max moment for compensation
            # - Airborne: use liftoff-latched moment
            # Note: z-force compensation is always applied (d_est[1])
            if dob is not None:
                u_mpc = hexa_converter.compute_u(w_cmd)
                u_comp = u_mpc.copy()

                # Get effective DOB moment (pitch-tolerance based for grounded)
                tau_effective = controller.get_tau_effective()

                u_comp[0] -= d_est[1]  # always subtract f_ext_z from Fz
                u_comp[1] -= tau_effective  # subtract pitch-latched tau_ext from My
                My_comp = u_comp[1]
                w_cmd = hexa_converter.compute_des_rotor_speed(u_comp)

                # Store tau_effective and grounded moment state for logging
                tau_effective_hist.append(tau_effective)
                grounded_moment_active_hist.append(controller.is_grounded_moment_active)
            else:
                My_comp = hexa_converter.compute_u(w_cmd)[1]
                tau_effective_hist.append(0.0)
                grounded_moment_active_hist.append(False)

            if status != 0 and i % 100 == 0:
                print(f"Warning: NMPC solver status {status} at t={t_now:.2f}s")

        elif control_type == 'pd':
            if control_mode == 'tracking':
                p_des = params['tracking_params']['target_position']
                v_des = np.array([0.0, 0.0])
                ref = np.concatenate([p_des, v_des])
            th_des = 0.0
            q_des = 0.0
            u = controller.compute_u(s_body, ref, d_est, dt)
            My_comp = u[1]
            w_cmd = hexa_converter.compute_des_rotor_speed(u)

        # Compute acceleration for jerk calculation
        acc = estimate_acceleration(s_body, w_rotor, C_T, m_nom)
        acc_hist.append(acc.copy())

        # Hehn trajectory jerk (tracking mode with NMPC only)
        if (control_mode == 'tracking' and control_type == 'nmpc'
                and controller.traj is not None):
            t_rel = t_now - controller.t_start
            jerk_henh_3d = controller.traj.get_jerk(t_rel)
            jerk_henh_x_hist.append(jerk_henh_3d[0])
            jerk_henh_z_hist.append(jerk_henh_3d[2])
        else:
            jerk_henh_x_hist.append(0.0)
            jerk_henh_z_hist.append(0.0)

        # Store history
        pos_hist.append(p.copy())
        vel_hist.append(v_world.copy())
        pitch_hist.append(theta)
        pitch_rate_hist.append(q)
        pos_des_hist.append(p_des.copy())
        vel_des_hist.append(v_des.copy())
        th_des_hist.append(th_des)
        q_des_hist.append(q_des)
        j_max_ang_hist.append(j_max_ang)
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

        # Compute actual physical disturbance moment
        sdot_actual = drone_sim_model.dynamics(t_now, s_drone, u_actual)
        dqdt_actual = sdot_actual[5]
        My_actual = u_actual[1]
        tau_actual = Jyy_true * dqdt_actual - My_actual
        tau_actual_hist.append(tau_actual)

        # Simulate drone dynamics
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u_actual, t_ode)

        # Prevent ground penetration
        s_drone = drone_sim_model.clamp_ground(s_drone)

        # Print progress
        if i % 1000 == 0:
            status_strs = []
            if enable_landing and control_type == 'nmpc':
                if controller.is_grounded:
                    status_strs.append("GROUNDED")
                if controller.is_grounded_moment_active:
                    status_strs.append(f"MOM_COMP={controller.tau_grounded_latched:.3f}")
            status_str = f" [{', '.join(status_strs)}]" if status_strs else ""
            print(f"t={t_sim[i]:.2f}s, z={p[1]:.3f}m, pitch={np.rad2deg(theta):.2f}deg, "
                  f"w_rotor=[{w_rotor[0]:.0f}, {w_rotor[1]:.0f}, {w_rotor[2]:.0f}]{status_str}")

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

    ang_ref_data = {
        'th_des': np.array(th_des_hist),
        'q_des': np.array(q_des_hist),
        'j_max_ang': np.array(j_max_ang_hist)
    }

    dob_data = {
        'f_est': np.array(f_est_hist),
        'tau_est': np.array(tau_est_hist),
        'tau_actual': np.array(tau_actual_hist),
        'My_comp': np.array(My_comp_hist),
        'tau_effective': np.array(tau_effective_hist),
        'grounded_moment_active': np.array(grounded_moment_active_hist)
    }

    # Compute drone jerk from acceleration (numerical differentiation)
    acc_array = np.array(acc_hist)  # [N-1, 2]
    jerk_drone = np.diff(acc_array, axis=0) / dt  # [N-2, 2]
    jerk_drone = np.vstack([np.zeros((1, 2)), jerk_drone])  # pad first sample

    jerk_data = {
        'jerk_drone': jerk_drone,
        'jerk_henh': np.column_stack([jerk_henh_x_hist, jerk_henh_z_hist])
    }

    # Print final statistics
    print(f"\n{'='*60}")
    print("Simulation Complete (V2)")
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
    plot_results_3d_v2(t_sim[:-1], drone_data, rotor_data, ref_data, dob_data,
                       ang_ref_data=ang_ref_data,
                       jerk_data=jerk_data,
                       nmpc_solve_times=nmpc_solve_times if control_type == 'nmpc' else None)

    # Cleanup for NMPC
    if control_type == 'nmpc':
        from utils.acados_cleanup import cleanup_acados_files
        cleanup_acados_files(controller.get_json_file_name())


if __name__ == '__main__':
    main()
