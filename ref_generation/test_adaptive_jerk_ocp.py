#!/usr/bin/env python3
"""
Test script for Adaptive Jerk OCP (3DOF)

Tests hover at (0, 1.0) with ground contact model.
Based on main_control_3d.py structure.

Author: Claude
Date: 2026-02-26
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ref_generation.adaptive_jerk_ocp import AdaptiveJerkOCP
from sim_model.S550_3d_model import S550_3D_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.custom_ode import custom_rk4
from utils.state_initializer import state_initialize


def main():
    print("\n" + "="*60)
    print("Adaptive Jerk OCP Test - 3DOF Hover")
    print("="*60)

    # Parameters (matching simulator_3d.yaml)
    DynParam = {
        'm': 3.0,
        'MoiArray': [0.065, 0.065, 0.10],
        'com_offset': [-0.0105, 0.0, 0.0]
    }
    DroneParam = {
        'arm_length': 0.265,
        'motor_const': 1.465e-7,
        'moment_const': 0.01569
    }
    RotorParam = {
        'p': [25.16687, 0.003933, 515.605],
        'w_rotor_min': 2000,
        'w_rotor_max': 7200,
        'alpha_rotor_max': 15000,
        'jerk_rotor_max': 250000,
        'num_rotors': 3
    }
    MpcParam = {
        't_horizon': 0.30,
        'n_nodes': 15,
        'QArray': [5, 10, 1, 1, 50, 5],  # Much higher pitch penalty
        'R': 0.001,
        'R_jerk': 0.5,
        'q_max': 3.0,
        'K_p': 0.2,
        'deadband': 0.15
    }

    # Create models
    print("\nInitializing models...")
    drone_model = S550_3D_Sim_Model(DynamicParam=DynParam)
    rotor_model = RotorModel(RotorParams=RotorParam)
    hexa_converter = HexaConverter(
        DroneParams=DroneParam,
        RotorParams=RotorParam,
        Dim=3
    )

    print("Creating adaptive jerk OCP...")
    controller = AdaptiveJerkOCP(DynParam, DroneParam, RotorParam, MpcParam)

    # Simulation parameters
    dt = 0.01
    tf = 10.0  # Reduced from 15s for initial test
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # Initial state - start at hover position with hover thrust
    m = DynParam['m']
    C_T = DroneParam['motor_const']
    T_hover_per_rotor = m * 9.81 / 6.0
    w_hover = np.sqrt(T_hover_per_rotor / C_T)

    # Start at hover: body z=1.0 means CM z = 1.0 + h_g = 1.258
    h_g = 0.258
    com_offset = DynParam['com_offset']
    initial_cm_pos = np.array([0.0 + com_offset[0], 1.0 + h_g + com_offset[2]])

    # Compute asymmetric hover thrust to compensate for COM offset
    # At hover: dqdt = (My + x_off * f) / Jyy = 0
    # So My = -x_off * f = -(-0.0105) * (m * g) = 0.309 Nm
    # With My = 2 * l * sqrt(3) * (-u1 + u3), solve for u1, u3
    x_off = com_offset[0]
    l = DroneParam['arm_length']
    My_hover = -x_off * m * 9.81  # Moment needed to counteract COM offset
    # My = l * sqrt(3) * (-u1 + u3) = l * sqrt(3) * delta_u
    delta_u = My_hover / (l * np.sqrt(3))  # u3 - u1

    # Total thrust constraint: u1 + u2 + u3 = m * g / 2
    u_total = m * 9.81 / 2.0  # Total thrust from 3 rotor groups
    u_avg = u_total / 3.0  # Average thrust per group

    # Solve: u1 + u2 + u3 = u_total, u3 - u1 = delta_u, u2 = u_avg
    # u1 + u3 = u_total - u_avg = 2*u_avg
    # u3 - u1 = delta_u
    # => u3 = u_avg + delta_u/2, u1 = u_avg - delta_u/2
    u1_hover = u_avg - delta_u / 2.0
    u2_hover = u_avg
    u3_hover = u_avg + delta_u / 2.0

    w1_hover = np.sqrt(u1_hover / C_T)
    w2_hover = np.sqrt(u2_hover / C_T)
    w3_hover = np.sqrt(u3_hover / C_T)

    print(f"\nAsymmetric hover speeds for COM offset compensation:")
    print(f"  My_hover = {My_hover:.4f} Nm")
    print(f"  w_hover = [{w1_hover:.1f}, {w2_hover:.1f}, {w3_hover:.1f}] rad/s")

    s_drone, s_rotor = state_initialize(np.array([w1_hover, w2_hover, w3_hover]), initial_cm_pos, Dim=3)

    # Disable ground contact for cleaner hover test
    drone_model.is_contact = False

    # Target: hover at (0, 1.0)
    p_des = np.array([0.0, 1.0])
    v_des = np.array([0.0, 0.0])
    ref = np.concatenate([p_des, v_des])

    print(f"\nTarget hover position: x={p_des[0]:.2f}, z={p_des[1]:.2f} m")
    print(f"Simulation time: {tf:.1f} s, dt: {dt*1000:.1f} ms")
    print(f"Hover rotor speed: {w_hover:.1f} rad/s")
    print(f"Hover thrust per rotor: {T_hover_per_rotor:.3f} N")
    print(f"Initial CM position: {initial_cm_pos}")
    print(f"Initial body state: {drone_model.get_state(s_drone)}")

    # Data storage
    pos_hist = []
    vel_hist = []
    pitch_hist = []
    pos_des_hist = []
    w_rotor_hist = []
    thrust_cmd_hist = []
    thrust_actual_hist = []
    j_z_hist = []
    j_x_hist = []
    eta_hist = []

    alpha_max = RotorParam['alpha_rotor_max']
    num_rotors = RotorParam['num_rotors']
    C_T = DroneParam['motor_const']

    print("\nRunning simulation...")

    for i in range(N - 1):
        t = t_sim[i]

        # Get state
        s_body = drone_model.get_state(s_drone)
        p = s_body[0:2]
        v_body = s_body[2:4]
        theta = s_body[4]
        q = s_body[5]

        # World frame velocity
        from utils.math_tool import pitch_to_rotm
        v_world = pitch_to_rotm(theta) @ v_body

        w_rotor, alpha_rotor = rotor_model.unpack_state(s_rotor)

        # Compute actual thrust
        T_actual = 2.0 * C_T * np.sum(w_rotor**2)

        # Solve NMPC
        status, w_cmd, info = controller.solve(s_body, ref, T_actual)

        if status != 0 and i % 100 == 0:
            print(f"  Warning: solver status {status} at t={t:.2f}s")

        # Store data
        pos_hist.append(p.copy())
        vel_hist.append(v_world.copy())
        pitch_hist.append(theta)
        pos_des_hist.append(p_des.copy())
        w_rotor_hist.append(w_rotor.copy())
        thrust_cmd_hist.append(2.0 * C_T * np.sum(w_cmd**2))
        thrust_actual_hist.append(T_actual)
        j_z_hist.append(info['j_z_limit'])
        j_x_hist.append(info['j_x_limit'])
        eta_hist.append(info['stats']['eta_mean'])

        # Simulate dynamics
        t_ode = [t, t + dt]

        # Rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_model.dynamics, s_rotor, w_cmd, t_ode)
        s_rotor[num_rotors:] = np.clip(s_rotor[num_rotors:], -alpha_max, alpha_max)

        # Compute control input from actual rotor speeds
        u_actual = hexa_converter.compute_u(s_rotor[:num_rotors])

        # Drone dynamics
        s_drone = custom_rk4.do_step(drone_model.dynamics, s_drone, u_actual, t_ode)

        # Ground clamping
        s_drone = drone_model.clamp_ground(s_drone)

        # Progress
        if i % 200 == 0:
            T_cmd = 2.0 * C_T * np.sum(w_cmd**2)
            print(f"  t={t:.2f}s, x={p[0]:.3f}m, z={p[1]:.3f}m, "
                  f"pitch={np.rad2deg(theta):.2f}deg, "
                  f"T_cmd={T_cmd:.2f}N, w_cmd=[{w_cmd[0]:.0f}, {w_cmd[1]:.0f}, {w_cmd[2]:.0f}]")

    # Convert to arrays
    pos_hist = np.array(pos_hist)
    vel_hist = np.array(vel_hist)
    pitch_hist = np.array(pitch_hist)
    pos_des_hist = np.array(pos_des_hist)
    w_rotor_hist = np.array(w_rotor_hist)
    thrust_cmd_hist = np.array(thrust_cmd_hist)
    thrust_actual_hist = np.array(thrust_actual_hist)
    j_z_hist = np.array(j_z_hist)
    j_x_hist = np.array(j_x_hist)
    eta_hist = np.array(eta_hist)

    # Plot results
    print("\nGenerating plots...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Position X
    axes[0, 0].plot(t_sim[:-1], pos_hist[:, 0], 'b-', label='x')
    axes[0, 0].plot(t_sim[:-1], pos_des_hist[:, 0], 'r--', label='x_des')
    axes[0, 0].set_ylabel('X [m]')
    axes[0, 0].set_title('Position X')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Position Z
    axes[0, 1].plot(t_sim[:-1], pos_hist[:, 1], 'b-', label='z')
    axes[0, 1].plot(t_sim[:-1], pos_des_hist[:, 1], 'r--', label='z_des')
    axes[0, 1].set_ylabel('Z [m]')
    axes[0, 1].set_title('Position Z')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Pitch
    axes[1, 0].plot(t_sim[:-1], np.rad2deg(pitch_hist), 'b-')
    axes[1, 0].set_ylabel('Pitch [deg]')
    axes[1, 0].set_title('Pitch Angle')
    axes[1, 0].grid(True)

    # Thrust comparison
    axes[1, 1].plot(t_sim[:-1], thrust_cmd_hist, 'b-', label='Commanded', alpha=0.7)
    axes[1, 1].plot(t_sim[:-1], thrust_actual_hist, 'r-', label='Actual', alpha=0.7)
    axes[1, 1].axhline(y=3.0*9.81, color='k', linestyle='--', alpha=0.5, label='Hover')
    axes[1, 1].set_ylabel('Thrust [N]')
    axes[1, 1].set_title('Total Thrust')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Jerk limits
    axes[2, 0].plot(t_sim[:-1], j_z_hist, 'b-', label='j_z limit')
    axes[2, 0].plot(t_sim[:-1], j_x_hist, 'g-', label='j_x limit')
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_ylabel('Jerk Limit [m/s³]')
    axes[2, 0].set_title('Adaptive Jerk Limits')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Tracking ratio (eta)
    axes[2, 1].plot(t_sim[:-1], eta_hist, 'g-')
    axes[2, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect')
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('eta')
    axes[2, 1].set_title('Actuator Tracking (eta)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig('ref_generation/adaptive_jerk_ocp_results.png', dpi=150)
    print("  Saved: ref_generation/adaptive_jerk_ocp_results.png")

    # Print summary
    print("\n" + "-"*40)
    print("Simulation Summary:")
    print("-"*40)
    final_pos = pos_hist[-1]
    pos_error = np.linalg.norm(final_pos - p_des)
    print(f"  Final position: x={final_pos[0]:.3f}, z={final_pos[1]:.3f} m")
    print(f"  Position error: {pos_error:.4f} m")
    print(f"  Final pitch: {np.rad2deg(pitch_hist[-1]):.2f} deg")
    print(f"  Mean j_z: {np.mean(j_z_hist):.2f} m/s³")
    print(f"  Mean j_x: {np.mean(j_x_hist):.2f} m/s³")
    print(f"  Mean eta: {np.mean(eta_hist[eta_hist > 0]):.3f}")

    if pos_error < 0.1:
        print("\n[PASS] Hover test passed!")
    else:
        print("\n[WARN] Large position error")

    print("="*60)


if __name__ == '__main__':
    main()
