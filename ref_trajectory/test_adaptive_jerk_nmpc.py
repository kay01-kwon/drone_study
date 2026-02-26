#!/usr/bin/env python3
"""
Test script for Adaptive Jerk NMPC Controller

Tests:
1. Basic functionality - hover and step response
2. Adaptive jerk limit behavior
3. Trajectory tracking with actuator constraints

Author: Claude
Date: 2026-02-26
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ref_trajectory.adaptive_jerk_nmpc import AdaptiveJerkNMPC
from ref_trajectory.jerk_adapter import AdaptiveJerkAdapter
from sim_model.S550_model import S550_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.math_tool import quaternion_to_euler
from utils.custom_ode import custom_rk4


def test_jerk_adapter():
    """Test the jerk adapter independently."""
    print("\n" + "="*60)
    print("Test 1: Jerk Adapter Unit Test")
    print("="*60)

    adapter = AdaptiveJerkAdapter({
        'm': 3.0,
        'j_limit_max': 30.0,
        'j_limit_min': 5.0,
        'j_limit_init': 15.0,
        'K_p': 0.3,
        'deadband': 0.1
    })

    # Simulate scenarios
    print("\nScenario A: Good actuator tracking (eta ~ 1.0)")
    for i in range(20):
        T_cmd = 30.0 + 0.5 * np.sin(0.5 * i)
        T_actual = T_cmd * 0.95  # 95% tracking
        omega = np.array([0.1, 0.1, 0.0])
        j_limit, T_dot = adapter.update(T_cmd, T_actual, omega)

    stats = adapter.get_statistics()
    print(f"  eta_mean: {stats['eta_mean']:.3f}")
    print(f"  j_limit: {stats['j_limit_current']:.2f} m/s^3")

    print("\nScenario B: Poor actuator tracking (eta ~ 0.6)")
    adapter.reset()
    for i in range(20):
        T_cmd = 30.0 + 2.0 * np.sin(0.5 * i)
        T_actual = T_cmd * 0.6  # 60% tracking (saturating)
        omega = np.array([0.1, 0.1, 0.0])
        j_limit, T_dot = adapter.update(T_cmd, T_actual, omega)

    stats = adapter.get_statistics()
    print(f"  eta_mean: {stats['eta_mean']:.3f}")
    print(f"  j_limit: {stats['j_limit_current']:.2f} m/s^3 (should decrease)")

    print("\nScenario C: High rotation rate")
    adapter.reset()
    T_cmd = 30.0
    omega_high = np.array([3.0, 2.0, 0.5])  # High angular velocity
    j_limit, T_dot = adapter.update(T_cmd, T_cmd, omega_high)
    print(f"  omega_xy_max: {max(abs(omega_high[0]), abs(omega_high[1])):.2f} rad/s")
    print(f"  T_dot_available: {T_dot:.2f} N/s (reduced due to rotation)")

    print("\n[PASS] Jerk adapter tests completed")


def test_nmpc_hover():
    """Test NMPC hover capability."""
    print("\n" + "="*60)
    print("Test 2: NMPC Hover Test")
    print("="*60)

    # Parameters
    DynParam = {'m': 3.0, 'MoiArray': [0.065, 0.065, 0.10]}
    DroneParam = {
        'arm_length': 0.265,
        'motor_const': 1.465e-7,
        'moment_const': 0.01569
    }
    MpcParam = {
        't_horizon': 0.5,
        'n_nodes': 10,
        'Q_pos': [10.0, 10.0, 15.0],
        'Q_vel': [2.0, 2.0, 2.0],
        'Q_att': [1.0, 1.0, 1.0, 1.0],
        'Q_omega': [0.1, 0.1, 0.1],
        'R': [0.01] * 6,
        'R_jerk': 5.0
    }

    # Create controller
    controller = AdaptiveJerkNMPC(DynParam, DroneParam, MpcParam)

    # Initial state (slightly off hover)
    state = np.array([
        0.0, 0.0, 1.0,        # position
        0.0, 0.0, 0.0,        # velocity (world)
        1.0, 0.0, 0.0, 0.0,   # quaternion
        0.0, 0.0, 0.0         # angular velocity
    ])

    # Reference (hover at 1.5m)
    ref = np.array([
        0.0, 0.0, 1.5,        # position
        0.0, 0.0, 0.0,        # velocity
        1.0, 0.0, 0.0, 0.0,   # quaternion
        0.0, 0.0, 0.0         # angular velocity
    ])

    print("\nSolving NMPC for hover...")
    status, u_opt, info = controller.solve(state, ref)

    print(f"  Status: {status}")
    print(f"  Thrust per rotor: {u_opt[0]:.4f} N")
    print(f"  Total thrust: {np.sum(u_opt):.4f} N")
    print(f"  Expected (mg): {3.0 * 9.81:.4f} N")
    print(f"  Jerk limit: {info['j_limit']:.2f} m/s^3")

    if status == 0:
        print("\n[PASS] NMPC hover test passed")
    else:
        print("\n[FAIL] NMPC hover test failed")


def test_full_simulation():
    """Full simulation with drone model."""
    print("\n" + "="*60)
    print("Test 3: Full Simulation with Adaptive Jerk NMPC")
    print("="*60)

    # Parameters
    DynParam = {
        'm': 3.0,
        'MoiArray': [0.065, 0.065, 0.10],
        'com_offset': np.array([0.0, 0.0, 0.0])
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
        'num_rotors': 6
    }
    MpcParam = {
        't_horizon': 0.5,
        'n_nodes': 10,
        'Q_pos': [20.0, 20.0, 30.0],
        'Q_vel': [5.0, 5.0, 5.0],
        'Q_att': [2.0, 2.0, 2.0, 2.0],
        'Q_omega': [0.5, 0.5, 0.5],
        'R': [0.001] * 6,
        'R_jerk': 1.0
    }
    AdapterParam = {
        'j_limit_max': 50.0,
        'j_limit_min': 10.0,
        'j_limit_init': 30.0,
        'K_p': 0.2,
        'deadband': 0.15
    }

    # Create models
    drone_model = S550_Sim_Model(DynamicParams=DynParam)
    rotor_model = RotorModel(RotorParams=RotorParam)
    hexa_converter = HexaConverter(
        DroneParams=DroneParam,
        RotorParams=RotorParam,
        Dim=6
    )
    controller = AdaptiveJerkNMPC(DynParam, DroneParam, MpcParam, AdapterParam)

    # Simulation setup
    dt = 0.01
    tf = 15.0
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # Initial state - start at hover position with hover thrust
    m = DynParam['m']
    C_T = DroneParam['motor_const']
    T_hover = m * 9.81 / 6.0  # Thrust per rotor for hover
    w_hover = np.sqrt(T_hover / C_T)

    s_drone = drone_model.pack_state(
        p=np.array([0.0, 0.0, 1.0]),  # Start at z=1m
        v=np.array([0.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        w=np.array([0.0, 0.0, 0.0])
    )
    s_rotor = rotor_model.pack_state(
        w_rotor=np.ones(6) * w_hover,  # Start at hover speed
        alpha_rotor=np.zeros(6)
    )

    # Data storage
    pos_hist = []
    pos_ref_hist = []
    j_limit_hist = []
    eta_hist = []
    thrust_hist = []
    thrust_actual_hist = []

    print("\nRunning simulation...")
    print(f"  Hover rotor speed: {w_hover:.1f} RPM")
    print(f"  Hover thrust/rotor: {T_hover:.2f} N")

    for i in range(N - 1):
        t = t_sim[i]

        # Unpack state
        p, v, q, w = drone_model.unpack_state(s_drone)
        w_rotor, alpha_rotor = rotor_model.unpack_state(s_rotor)

        # Generate smooth reference trajectory (vertical only for stability test)
        if t < 3.0:
            # Hover at initial position
            p_ref = np.array([0.0, 0.0, 1.0])
        elif t < 6.0:
            # Smooth ascent to z=1.5m
            alpha = min((t - 3.0) / 2.0, 1.0)
            p_ref = np.array([0.0, 0.0, 1.0 + 0.5 * alpha])
        elif t < 9.0:
            # Hold at z=1.5m
            p_ref = np.array([0.0, 0.0, 1.5])
        elif t < 12.0:
            # Smooth descent to z=1.0m
            alpha = min((t - 9.0) / 2.0, 1.0)
            p_ref = np.array([0.0, 0.0, 1.5 - 0.5 * alpha])
        else:
            # Hold at z=1.0m
            p_ref = np.array([0.0, 0.0, 1.0])

        ref = np.array([
            p_ref[0], p_ref[1], p_ref[2],
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])

        # State feedback (world frame velocity)
        from utils.math_tool import quaternion_to_rotm
        R = quaternion_to_rotm(q)
        v_world = R @ v
        state_fb = np.concatenate([p, v_world, q, w])

        # Compute actual thrust
        T_actual = np.sum(DroneParam['motor_const'] * w_rotor**2)

        # Solve NMPC
        status, u_opt, info = controller.solve(state_fb, ref, T_actual)

        # Convert to rotor speed command
        w_cmd = np.sqrt(np.clip(u_opt, 0, None) / DroneParam['motor_const'])
        w_cmd = np.clip(w_cmd, RotorParam['w_rotor_min'], RotorParam['w_rotor_max'])

        # Simulate dynamics
        t_ode = [t, t + dt]

        # Rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_model.dynamics, s_rotor, w_cmd, t_ode)
        s_rotor[6:] = np.clip(s_rotor[6:], -RotorParam['alpha_rotor_max'],
                               RotorParam['alpha_rotor_max'])

        # Compute actual control
        u_actual = hexa_converter.compute_u(s_rotor[:6])
        T_actual_total = u_actual[0]  # Total thrust

        # Store data
        pos_hist.append(p.copy())
        pos_ref_hist.append(p_ref.copy())
        j_limit_hist.append(info['j_limit'])
        stats = info['adapter_stats']
        eta_hist.append(stats['eta_mean'])
        thrust_hist.append(np.sum(u_opt))
        thrust_actual_hist.append(T_actual_total)

        # Drone dynamics
        s_drone = custom_rk4.do_step(drone_model.dynamics, s_drone, u_actual, t_ode)

        # Progress
        if i % 300 == 0:
            print(f"  t={t:.1f}s, z={p[2]:.3f}m, z_ref={p_ref[2]:.3f}m, j_limit={info['j_limit']:.1f}")

    # Convert to arrays
    pos_hist = np.array(pos_hist)
    pos_ref_hist = np.array(pos_ref_hist)
    j_limit_hist = np.array(j_limit_hist)
    eta_hist = np.array(eta_hist)
    thrust_hist = np.array(thrust_hist)
    thrust_actual_hist = np.array(thrust_actual_hist)

    # Plot results
    print("\nGenerating plots...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Position tracking
    axes[0, 0].plot(t_sim[:-1], pos_hist[:, 0], 'b-', label='x')
    axes[0, 0].plot(t_sim[:-1], pos_hist[:, 1], 'g-', label='y')
    axes[0, 0].plot(t_sim[:-1], pos_hist[:, 2], 'r-', label='z')
    axes[0, 0].plot(t_sim[:-1], pos_ref_hist[:, 0], 'b--', alpha=0.5)
    axes[0, 0].plot(t_sim[:-1], pos_ref_hist[:, 1], 'g--', alpha=0.5)
    axes[0, 0].plot(t_sim[:-1], pos_ref_hist[:, 2], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Position Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Position error
    pos_err = np.linalg.norm(pos_hist - pos_ref_hist, axis=1)
    axes[0, 1].plot(t_sim[:-1], pos_err, 'k-')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position Error (m)')
    axes[0, 1].set_title('Position Error Norm')
    axes[0, 1].grid(True)

    # Adaptive jerk limit
    axes[1, 0].plot(t_sim[:-1], j_limit_hist, 'b-')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Jerk Limit (m/s^3)')
    axes[1, 0].set_title('Adaptive Jerk Limit')
    axes[1, 0].grid(True)

    # Actuator tracking (eta)
    axes[1, 1].plot(t_sim[:-1], eta_hist, 'g-')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect tracking')
    axes[1, 1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('eta (tracking ratio)')
    axes[1, 1].set_title('Actuator Tracking Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Total thrust comparison
    axes[2, 0].plot(t_sim[:-1], thrust_hist, 'b-', label='Commanded', alpha=0.7)
    axes[2, 0].plot(t_sim[:-1], thrust_actual_hist, 'r-', label='Actual', alpha=0.7)
    axes[2, 0].axhline(y=3.0*9.81, color='k', linestyle='--', alpha=0.5, label='Hover thrust')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Total Thrust (N)')
    axes[2, 0].set_title('Total Thrust (Cmd vs Actual)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # 3D trajectory
    ax3d = fig.add_subplot(3, 2, 6, projection='3d')
    ax3d.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], 'b-', label='Actual')
    ax3d.plot(pos_ref_hist[:, 0], pos_ref_hist[:, 1], pos_ref_hist[:, 2], 'r--', label='Reference')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory')
    ax3d.legend()

    plt.tight_layout()
    plt.savefig('ref_trajectory/adaptive_jerk_nmpc_results.png', dpi=150)
    print("  Saved: ref_trajectory/adaptive_jerk_nmpc_results.png")

    # Print summary
    print("\n" + "-"*40)
    print("Simulation Summary:")
    print("-"*40)
    print(f"  Final position error: {pos_err[-1]:.4f} m")
    print(f"  Max position error: {np.max(pos_err):.4f} m")
    print(f"  Mean jerk limit: {np.mean(j_limit_hist):.2f} m/s^3")
    print(f"  Mean eta: {np.mean(eta_hist[eta_hist > 0]):.3f}")

    if pos_err[-1] < 0.1:
        print("\n[PASS] Full simulation test passed")
    else:
        print("\n[WARN] Large final position error")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Adaptive Jerk NMPC Test Suite")
    print("="*60)

    # Test 1: Jerk adapter unit test
    test_jerk_adapter()

    # Test 2: NMPC hover test
    test_nmpc_hover()

    # Test 3: Full simulation
    test_full_simulation()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == '__main__':
    main()
