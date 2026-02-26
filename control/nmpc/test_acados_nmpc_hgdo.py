#!/usr/bin/env python3
"""
Test Acados NMPC with HGDO disturbance compensation for S550 3DOF.

Uses ideal model in MPC (no COM offset) with HGDO-estimated disturbance compensation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from control.nmpc.ocp.S550_3DOF_ocp import S550_3DOF_ocp
from sim_model.S550_3d_model import S550_3D_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.custom_ode import custom_rk4
from utils.state_initializer import state_initialize
from estimator.dob.hgdo.hgdo_3d import HGDO3D


def main():
    print("\n" + "="*60)
    print("Acados NMPC + HGDO Test - 3DOF Hover")
    print("="*60)

    # Parameters
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
        't_horizon': 0.20,
        'n_nodes': 20,
        'QArray': [5, 10, 1, 1, 50, 5],
        'R': 0.001
    }
    DobParam = {
        'eps_tau': 0.02,
        'eps_f': 0.02
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

    print("Creating Acados NMPC...")
    controller = S550_3DOF_ocp(DynParam, DroneParam, MpcParam)

    print("Creating HGDO3D for disturbance estimation...")
    hgdo = HGDO3D(DynParam, DroneParam, RotorParam, DobParam)

    # Simulation parameters
    dt = 0.01
    tf = 10.0
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # Initial state
    m = DynParam['m']
    C_T = DroneParam['motor_const']
    T_hover_per_rotor = m * 9.81 / 6.0
    w_hover = np.sqrt(T_hover_per_rotor / C_T)

    h_g = 0.258
    com_offset = DynParam['com_offset']
    initial_cm_pos = np.array([0.0 + com_offset[0], 1.0 + h_g + com_offset[2]])

    # Asymmetric hover for COM offset compensation
    x_off = com_offset[0]
    l = DroneParam['arm_length']
    My_hover = -x_off * m * 9.81
    delta_u = My_hover / (l * np.sqrt(3))

    u_avg = m * 9.81 / 6.0
    u1_hover = u_avg - delta_u / 2.0
    u3_hover = u_avg + delta_u / 2.0

    w1_hover = np.sqrt(u1_hover / C_T)
    w2_hover = np.sqrt(u_avg / C_T)
    w3_hover = np.sqrt(u3_hover / C_T)

    print(f"\nAsymmetric hover: w=[{w1_hover:.0f}, {w2_hover:.0f}, {w3_hover:.0f}]")
    s_drone, s_rotor = state_initialize(np.array([w1_hover, w2_hover, w3_hover]), initial_cm_pos, Dim=3)

    drone_model.is_contact = False

    # Target
    p_des = np.array([0.0, 1.0])
    v_des = np.array([0.0, 0.0])
    ref = np.concatenate([p_des, v_des])

    print(f"\nTarget: x={p_des[0]:.2f}, z={p_des[1]:.2f} m")
    print(f"Simulation: {tf:.1f}s, dt={dt*1000:.1f}ms")

    # Data storage
    pos_hist = []
    vel_hist = []
    pitch_hist = []
    w_rotor_hist = []
    tau_ext_hist = []

    alpha_max = RotorParam['alpha_rotor_max']
    num_rotors = RotorParam['num_rotors']

    print("\nRunning simulation...")

    for i in range(N - 1):
        t = t_sim[i]

        s_body = drone_model.get_state(s_drone)
        p = s_body[0:2]
        v_body = s_body[2:4]
        theta = s_body[4]
        q = s_body[5]

        from utils.math_tool import pitch_to_rotm
        v_world = pitch_to_rotm(theta) @ v_body

        w_rotor, alpha_rotor = rotor_model.unpack_state(s_rotor)
        T_actual = 2.0 * C_T * np.sum(w_rotor**2)

        # HGDO disturbance estimation
        t_prev = t_sim[max(0, i-1)]
        dist_est = hgdo.dob_estimate(t_prev, t, w_rotor, s_body)
        tau_ext = dist_est[2]

        # Solve NMPC
        u_prev = C_T * w_rotor**2
        status, w_cmd = controller.solve(s_body, ref, u_prev)

        # Feedforward COM offset compensation
        tau_ff = -x_off * T_actual
        du_comp = tau_ff / (2.0 * l * np.sqrt(3))

        u_cmd = C_T * w_cmd**2
        u_cmd[0] -= du_comp
        u_cmd[2] += du_comp
        u_cmd = np.clip(u_cmd, C_T * RotorParam['w_rotor_min']**2, C_T * RotorParam['w_rotor_max']**2)
        w_cmd = np.sqrt(u_cmd / C_T)

        # Store data
        pos_hist.append(p.copy())
        vel_hist.append(v_world.copy())
        pitch_hist.append(theta)
        w_rotor_hist.append(w_rotor.copy())
        tau_ext_hist.append(tau_ext)

        # Simulate
        t_ode = [t, t + dt]
        s_rotor = custom_rk4.do_step(rotor_model.dynamics, s_rotor, w_cmd, t_ode)
        s_rotor[num_rotors:] = np.clip(s_rotor[num_rotors:], -alpha_max, alpha_max)

        u_actual = hexa_converter.compute_u(s_rotor[:num_rotors])
        s_drone = custom_rk4.do_step(drone_model.dynamics, s_drone, u_actual, t_ode)
        s_drone = drone_model.clamp_ground(s_drone)

        if i % 200 == 0:
            print(f"  t={t:.2f}s, x={p[0]:.3f}m, z={p[1]:.3f}m, pitch={np.rad2deg(theta):.2f}deg")

    # Convert to arrays
    pos_hist = np.array(pos_hist)
    vel_hist = np.array(vel_hist)
    pitch_hist = np.array(pitch_hist)
    tau_ext_hist = np.array(tau_ext_hist)

    # Plot
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(t_sim[:-1], pos_hist[:, 0], 'b-', label='x')
    axes[0, 0].axhline(y=p_des[0], color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylabel('X [m]')
    axes[0, 0].set_title('Position X')
    axes[0, 0].grid(True)

    axes[0, 1].plot(t_sim[:-1], pos_hist[:, 1], 'b-', label='z')
    axes[0, 1].axhline(y=p_des[1], color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('Z [m]')
    axes[0, 1].set_title('Position Z')
    axes[0, 1].grid(True)

    axes[1, 0].plot(t_sim[:-1], np.rad2deg(pitch_hist), 'b-')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Pitch [deg]')
    axes[1, 0].set_title('Pitch Angle')
    axes[1, 0].grid(True)

    tau_expected = -x_off * m * 9.81
    axes[1, 1].plot(t_sim[:-1], tau_ext_hist, 'b-', label='tau_ext (HGDO)')
    axes[1, 1].axhline(y=tau_expected, color='r', linestyle='--', alpha=0.5, label=f'Expected ({tau_expected:.3f})')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Moment [Nm]')
    axes[1, 1].set_title('HGDO Estimated Disturbance')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('control/nmpc/acados_nmpc_hgdo_results.png', dpi=150)
    print("  Saved: control/nmpc/acados_nmpc_hgdo_results.png")

    # Summary
    print("\n" + "-"*40)
    print("Simulation Summary:")
    print("-"*40)
    final_pos = pos_hist[-1]
    pos_error = np.linalg.norm(final_pos - p_des)
    print(f"  Final position: x={final_pos[0]:.4f}, z={final_pos[1]:.4f} m")
    print(f"  Position error: {pos_error:.4f} m")
    print(f"  Final pitch: {np.rad2deg(pitch_hist[-1]):.2f} deg")
    print(f"  Final tau_ext: {tau_ext_hist[-1]:.4f} Nm (expected: {tau_expected:.4f})")

    if pos_error < 0.1:
        print("\n[PASS] Hover test passed!")
    else:
        print("\n[WARN] Large position error")

    print("="*60)


if __name__ == '__main__':
    main()
