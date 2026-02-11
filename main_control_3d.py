#!/usr/bin/env python3
"""
3DOF (x, z, theta) Drone Simulation with Contact Dynamics

Supports:
- PD control with pitch
- HGDO disturbance observer

Examples:
    python3 main_control_3d.py --dob none
    python3 main_control_3d.py --dob hgdo

Author: Geonwoo Kwon
Date: 2026-02-11
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils import yaml_loader
from utils.state_initializer import state_initialize
from sim_model.S550_3d_model import S550_3D_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.custom_ode import custom_rk4


def load_parameters_3d(dob_type):
    """Load parameters for 3DOF simulation"""
    params = {}

    # Load True parameters from simulator_3d config
    config_sim = yaml_loader.load_yaml('config/simulator/simulator_3d.yaml')
    params['true_dynamic_params'] = yaml_loader.get_dynamic_params(config_sim)
    params['true_drone_params'] = yaml_loader.get_drone_params(config_sim)
    params['true_rotor_params'] = yaml_loader.get_rotor_params(config_sim)
    params['sim_params'] = yaml_loader.get_sim_params(config_sim)

    # Load NOMINAL parameters from pd_3d_params
    config_control = yaml_loader.load_yaml('config/control/pd/pd_3d_params.yaml')
    params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
    params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
    params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)
    params['gain_params'] = yaml_loader.get_pd_gain_params(config_control)

    # Load DOB parameters
    if dob_type == 'hgdo':
        config_dob = yaml_loader.load_yaml('config/estimator/dob/hgdo_3d.yaml')
        params['dob_params'] = yaml_loader.get_hgdo_params(config_dob)

    return params


def setup_controller_3d(dob_type, params):
    """Setup controller and DOB for 3DOF"""
    from control.PID.pitch_control import PitchControl

    use_dob = (dob_type != 'none')
    controller = PitchControl(DynamicParams=params['nominal_dynamic_params'],
                              GainParams=params['gain_params'],
                              DobMode=use_dob)

    dob = None
    if dob_type == 'hgdo':
        from estimator.dob.hgdo.hgdo_3d import HGDO3D
        dob = HGDO3D(DynamicParams=params['nominal_dynamic_params'],
                     DroneParams=params['nominal_drone_params'],
                     RotorParams=params['nominal_rotor_params'],
                     DobParams=params['dob_params'])

    return controller, dob


def plot_results_3d(t, drone_data, rotor_data, ref_data, dob_data):
    """Plot results for 3DOF simulation"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

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

    # Pitch
    axes[1, 0].plot(t, np.rad2deg(drone_data['pitch']), 'b-', label='pitch')
    axes[1, 0].set_ylabel('Pitch [deg]')
    axes[1, 0].set_title('Pitch Angle')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Pitch rate
    axes[1, 1].plot(t, np.rad2deg(drone_data['pitch_rate']), 'b-', label='pitch_rate')
    axes[1, 1].set_ylabel('Pitch Rate [deg/s]')
    axes[1, 1].set_title('Pitch Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Rotor speeds
    for i in range(rotor_data['w_rotor'].shape[1]):
        axes[2, 0].plot(t, rotor_data['w_rotor'][:, i], label=f'w{i+1}')
    axes[2, 0].set_ylabel('Rotor Speed [RPM]')
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_title('Rotor Speeds')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # DOB estimates
    axes[2, 1].plot(t, dob_data['f_est'][:, 0], 'b-', label='f_ext_x')
    axes[2, 1].plot(t, dob_data['f_est'][:, 1], 'g-', label='f_ext_z')
    axes[2, 1].plot(t, dob_data['tau_est'], 'r-', label='tau_ext')
    axes[2, 1].set_ylabel('Disturbance')
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_title('DOB Estimates')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig('results_3d.png', dpi=150)
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='3DOF Control simulation with contact dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python3 main_control_3d.py --dob none
        python3 main_control_3d.py --dob hgdo
        """
    )

    parser.add_argument('--dob', type=str, default='hgdo',
                        choices=['none', 'hgdo'],
                        help='Disturbance observer: none or hgdo')

    args = parser.parse_args()
    dob_type = args.dob

    print(f"\n{'='*60}")
    print(f"3DOF Simulation (x, z, theta)")
    print(f"DOB type: {dob_type}")
    print(f"{'='*60}")

    # Load parameters
    params = load_parameters_3d(dob_type)

    print("Parameter Configuration:")
    print("- Simulator: Uses TRUE parameters from config/simulator/simulator_3d.yaml")
    print("- Controller & DOB: Use NOMINAL parameters from config/control/pd/pd_3d_params.yaml")
    print(f"- CoM offset: {params['true_dynamic_params']['com_offset']}")

    # Create simulation models (TRUE parameters)
    drone_sim_model = S550_3D_Sim_Model(DynamicParam=params['true_dynamic_params'])
    rotor_sim_model = RotorModel(RotorParams=params['true_rotor_params'])
    hexa_converter = HexaConverter(DroneParams=params['true_drone_params'],
                                   RotorParams=params['true_rotor_params'],
                                   Dim=3)

    # Setup controller and DOB (NOMINAL parameters)
    controller, dob = setup_controller_3d(dob_type, params)

    # State initialization
    w_rotor_idle = params['sim_params']['w_rotor_idle']
    s_drone, s_rotor = state_initialize(w_rotor_idle, Dim=3)

    # Simulation parameters
    tf = params['sim_params']['tf']
    dt = params['sim_params']['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # For hard clamp
    alpha_max = params['true_rotor_params']['alpha_rotor_max']
    num_rotors = params['true_rotor_params']['num_rotors']

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

    # Reference: simple hover at z=1.0
    p_des = np.array([0.0, 1.0])  # x, z
    v_des = np.array([0.0, 0.0])  # vx, vz
    ref = np.concatenate([p_des, v_des])

    print(f"\nTarget position: x={p_des[0]:.2f}, z={p_des[1]:.2f} m")
    print(f"Simulation time: {tf:.1f} s, dt: {dt*1000:.1f} ms\n")

    # Main simulation loop
    for i in range(N - 1):
        # Get body frame state from simulation
        s_body = drone_sim_model.get_state(s_drone)

        # Unpack state
        p = s_body[0:2]
        v_body = s_body[2:4]
        theta = s_body[4]
        q = s_body[5]

        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)

        # DOB estimate
        if dob is not None and i > 1:
            d_est = dob.dob_estimate(t_sim[i-1], t_sim[i], w_rotor, s_body)
        else:
            d_est = np.zeros(3)

        f_est = d_est[0:2]
        tau_est = d_est[2]

        # Store history
        pos_hist.append(p.copy())
        vel_hist.append(v_body.copy())
        pitch_hist.append(theta)
        pitch_rate_hist.append(q)

        pos_des_hist.append(p_des.copy())
        vel_des_hist.append(v_des.copy())

        w_rotor_hist.append(w_rotor.copy())
        alpha_rotor_hist.append(alpha_rotor.copy())

        f_est_hist.append(f_est.copy())
        tau_est_hist.append(tau_est)

        # Compute control
        u = controller.compute_u(s_body, ref, d_est, dt)
        w_cmd = hexa_converter.compute_des_rotor_speed(u)

        # Simulation step
        t_ode = [t_sim[i], t_sim[i + 1]]

        # Simulate rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)

        # Hard clamp for rotor acceleration
        s_rotor[num_rotors:] = np.clip(s_rotor[num_rotors:], -alpha_max, alpha_max)

        # Compute actual control input from rotor speeds
        u_actual = hexa_converter.compute_u(s_rotor[:num_rotors])

        # Simulate drone dynamics
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u_actual, t_ode)

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
        'tau_est': np.array(tau_est_hist)
    }

    # Print final statistics
    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"Final position: x={drone_data['pos'][-1, 0]:.3f}, z={drone_data['pos'][-1, 1]:.3f} m")
    print(f"Final pitch: {np.rad2deg(drone_data['pitch'][-1]):.2f} deg")
    print(f"{'='*60}")

    # Plot results
    plot_results_3d(t_sim[:-1], drone_data, rotor_data, ref_data, dob_data)


if __name__ == '__main__':
    main()
