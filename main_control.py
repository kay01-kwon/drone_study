#!/usr/bin/env python3
"""

Supports multiple control methods, disturbance observers and dynamic parameter estimator:

Control types:
- nmpc: NMPC compensation DOB
- pd: Geometric control

DOB types:
- none: No disturbance observer
- hgdo: High Gain Disturbance Observer
- l1: L1 Adaptation

Examples:
    python3 main_control.py --control nmpc --dob none
    python3 main_control.py --control nmpc --dob l1
    python3 main_control.py --control pd --dob hgdo

Author: Geonwoo Kwon
Date: 2025-01-16
"""

import numpy as np
import argparse

from estimator.rls.dynamic_param_estimator import DynamicParamEstimator
from utils import parameter_loader
from utils.plot_sim_result import plot_results

from utils.state_initializer import state_initialize
from sim_model.S550_model import S550_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.math_tool import quaternion_to_euler
from utils.custom_ode import custom_rk4
from utils.print_tool import print_statistics

def main():
    # Parse command line arguments
    parser =  argparse.ArgumentParser(
        description='Control simulation with control and DOB selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python3 main_control.py --control nmpc --dob none
        python3 main_control.py --control nmpc --dob hgdo
        python3 main_control.py --control nmpc --dob l1
        python3 main_control.py --control pd --dob hgdo
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
                        help='Control mode: tracking (waypoint following) or regulation (fixed hover)')

    args = parser.parse_args()

    control_type = args.control
    dob_type = args.dob
    control_mode = args.mode

    print(f"\n{'='*60}")
    print(f"Control type: {control_type}")
    print(f"{'='*60}")

    # Load all parameters from split config files
    params = parameter_loader.load_parameters(control_type, dob_type)

    print("Parameter Configuration:")
    print("- Simulator: Uses TRUE parameters from config/simulator/simulator.yaml")
    print(f"- Controller & DOB: Use NOMINAL parameters from config/control/{control_type}/")
    print("  (Controller and DOB share the same nominal model)\n")

    # Create simulation models using TRUE parameters (actual system)
    drone_sim_model = S550_Sim_Model(DynamicParams=params['true_dynamic_params'])
    # For hard clamp, store maximum rotor acceleration
    alpha_max = params['true_rotor_params']['alpha_rotor_max']

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

    # State initialization
    w_rotor_idle = params['sim_params']['w_rotor_idle']

    # Initialize at ground level (like Gazebo)
    initial_pos = [0.0, 0.0, 0.0]  # Ground level
    s_drone, s_rotor = state_initialize(w_rotor_idle,
                                        initial_offset=initial_pos,
                                        Dim=6)

    # Simulation parameters
    tf = params['sim_params']['tf']
    dt = params['sim_params']['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

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

    # Parameter estimate storage
    m_est_hist = []
    com_x_est_hist = []
    com_y_est_hist = []

    from ref_generation.ref_generator import get_reference, unpack_ref

    # Print contorl mode information
    if control_mode == 'regulation':
        print("\n========== Regulation Control ==========")
        print(f"Setpoint position: {params['regulation_params']['setpoint_position']}")
        print(f"Setpoint yaw: {params['regulation_params']['setpoint_yaw']:.2f} rad")
        print("==========================================\n")

    # Main simulation loop
    for i in range(N-1):
        # Unpack state
        p, v, q, w = drone_sim_model.unpack_state(s_drone)
        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)
        roll, pitch, yaw = quaternion_to_euler(q)
        s_feedback = np.concatenate([p, v, q, w])

        # Get reference (tracking or regulation)
        if control_mode == 'tracking':
            if i == 0:
                ref = get_reference(t_sim[i], params['trajectory_params'])
            else:
                ref = get_reference(t_sim[i])
            p_des, v_des, yaw_des, yaw_des_dot = unpack_ref(ref)
        else:  # regulation
            p_des = params['regulation_params']['setpoint_position']
            v_des = np.zeros(3)
            yaw_des = params['regulation_params']['setpoint_yaw']
            yaw_des_dot = 0.0

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

        # Store history
        pos_hist.append(p)
        vel_hist.append(v)

        pos_des_hist.append(p_des)
        vel_des_hist.append(v_des)

        roll_hist.append(roll)
        pitch_hist.append(pitch)
        yaw_hist.append(yaw)

        yaw_des_hist.append(yaw_des)
        yaw_des_dot_hist.append(yaw_des_dot)

        w_rotor_hist.append(w_rotor)
        alpha_rotor_hist.append(alpha_rotor)

        f_est_hist.append(f_est)
        tau_est_hist.append(tau_est)

        m_est_hist.append(param_est[0])
        com_x_est_hist.append(param_est[1])
        com_y_est_hist.append(param_est[2])


        # Compute control
        if control_type == 'nmpc':
            status, w_cmd = controller.solve_for_trajectory(s_feedback, t_sim[i])

            if dob is not None:
                u_mpc = hexa_converter.compute_u(w_cmd)
                u_d_match = np.array([d_est[2],
                                      d_est[3],
                                      d_est[4],
                                      d_est[5]])
                u_comp = u_mpc - u_d_match
                # Without actuator model
                w_cmd = hexa_converter.compute_des_rotor_speed(u_comp)

            if status != 0 and i % 100 == 0:
                print(f"Warning: NMPC solver status {status} at t = {t_sim[i]:.2f}s")

        elif control_type == 'pd':
            u = controller.compute_u(s_feedback, ref, d_est)
            w_cmd = hexa_converter.compute_des_rotor_speed(u)

        # Simulation step
        t_ode = [t_sim[i], t_sim[i+1]]

        # Simulate rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)

        # Hard clamp for rotor acceleration
        s_rotor[6:] = np.clip(s_rotor[6:], -alpha_max, alpha_max)
        u = hexa_converter.compute_u(s_rotor[:6])

        # Simulate drone dynamics
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u, t_ode)

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
                'tau_est':np.array(tau_est_hist)}

    param_est_data = {'m_est': np.array(m_est_hist),
                      'com_x_est': np.array(com_x_est_hist),
                      'com_y_est': np.array(com_y_est_hist)}

    plot_results(control_type, dob_type, drone_state_data,
                 rotor_state_data, ref_data, dob_data, param_est_data,
                 params['true_dynamic_params'])

    print_statistics(control_type, dob_type,
                     drone_state_data, rotor_state_data,
                     ref_data, dob_data, param_est_data,
                     params['true_dynamic_params'])

    # Cleanup for NMPC
    if control_type in ('nmpc'):
        from utils.acados_cleanup import cleanup_acados_files
        cleanup_acados_files(controller.get_json_file_name())



if __name__ == '__main__':
    main()