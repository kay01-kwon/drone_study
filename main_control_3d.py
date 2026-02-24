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


class ScaledTrajectory:
    """Time-scaled wrapper for HehnTrajectory.
    Stretches time by factor s: real_duration = original_duration * s.
    Position is interpolated; velocity/acceleration are scaled accordingly.
    """
    def __init__(self, traj, time_scale):
        self._traj = traj
        self._s = time_scale

    def get_position(self, t):
        return self._traj.get_position(t / self._s)

    def get_velocity(self, t):
        return self._traj.get_velocity(t / self._s) / self._s

    def get_acceleration(self, t):
        return self._traj.get_acceleration(t / self._s) / (self._s ** 2)

    @property
    def duration(self):
        return self._traj.duration * self._s


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


class TrajectoryManager:
    """Manages Hehn trajectory generation with online replanning."""

    def __init__(self, params, target_2d):
        from ref_generation.hehn_trajectory import HehnTrajectoryGenerator, QuadParams

        tp = params['tracking_params']
        self.qp = QuadParams(a_max=tp['a_max'], a_min=tp['a_min'],
                             omega_xy_max=tp['omega_xy_max'])
        self.gen = HehnTrajectoryGenerator(qp=self.qp)
        self.time_scale = tp.get('time_scale', 1.0)
        self.target = np.array([target_2d[0], 0.0, target_2d[1]])

        self.traj = None
        self.t_start = 0.0  # Time when current trajectory started
        self.last_replan_time = -np.inf  # Last absolute time when replanning occurred
        self.replan_interval = 2.0  # Minimum interval between replans (seconds)

    def generate(self, state_2d, acc_2d, t_now):
        """
        Generate new trajectory from current state to target.

        :param state_2d: [px, pz, vx, vz]
        :param acc_2d: [ax, az] current acceleration
        :param t_now: current simulation time
        """
        pos0 = np.array([state_2d[0], 0.0, state_2d[1]])
        vel0 = np.array([state_2d[2], 0.0, state_2d[3]])
        acc0 = np.array([acc_2d[0], 0.0, acc_2d[1]])

        traj_raw = self.gen.generate(pos0, vel0, acc0, target=self.target)

        if self.time_scale != 1.0:
            self.traj = ScaledTrajectory(traj_raw, self.time_scale)
        else:
            self.traj = traj_raw

        self.t_start = t_now
        self.last_replan_time = t_now

    def should_replan(self, t_now):
        """Check if replanning is needed based on time interval."""
        time_since_replan = t_now - self.last_replan_time
        t_rel = t_now - self.t_start
        # Replan if: enough time passed AND not near trajectory end
        return (time_since_replan >= self.replan_interval) and (t_rel < self.traj.duration - 0.5)

    def get_reference(self, t_now):
        """Get trajectory reference at current time (relative to trajectory start)."""
        t_rel = t_now - self.t_start
        return self.traj, t_rel

    @property
    def duration(self):
        return self.traj.duration if self.traj else 0.0


def setup_trajectory_3d(mode, params, state0):
    """Setup trajectory generator for 3DOF.
    Returns: TrajectoryManager object or None for regulation.
    """
    if mode == 'regulation':
        return None

    tp = params['tracking_params']
    target_2d = tp['target_position']

    traj_mgr = TrajectoryManager(params, target_2d)

    # Initial trajectory with acc0 = 0 (drone at rest)
    state_2d = np.array([state0[0], state0[1], state0[2], state0[3]])
    acc_2d = np.array([0.0, 0.0])
    traj_mgr.generate(state_2d, acc_2d, t_now=0.0)

    return traj_mgr


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
    plt.savefig('results_3d.png', dpi=300)
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

    # Setup trajectory for tracking mode
    s_body_init = drone_sim_model.get_state(s_drone)
    traj_mgr = setup_trajectory_3d(control_mode, params, s_body_init)

    # Parameters for acceleration estimation (for replanning)
    C_T = params['nominal_drone_params']['motor_const']
    m_nom = params['nominal_dynamic_params']['m']

    if control_mode == 'regulation':
        p_target = params['regulation_params']['setpoint_position']
        print(f"\nRegulation: target position x={p_target[0]:.2f}, z={p_target[1]:.2f} m")
    else:
        print(f"\nTracking: Hehn trajectory, duration={traj_mgr.duration:.2f}s (with replanning)")
        print(f"  Target: {params['tracking_params']['target_position']}")

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

        # Compute reference
        t_now = t_sim[i]
        if control_mode == 'regulation':
            p_des = params['regulation_params']['setpoint_position']
            v_des = np.array([0.0, 0.0])
        else:
            # Tracking: from hehn trajectory with replanning
            traj, t_rel = traj_mgr.get_reference(t_now)

            if traj_mgr.should_replan(t_now):
                # Estimate current acceleration from rotor speeds
                acc_est = estimate_acceleration(s_body, w_rotor, C_T, m_nom)
                state_2d = np.array([p[0], p[1], v_body[0], v_body[1]])
                traj_mgr.generate(state_2d, acc_est, t_now)
                traj, t_rel = traj_mgr.get_reference(t_now)
                if i % 100 == 0:
                    print(f"[Replan] t={t_now:.2f}s, pos=[{p[0]:.3f}, {p[1]:.3f}], "
                          f"new duration={traj.duration:.2f}s")

            pos_3d = traj.get_position(t_rel)
            vel_3d = traj.get_velocity(t_rel)
            p_des = np.array([pos_3d[0], pos_3d[2]])  # x, z
            v_des = np.array([vel_3d[0], vel_3d[2]])  # vx, vz

        ref = np.concatenate([p_des, v_des])

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
        if control_type == 'nmpc':
            if control_mode == 'tracking':
                traj, t_rel = traj_mgr.get_reference(t_now)
                status, w_cmd = controller.solve_for_trajectory(s_body, t_rel, traj)
            else:
                status, w_cmd = controller.solve(s_body, ref)

            if dob is not None and t_now > 2.0:
                # DOB compensation: subtract disturbance moment from NMPC output
                # Delay compensation until DOB has converged (t > 2s)
                u_mpc = hexa_converter.compute_u(w_cmd)
                u_comp = u_mpc.copy()
                u_comp[1] -= d_est[2]              # subtract tau_ext from My
                w_cmd = hexa_converter.compute_des_rotor_speed(u_comp)

            if status != 0 and i % 100 == 0:
                print(f"Warning: NMPC solver status {status} at t={t_now:.2f}s")

        elif control_type == 'pd':
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

    # Cleanup for NMPC
    if control_type == 'nmpc':
        from utils.acados_cleanup import cleanup_acados_files
        cleanup_acados_files(controller.get_json_file_name())


if __name__ == '__main__':
    main()
