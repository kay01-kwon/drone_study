import numpy as np
from scipy.optimize import minimize
import yaml
import os

from sim_model.S550_model import S550_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.yaml_loader import load_params
from PID.geometric_control import GeometricControl
from ref_generation.ref_generator import get_reference, unpack_ref, reset_trajectory
from custom_ode import custom_rk4


def run_simulation(gain_vector, dynamic_params, drone_params, rotor_params,
                   trajectory_params, sim_params, verbose=False):
    """
    Run simulation with given gains and return performance metric

    :param gain_vector: [KpPos_x, KpPos_y, KpPos_z, KdPos_x, KdPos_y, KdPos_z,
                         KpOri_roll, KpOri_pitch, KpOri_yaw, KdOri_roll, KdOri_pitch, KdOri_yaw]
    :return: cost (mean error + penalty)
    """

    # Reset trajectory for each simulation
    reset_trajectory()

    # Unpack gains
    KpPosArray = gain_vector[0:3]
    KdPosArray = gain_vector[3:6]
    KpOriArray = gain_vector[6:9]
    KdOriArray = gain_vector[9:12]

    # Create gain params
    gain_params = {
        'KpPosArray': KpPosArray,
        'KdPosArray': KdPosArray,
        'KiPosArray': np.array([0.001, 0.001, 0.001]),
        'ITransLimit': np.array([1.0, 1.0, 1.0]),
        'KpOriArray': KpOriArray,
        'KdOriArray': KdOriArray,
        'KiOriArray': np.array([0.01, 0.01, 0.01]),
        'IOriLimit': np.array([1.0, 1.0, 1.0]),
        'AccelMax': np.array([0.5, 0.5, 20.0])
    }

    # Initialize models
    drone_sim_model = S550_Sim_Model(DynamicParams=dynamic_params)
    rotor_sim_model = RotorModel(RotorParams=rotor_params)
    hexa_converter = HexaConverter(DroneParams=drone_params, RotorParams=rotor_params)
    geometric_control = GeometricControl(DynamicParams=dynamic_params,
                                        GainParams=gain_params,
                                        DobMode=True)

    # State initialization
    w_rotor_idle = sim_params['w_rotor_idle']
    p = np.zeros((3,))
    v = np.zeros((3,))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros((3,))
    s_drone = np.concatenate([p, v, q, w])
    w_rotor = w_rotor_idle * np.ones((6,))
    alpha_rotor = np.zeros((6,))
    s_rotor = np.concatenate([w_rotor, alpha_rotor])

    # Simulation parameters
    tf = sim_params['tf']
    dt = sim_params['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # Data storage
    pos_errors = []

    try:
        for i in range(N-1):
            # Unpack state
            p = s_drone[0:3]
            v = s_drone[3:6]
            q = s_drone[6:10]
            w_body = s_drone[10:13]
            s_feedback = np.concatenate([p, v, q, w_body])

            # Get reference
            if i == 0:
                ref = get_reference(t_sim[i], trajectory_params)
            else:
                ref = get_reference(t_sim[i])
            p_des, v_des, yaw_des, yaw_des_dot = unpack_ref(ref)

            # Calculate position error
            pos_error = np.linalg.norm(p - p_des)
            pos_errors.append(pos_error)

            # Control
            u = geometric_control.compute_u(s_feedback, ref)
            w_cmd = hexa_converter.compute_des_rotor_speed(u)

            # Simulate
            t_ode = [t_sim[i], t_sim[i+1]]
            s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics, s_rotor, w_cmd, t_ode)
            u = hexa_converter.compute_u(s_rotor[0:6])
            s_drone = custom_rk4.do_step(drone_sim_model.dynamics, s_drone, u, t_ode)

            # Check for instability
            if pos_error > 1.0 or np.any(np.isnan(s_drone)):
                return 1000.0  # Large penalty for unstable behavior

        pos_errors = np.array(pos_errors)
        mean_error = np.mean(pos_errors)
        max_error = np.max(pos_errors)

        # Cost function: mean error + penalty for large max error
        cost = mean_error + 0.2 * max_error

        if verbose:
            print(f"Mean error: {mean_error*1000:.2f}mm, Max error: {max_error*1000:.2f}mm, Cost: {cost*1000:.2f}")

        return cost

    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1000.0


def optimize_gains():
    """Optimize controller gains using scipy.optimize"""

    print("=" * 60)
    print("Gain Optimization for Geometric Controller")
    print("=" * 60)

    # Load parameters
    dynamic_params, drone_params, rotor_params, gain_params, trajectory_params, sim_params = load_params()

    # Initial gains (current values)
    initial_gains = np.array([
        2.0, 2.0, 7.0,      # KpPos
        1.5, 1.5, 5.0,      # KdPos
        10.5, 10.5, 1.5,    # KpOri
        5.0, 5.0, 0.5       # KdOri
    ])

    print("\nInitial gains:")
    print(f"  KpPos: [{initial_gains[0]:.2f}, {initial_gains[1]:.2f}, {initial_gains[2]:.2f}]")
    print(f"  KdPos: [{initial_gains[3]:.2f}, {initial_gains[4]:.2f}, {initial_gains[5]:.2f}]")
    print(f"  KpOri: [{initial_gains[6]:.2f}, {initial_gains[7]:.2f}, {initial_gains[8]:.2f}]")
    print(f"  KdOri: [{initial_gains[9]:.2f}, {initial_gains[10]:.2f}, {initial_gains[11]:.2f}]")

    # Test initial performance
    print("\nEvaluating initial performance...")
    initial_cost = run_simulation(initial_gains, dynamic_params, drone_params,
                                 rotor_params, trajectory_params, sim_params, verbose=True)

    # Define bounds
    bounds = [
        (1.0, 10.0), (1.0, 10.0), (3.0, 15.0),  # KpPos bounds
        (0.5, 8.0), (0.5, 8.0), (2.0, 10.0),    # KdPos bounds
        (5.0, 20.0), (5.0, 20.0), (0.5, 5.0),   # KpOri bounds
        (2.0, 10.0), (2.0, 10.0), (0.2, 2.0)    # KdOri bounds
    ]

    print("\n" + "=" * 60)
    print("Starting optimization (this may take several minutes)...")
    print("=" * 60)

    # Optimization callback
    iteration = [0]
    def callback(xk):
        iteration[0] += 1
        if iteration[0] % 5 == 0:
            cost = run_simulation(xk, dynamic_params, drone_params,
                                rotor_params, trajectory_params, sim_params, verbose=False)
            print(f"Iteration {iteration[0]}: Cost = {cost*1000:.2f}mm")

    # Run optimization
    result = minimize(
        fun=lambda x: run_simulation(x, dynamic_params, drone_params,
                                     rotor_params, trajectory_params, sim_params,
                                     verbose=False),
        x0=initial_gains,
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback,
        options={'maxiter': 50, 'ftol': 1e-6}
    )

    optimal_gains = result.x

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)

    print("\nOptimal gains:")
    print(f"  KpPos: [{optimal_gains[0]:.3f}, {optimal_gains[1]:.3f}, {optimal_gains[2]:.3f}]")
    print(f"  KdPos: [{optimal_gains[3]:.3f}, {optimal_gains[4]:.3f}, {optimal_gains[5]:.3f}]")
    print(f"  KpOri: [{optimal_gains[6]:.3f}, {optimal_gains[7]:.3f}, {optimal_gains[8]:.3f}]")
    print(f"  KdOri: [{optimal_gains[9]:.3f}, {optimal_gains[10]:.3f}, {optimal_gains[11]:.3f}]")

    print("\nFinal performance:")
    final_cost = run_simulation(optimal_gains, dynamic_params, drone_params,
                               rotor_params, trajectory_params, sim_params, verbose=True)

    improvement = (initial_cost - final_cost) / initial_cost * 100
    print(f"\nImprovement: {improvement:.1f}%")

    # Save optimal gains to YAML
    save_option = input("\nSave optimal gains to config/pd_params.yaml? (y/n): ")
    if save_option.lower() == 'y':
        # Load current config
        config_path = 'config/pd_params.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Update gains
        config['gain_params']['KpPosArray'] = [float(optimal_gains[0]),
                                               float(optimal_gains[1]),
                                               float(optimal_gains[2])]
        config['gain_params']['KdPosArray'] = [float(optimal_gains[3]),
                                               float(optimal_gains[4]),
                                               float(optimal_gains[5])]
        config['gain_params']['KpOriArray'] = [float(optimal_gains[6]),
                                               float(optimal_gains[7]),
                                               float(optimal_gains[8])]
        config['gain_params']['KdOriArray'] = [float(optimal_gains[9]),
                                               float(optimal_gains[10]),
                                               float(optimal_gains[11])]

        # Save to file
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)

        print(f"\nOptimal gains saved to {config_path}")

    return optimal_gains


if __name__ == '__main__':
    optimize_gains()
