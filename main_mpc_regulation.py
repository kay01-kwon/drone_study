import numpy as np

from sim_model.S550_model import S550_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.math_tool import quaternion_to_euler
from utils import yaml_loader
from utils.reference_packer import reference_packer
from nmpc.ocp.S550_simple_ocp import S550SimpleOcp
from custom_ode import custom_rk4

from matplotlib import pyplot as plt

def state_initialize(w_rotor_idle):
    """
    Initialize drone state and rotor state
    Start with small initial position offset to test regulation
    """
    p = np.array([0.1, 0.1, 0.0])  # Small offset from origin
    v = np.zeros((3,))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros((3,))
    s_drone = np.concatenate([p, v, q, w])

    w_rotor = w_rotor_idle * np.ones((6,))
    alpha_rotor = np.zeros((6,))
    s_rotor = np.concatenate([w_rotor, alpha_rotor])

    return s_drone, s_rotor


def main():

    # Load parameters from YAML file for regulation control
    config = yaml_loader.load_yaml('config/nmpc_regulation_params.yaml')
    dynamic_params = yaml_loader.get_dynamic_params(config)
    drone_params = yaml_loader.get_drone_params(config)
    rotor_params = yaml_loader.get_rotor_params(config)
    nmpc_params = yaml_loader.get_nmpc_params(config)
    regulation_params = yaml_loader.get_regulation_params(config)
    sim_params = yaml_loader.get_sim_params(config)

    # Create simulation models and NMPC controller
    drone_sim_model = S550_Sim_Model(DynamicParams=dynamic_params)
    rotor_sim_model = RotorModel(RotorParams=rotor_params)
    hexa_converter = HexaConverter(DroneParams=drone_params,
                                   RotorParams=rotor_params)
    nmpc_control = S550SimpleOcp(DynParam=dynamic_params,
                                 DroneParam=drone_params,
                                 MpcParam=nmpc_params)

    # State initialization
    w_rotor_idle = sim_params['w_rotor_idle']
    s_drone, s_rotor = state_initialize(w_rotor_idle)

    # Fixed regulation setpoint
    p_setpoint = regulation_params['setpoint_position']
    v_setpoint = np.zeros(3)  # Zero velocity for regulation
    yaw_setpoint = regulation_params['setpoint_yaw']
    yaw_dot_setpoint = 0.0  # Zero yaw rate for regulation

    # Pack setpoint into reference format
    ref_setpoint = reference_packer(p_setpoint, v_setpoint,
                                   np.array([yaw_setpoint]),
                                   np.array([yaw_dot_setpoint]))

    print("\n========== Regulation Control Test ==========")
    print(f"Setpoint position: {p_setpoint}")
    print(f"Setpoint yaw: {np.rad2deg(yaw_setpoint):.2f}°")
    print(f"Initial position: {s_drone[0:3]}")
    print("============================================\n")

    # Simulation parameters
    tf = sim_params['tf']
    dt = sim_params['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # ========== Data Storage ==========
    pos_hist = []
    vel_hist = []
    roll_hist = []
    pitch_hist = []
    yaw_hist = []
    w_rotor_hist = []
    alpha_rotor_hist = []
    pos_error_hist = []

    for i in range(N-1):

        # Unpack current state
        p, v, q, w = drone_sim_model.unpack_state(s_drone)
        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)
        roll, pitch, yaw = quaternion_to_euler(q)

        # State feedback for NMPC (p, v, q, w)
        s_feedback = np.concatenate([p, v, q, w])

        # Calculate position error
        pos_error = np.linalg.norm(p - p_setpoint)

        # Append history
        pos_hist.append(p.copy())
        vel_hist.append(v.copy())
        roll_hist.append(np.rad2deg(roll))
        pitch_hist.append(np.rad2deg(pitch))
        yaw_hist.append(np.rad2deg(yaw))
        w_rotor_hist.append(w_rotor.copy())
        alpha_rotor_hist.append(alpha_rotor.copy())
        pos_error_hist.append(pos_error)

        # NMPC regulation control with fixed setpoint
        status, w_cmd = nmpc_control.solve(s_feedback, ref_setpoint)

        # Check solver status
        if status != 0 and i % 100 == 0:
            print(f"Warning: NMPC solver status {status} at t={t_sim[i]:.2f}s")

        t_ode = [t_sim[i], t_sim[i+1]]

        # Simulate rotor dynamics
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)

        # Convert rotor speeds to control input
        u = hexa_converter.compute_u(s_rotor[0:6])

        # Simulate drone dynamics
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u, t_ode)

    # ========== Post-processing ==========
    pos_hist = np.array(pos_hist)
    vel_hist = np.array(vel_hist)
    roll_hist = np.array(roll_hist)
    pitch_hist = np.array(pitch_hist)
    yaw_hist = np.array(yaw_hist)
    w_rotor_hist = np.array(w_rotor_hist)
    alpha_rotor_hist = np.array(alpha_rotor_hist)
    pos_error_hist = np.array(pos_error_hist)
    t_plot = t_sim[:-1]

    # Calculate settling time (when error < 5% of initial error)
    settling_threshold = 0.05 * pos_error_hist[0]
    settling_indices = np.where(pos_error_hist < settling_threshold)[0]
    settling_time = t_plot[settling_indices[0]] if len(settling_indices) > 0 else None

    # ========== Plotting ==========
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    # Plot Position X
    axs[0, 0].plot(t_plot, pos_hist[:, 0], 'b-', label='Actual', linewidth=2)
    axs[0, 0].axhline(y=p_setpoint[0], color='r', linestyle='--', label='Setpoint', linewidth=2)
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('X Position [m]')
    axs[0, 0].set_title('X Position Regulation')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Position Y
    axs[0, 1].plot(t_plot, pos_hist[:, 1], 'b-', label='Actual', linewidth=2)
    axs[0, 1].axhline(y=p_setpoint[1], color='r', linestyle='--', label='Setpoint', linewidth=2)
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Y Position [m]')
    axs[0, 1].set_title('Y Position Regulation')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Position Z
    axs[0, 2].plot(t_plot, pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    axs[0, 2].axhline(y=p_setpoint[2], color='r', linestyle='--', label='Setpoint', linewidth=2)
    axs[0, 2].set_xlabel('Time [s]')
    axs[0, 2].set_ylabel('Z Position [m]')
    axs[0, 2].set_title('Z Position Regulation')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Plot Velocity
    axs[1, 0].plot(t_plot, vel_hist[:, 0], 'r-', label='vx', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 1], 'g-', label='vy', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 2], 'b-', label='vz', linewidth=2)
    axs[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Velocity [m/s]')
    axs[1, 0].set_title('Velocity (should converge to 0)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Attitude Angles
    axs[1, 1].plot(t_plot, roll_hist, 'r-', label='Roll', linewidth=2)
    axs[1, 1].plot(t_plot, pitch_hist, 'g-', label='Pitch', linewidth=2)
    axs[1, 1].plot(t_plot, yaw_hist, 'b-', label='Yaw', linewidth=2)
    axs[1, 1].axhline(y=np.rad2deg(yaw_setpoint), color='b', linestyle='--', alpha=0.5, label='Yaw Setpoint')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Angle [deg]')
    axs[1, 1].set_title('Attitude Angles')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # # Position Error (3D norm)
    # axs[1, 2].plot(t_plot, pos_error_hist, 'b-', linewidth=2)
    # axs[1, 2].axhline(y=settling_threshold, color='r', linestyle='--', alpha=0.5, label='Settling threshold')
    # if settling_time is not None:
    #     axs[1, 2].axvline(x=settling_time, color='g', linestyle='--', alpha=0.5, label=f'Settling time: {settling_time:.2f}s')
    # axs[1, 2].set_xlabel('Time [s]')
    # axs[1, 2].set_ylabel('Position Error [m]')
    # axs[1, 2].set_title('Position Error Norm')
    # axs[1, 2].legend()
    # axs[1, 2].grid(True)
    # axs[1, 2].set_yscale('log')

    # 3D Trajectory
    ax_3d = fig.add_subplot(3, 3, 6, projection='3d')
    ax_3d.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    ax_3d.scatter(p_setpoint[0], p_setpoint[1], p_setpoint[2], c='r', marker='*', s=200, label='Setpoint')
    ax_3d.scatter(pos_hist[0, 0], pos_hist[0, 1], pos_hist[0, 2], c='g', marker='o', s=100, label='Start')
    ax_3d.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2], c='orange', marker='x', s=100, label='End')
    ax_3d.set_xlabel('X [m]')
    ax_3d.set_ylabel('Y [m]')
    ax_3d.set_zlabel('Z [m]')
    ax_3d.set_title('3D Position (Regulation)')
    ax_3d.legend()
    ax_3d.grid(True)

    # Plot Rotor Speeds
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 0], label='Rotor 1')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 1], label='Rotor 2')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 2], label='Rotor 3')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 3], label='Rotor 4')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 4], label='Rotor 5')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 5], label='Rotor 6')
    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 0].set_ylabel('Rotor Speed [RPM]')
    axs[2, 0].set_title('Rotor Speeds')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Plot Rotor Accelerations
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 0], label='Rotor 1')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 1], label='Rotor 2')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 2], label='Rotor 3')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 3], label='Rotor 4')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 4], label='Rotor 5')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 5], label='Rotor 6')
    axs[2, 1].set_xlabel('Time [s]')
    axs[2, 1].set_ylabel('Rotor Acceleration [RPM/s]')
    axs[2, 1].set_title('Rotor Accelerations')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # Position error per axis
    pos_error_per_axis = np.abs(pos_hist - p_setpoint)
    axs[2, 2].plot(t_plot, pos_error_per_axis[:, 0], 'r-', label='X error', linewidth=2)
    axs[2, 2].plot(t_plot, pos_error_per_axis[:, 1], 'g-', label='Y error', linewidth=2)
    axs[2, 2].plot(t_plot, pos_error_per_axis[:, 2], 'b-', label='Z error', linewidth=2)
    axs[2, 2].set_xlabel('Time [s]')
    axs[2, 2].set_ylabel('Position Error per Axis [m]')
    axs[2, 2].set_title('Position Error by Axis')
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print final statistics
    print("\n========== Regulation Results ==========")
    print(f"Setpoint: [{p_setpoint[0]:.3f}, {p_setpoint[1]:.3f}, {p_setpoint[2]:.3f}] m")
    print(f"Final position: [{pos_hist[-1, 0]:.3f}, {pos_hist[-1, 1]:.3f}, {pos_hist[-1, 2]:.3f}] m")
    print(f"Final position error: {pos_error_hist[-1]:.6f} m")
    print(f"Mean position error: {np.mean(pos_error_hist):.6f} m")
    print(f"Max position error: {np.max(pos_error_hist):.6f} m")
    if settling_time is not None:
        print(f"Settling time (5% criterion): {settling_time:.2f} s")
    else:
        print("Settling time: Did not settle within simulation time")
    print()
    print(f"Final velocity: [{vel_hist[-1, 0]:.6f}, {vel_hist[-1, 1]:.6f}, {vel_hist[-1, 2]:.6f}] m/s")
    print(f"Final velocity magnitude: {np.linalg.norm(vel_hist[-1]):.6f} m/s")
    print()
    print(f"Final yaw: {yaw_hist[-1]:.2f}°, Setpoint: {np.rad2deg(yaw_setpoint):.2f}°")
    print(f"Final yaw error: {abs(yaw_hist[-1] - np.rad2deg(yaw_setpoint)):.4f}°")
    print("========================================\n")

if __name__ == '__main__':
    main()
