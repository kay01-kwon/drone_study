import numpy as np

from sim_model.S550_model import S550_Sim_Model
from sim_model.rotor_model import RotorModel
from utils.drone_converter import HexaConverter
from utils.math_tool import quaternion_to_euler
from utils import yaml_loader
from control.PID.geometric_control import GeometricControl
from estimator.dob.hgdo.hgdo import HGDO
from ref_generation.ref_generator import get_reference, unpack_ref
from utils.custom_ode import custom_rk4

from matplotlib import pyplot as plt

def state_initialize(w_rotor_idle):
    p = np.zeros((3,))
    v = np.zeros((3,))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros((3,))
    s_drone = np.concatenate([p, v, q, w])

    w_rotor = w_rotor_idle * np.ones((6,))
    alpha_rotor = np.zeros((6,))
    s_rotor = np.concatenate([w_rotor, alpha_rotor])

    return s_drone, s_rotor


def main():

    # Load parameters from YAML file
    config = yaml_loader.load_yaml('config/pd_params.yaml')
    dynamic_params = yaml_loader.get_dynamic_params(config)
    drone_params = yaml_loader.get_drone_params(config)
    rotor_params = yaml_loader.get_rotor_params(config)
    gain_params = yaml_loader.get_pd_gain_params(config)
    trajectory_params = yaml_loader.get_trajectory_params(config)
    sim_params = yaml_loader.get_sim_params(config)

    config_dob = yaml_loader.load_yaml('config/hgdo.yaml')
    hgdo_params = yaml_loader.get_hgdo_params(config_dob)

    # Load parameters to the objects
    drone_sim_model = S550_Sim_Model(DynamicParams=dynamic_params)
    rotor_sim_model = RotorModel(RotorParams=rotor_params)
    hexa_converter = HexaConverter(DroneParams=drone_params,
                                   RotorParams=rotor_params)
    geometric_control = GeometricControl(DynamicParams=dynamic_params,
                                   GainParams=gain_params,
                                   DobMode=True)
    hgdo_obj = HGDO(DynParam=dynamic_params,
                    DroneParam=drone_params,
                    RotorParam=rotor_params,
                    DobParam=hgdo_params)

    # State initialization
    w_rotor_idle = sim_params['w_rotor_idle']
    s_drone, s_rotor = state_initialize(w_rotor_idle)

    # Simulation parameters
    tf = sim_params['tf']
    dt = sim_params['dt']
    t_sim = np.arange(0, tf, dt)
    N = len(t_sim)

    # ========== Data Storage ==========
    pos_hist = []
    vel_hist = []
    pos_des_hist = []
    roll_hist = []
    pitch_hist = []
    yaw_hist = []
    yaw_des_hist = []
    w_rotor_hist = []
    alpha_rotor_hist = []

    for i in range(N-1):

        # Velocity in body frame
        p, v, q, w = drone_sim_model.unpack_state(s_drone)
        w_rotor, alpha_rotor = rotor_sim_model.unpack_state(s_rotor)
        roll, pitch, yaw = quaternion_to_euler(q)
        # Concatenate for geometric control feedback state
        s_feedback = np.concatenate([p, v, q, w])

        # Get reference (trajectory_params passed on first call only)
        if i == 0:
            ref = get_reference(t_sim[i], trajectory_params)
        else:
            ref = get_reference(t_sim[i])
        p_des, v_des, yaw_des, yaw_des_dot = unpack_ref(ref)


        # Append history
        pos_hist.append(p.copy())
        vel_hist.append(v.copy())
        pos_des_hist.append(p_des)
        roll_hist.append(np.rad2deg(roll.copy()))
        pitch_hist.append(np.rad2deg(pitch.copy()))
        yaw_hist.append(np.rad2deg(yaw.copy()))
        yaw_des_hist.append(yaw_des)
        w_rotor_hist.append(w_rotor.copy())
        alpha_rotor_hist.append(alpha_rotor.copy())

        if i > 1:
            disturbance_estimate = hgdo_obj.dob_estimate(t_sim[i-1], t_sim[i],
                                                        s_rotor[:6], s_feedback)
            print(disturbance_estimate[3:])
            u = geometric_control.compute_u(s_feedback, ref, disturbance_estimate)
        else:
            u = geometric_control.compute_u(s_feedback, ref)


        w_cmd = hexa_converter.compute_des_rotor_speed(u)

        t_ode = [t_sim[i], t_sim[i+1]]

        # Simulate rotor first
        s_rotor = custom_rk4.do_step(rotor_sim_model.dynamics,
                                     s_rotor, w_cmd, t_ode)
        # Convert from actual rotor speed to control input
        u = hexa_converter.compute_u(s_rotor[0:6])
        # Simulate drone
        s_drone = custom_rk4.do_step(drone_sim_model.dynamics,
                                     s_drone, u, t_ode)

    # ========== Post-processing ==========
    pos_hist = np.array(pos_hist)
    vel_hist = np.array(vel_hist)
    pos_des_hist = np.array(pos_des_hist)
    roll_hist = np.array(roll_hist)
    pitch_hist = np.array(pitch_hist)
    yaw_hist = np.array(yaw_hist)
    yaw_des_hist = np.array(yaw_des_hist)
    w_rotor_hist = np.array(w_rotor_hist)
    alpha_rotor_hist = np.array(alpha_rotor_hist)
    t_plot = t_sim[:-1]

    # Calculate yaw error (handle angle wrapping)
    yaw_error_rad = yaw_hist - yaw_des_hist
    yaw_error_rad = np.arctan2(np.sin(yaw_error_rad), np.cos(yaw_error_rad))  # Wrap to [-pi, pi]
    yaw_error_deg = np.degrees(yaw_error_rad)

    # ========== Plotting ==========
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    # Plot Position X
    axs[0, 0].plot(t_plot, pos_hist[:, 0], 'b-', label='Actual', linewidth=2)
    axs[0, 0].plot(t_plot, pos_des_hist[:, 0], 'r--', label='Desired', linewidth=2)
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('X Position [m]')
    axs[0, 0].set_title('X Position Tracking')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Position Y
    axs[0, 1].plot(t_plot, pos_hist[:, 1], 'b-', label='Actual', linewidth=2)
    axs[0, 1].plot(t_plot, pos_des_hist[:, 1], 'r--', label='Desired', linewidth=2)
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Y Position [m]')
    axs[0, 1].set_title('Y Position Tracking')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Position Z
    axs[0, 2].plot(t_plot, pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    axs[0, 2].plot(t_plot, pos_des_hist[:, 2], 'r--', label='Desired', linewidth=2)
    axs[0, 2].set_xlabel('Time [s]')
    axs[0, 2].set_ylabel('Z Position [m]')
    axs[0, 2].set_title('Z Position Tracking')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Plot Velocity
    axs[1, 0].plot(t_plot, vel_hist[:, 0], 'r-', label='vx', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 1], 'g-', label='vy', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 2], 'b-', label='vz', linewidth=2)
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Velocity [m/s]')
    axs[1, 0].set_title('Velocity')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Attitude Angles
    axs[1, 1].plot(t_plot, roll_hist, 'r-', label='Roll', linewidth=2)
    axs[1, 1].plot(t_plot, pitch_hist, 'g-', label='Pitch', linewidth=2)
    axs[1, 1].plot(t_plot, yaw_hist, 'b-', label='Yaw', linewidth=2)
    axs[1, 1].plot(t_plot, np.degrees(yaw_des_hist), 'b--', label='Yaw Desired', linewidth=2)
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Angle [deg]')
    axs[1, 1].set_title('Attitude Angles')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 3D Trajectory
    ax_3d = fig.add_subplot(3, 3, 6, projection='3d')
    ax_3d.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    ax_3d.plot(pos_des_hist[:, 0], pos_des_hist[:, 1], pos_des_hist[:, 2], 'r--', label='Desired', linewidth=2)
    ax_3d.scatter(pos_hist[0, 0], pos_hist[0, 1], pos_hist[0, 2], c='g', marker='o', s=100, label='Start')
    ax_3d.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2], c='r', marker='x', s=100, label='End')
    ax_3d.set_xlabel('X [m]')
    ax_3d.set_ylabel('Y [m]')
    ax_3d.set_zlabel('Z [m]')
    ax_3d.set_title('3D Trajectory')
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

    # Position Error
    pos_error = np.linalg.norm(pos_hist - pos_des_hist, axis=1)
    axs[2, 2].plot(t_plot, pos_error, 'b-', linewidth=2)
    axs[2, 2].set_xlabel('Time [s]')
    axs[2, 2].set_ylabel('Position Error [m]')
    axs[2, 2].set_title('Position Tracking Error')
    axs[2, 2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print final statistics
    print("\n========== Simulation Results ==========")
    print(f"Final position: [{pos_hist[-1, 0]:.3f}, {pos_hist[-1, 1]:.3f}, {pos_hist[-1, 2]:.3f}] m")
    # print(f"Desired position: [{pos_des_hist[-1, 0]:.3f}, {pos_des_hist[-1, 1]:.3f}, {pos_des_hist[-1, 2]:.3f}] m")
    print(f"Final position error: {pos_error[-1]:.4f} m")
    print(f"Mean position error: {np.mean(pos_error):.4f} m")
    print(f"Max position error: {np.max(pos_error):.4f} m")
    print()
    print(f"Final yaw: {yaw_hist[-1]:.2f}°, Desired: {np.degrees(yaw_des_hist[-1]):.2f}°")
    print(f"Final yaw error: {abs(yaw_error_deg[-1]):.2f}°")
    print(f"Mean yaw error: {np.mean(np.abs(yaw_error_deg)):.2f}°")
    print(f"Max yaw error: {np.max(np.abs(yaw_error_deg)):.2f}°")
    print("========================================\n")

if __name__ == '__main__':
    main()