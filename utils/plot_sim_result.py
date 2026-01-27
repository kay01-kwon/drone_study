import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def setup_plot_style():
    """Plot style 설정"""
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })


def plot_results(control_type, dob_type,
                 drone_state, rotor_state,
                 ref, dob_est, param_est,
                 true_dynamic_params):
    # Setup style
    setup_plot_style()

    # Get time
    t_plot = drone_state['t']

    # Get drone state
    pos = drone_state['pos']
    vel = drone_state['vel']
    roll = drone_state['roll']
    pitch = drone_state['pitch']
    yaw = drone_state['yaw']

    # Get rotor state
    w_rotor = rotor_state['w_rotor']
    alpha_rotor = rotor_state['alpha_rotor']

    # Get reference
    pos_des = ref['pos_des']
    vel_des = ref['vel_des']
    yaw_des = ref['yaw_des']

    # Get disturbance estimate
    f_est = dob_est['f_est']
    tau_est = dob_est['tau_est']

    # Get dynamic parameter estimate
    m_est = param_est['m_est']
    com_x_est = param_est['com_x_est']
    com_y_est = param_est['com_y_est']

    # Create title string
    dob_str = dob_type.upper() if dob_type != 'none' else 'No DOB'
    title_str = f'Control: {control_type.upper()} | DOB: {dob_str}'

    # Create all figures
    fig1 = create_state_figure(t_plot, pos, vel, roll, pitch, yaw,
                               pos_des, vel_des, yaw_des, title_str)
    fig2 = create_error_figure(t_plot, pos, vel, pos_des, vel_des, title_str)
    fig3_speed, fig3_accel = create_rotor_figure(t_plot, w_rotor, alpha_rotor, title_str)
    fig4 = create_disturbance_figure(t_plot, f_est, tau_est, title_str)
    fig5 = create_parameter_figure(t_plot, m_est, com_x_est, com_y_est, true_dynamic_params, title_str)
    fig6 = create_3d_trajectory_figure(pos, pos_des, title_str)  # 새로운 3D trajectory figure

    plt.show()

    return fig1, fig2, fig3_speed, fig3_accel, fig4, fig5, fig6


def create_3d_trajectory_figure(pos, pos_des, title_str):
    """Figure 6: 3D Position Trajectory Tracking"""
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'{title_str} - 3D Trajectory Tracking', fontsize=18, fontweight='bold')

    ax = fig.add_subplot(111, projection='3d')

    # Plot actual trajectory
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            'b-', linewidth=2.5, label='Actual trajectory')

    # Plot desired trajectory
    ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2],
            'r--', linewidth=2, label='Desired trajectory')

    # Mark start and end points
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2],
               c='g', marker='o', s=100, label='Start', edgecolors='k', linewidths=1.5)
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2],
               c='r', marker='s', s=100, label='End', edgecolors='k', linewidths=1.5)

    # Set labels
    ax.set_xlabel(r'$x$ [m]', fontsize=14)
    ax.set_ylabel(r'$y$ [m]', fontsize=14)
    ax.set_zlabel(r'$z$ [m]', fontsize=14)

    # Set equal aspect ratio for better visualization
    max_range = np.array([
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
        pos[:, 2].max() - pos[:, 2].min()
    ]).max() / 2.0

    mid_x = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max() + pos[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    fig.tight_layout()
    return fig


def create_state_figure(t, pos, vel, roll, pitch, yaw, pos_des, vel_des, yaw_des, title_str):
    """Figure 1: Position, Velocity, and Attitude"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle(f'{title_str} - State Tracking', fontsize=18, fontweight='bold')

    # Row 1: Position
    axes[0, 0].plot(t, pos[:, 0], 'b-', label=r'$x$', linewidth=2)
    axes[0, 0].plot(t, pos_des[:, 0], 'r--', label=r'$x_{\mathrm{des}}$', linewidth=2)
    axes[0, 0].set_ylabel(r'$x$ [m]')
    axes[0, 0].set_xlabel('time [s]')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[1, 0].plot(t, pos[:, 1], 'b-', label=r'$y$', linewidth=2)
    axes[1, 0].plot(t, pos_des[:, 1], 'r--', label=r'$y_{\mathrm{des}}$', linewidth=2)
    axes[1, 0].set_ylabel(r'$y$ [m]')
    axes[1, 0].set_xlabel('time [s]')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[2, 0].plot(t, pos[:, 2], 'b-', label=r'$z$', linewidth=2)
    axes[2, 0].plot(t, pos_des[:, 2], 'r--', label=r'$z_{\mathrm{des}}$', linewidth=2)
    axes[2, 0].set_ylabel(r'$z$ [m]')
    axes[2, 0].set_xlabel('time [s]')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Row 2: Velocity
    axes[0, 1].plot(t, vel[:, 0], 'b-', label=r'$v_x$', linewidth=2)
    axes[0, 1].plot(t, vel_des[:, 0], 'r--', label=r'$v_{x,\mathrm{des}}$', linewidth=2)
    axes[0, 1].set_ylabel(r'$v_x$ [m/s]')
    axes[0, 1].set_xlabel('time [s]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 1].plot(t, vel[:, 1], 'b-', label=r'$v_y$', linewidth=2)
    axes[1, 1].plot(t, vel_des[:, 1], 'r--', label=r'$v_{y,\mathrm{des}}$', linewidth=2)
    axes[1, 1].set_ylabel(r'$v_y$ [m/s]')
    axes[1, 1].set_xlabel('time [s]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[2, 1].plot(t, vel[:, 2], 'b-', label=r'$v_z$', linewidth=2)
    axes[2, 1].plot(t, vel_des[:, 2], 'r--', label=r'$v_{z,\mathrm{des}}$', linewidth=2)
    axes[2, 1].set_ylabel(r'$v_z$ [m/s]')
    axes[2, 1].set_xlabel('time [s]')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    # Row 3: Attitude
    axes[0, 2].plot(t, roll, 'b-', label=r'$\phi$', linewidth=2)
    axes[0, 2].set_ylabel(r'$\phi$ [rad]')
    axes[0, 2].set_xlabel('time [s]')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 2].plot(t, pitch, 'b-', label=r'$\theta$', linewidth=2)
    axes[1, 2].set_ylabel(r'$\theta$ [rad]')
    axes[1, 2].set_xlabel('time [s]')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    axes[2, 2].plot(t, yaw, 'b-', label=r'$\psi$', linewidth=2)
    axes[2, 2].plot(t, yaw_des, 'r--', label=r'$\psi_{\mathrm{des}}$', linewidth=2)
    axes[2, 2].set_ylabel(r'$\psi$ [rad]')
    axes[2, 2].set_xlabel('time [s]')
    axes[2, 2].legend()
    axes[2, 2].grid(True)

    fig.tight_layout()
    return fig


def create_error_figure(t, pos, vel, pos_des, vel_des, title_str):
    """Figure 2: Tracking Errors"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle(f'{title_str} - Tracking Errors', fontsize=18, fontweight='bold')

    # Calculate errors
    pos_error = pos - pos_des
    vel_error = vel - vel_des
    norm_pos_error = np.linalg.norm(pos_error, axis=1)

    # Position errors
    axes[0, 0].plot(t, pos_error[:, 0], 'r-', linewidth=2)
    axes[0, 0].axhline(0, color='k', linestyle=':', linewidth=1)
    axes[0, 0].set_ylabel(r'$e_x$ [m]')
    axes[0, 0].set_xlabel('time [s]')
    axes[0, 0].grid(True)

    axes[1, 0].plot(t, pos_error[:, 1], 'r-', linewidth=2)
    axes[1, 0].axhline(0, color='k', linestyle=':', linewidth=1)
    axes[1, 0].set_ylabel(r'$e_y$ [m]')
    axes[1, 0].set_xlabel('time [s]')
    axes[1, 0].grid(True)

    axes[2, 0].plot(t, pos_error[:, 2], 'r-', linewidth=2)
    axes[2, 0].axhline(0, color='k', linestyle=':', linewidth=1)
    axes[2, 0].set_ylabel(r'$e_z$ [m]')
    axes[2, 0].set_xlabel('time [s]')
    axes[2, 0].grid(True)

    # Velocity errors
    axes[0, 1].plot(t, vel_error[:, 0], 'r-', linewidth=2)
    axes[0, 1].axhline(0, color='k', linestyle=':', linewidth=1)
    axes[0, 1].set_ylabel(r'$e_{v_x}$ [m/s]')
    axes[0, 1].set_xlabel('time [s]')
    axes[0, 1].grid(True)

    axes[1, 1].plot(t, vel_error[:, 1], 'r-', linewidth=2)
    axes[1, 1].axhline(0, color='k', linestyle=':', linewidth=1)
    axes[1, 1].set_ylabel(r'$e_{v_y}$ [m/s]')
    axes[1, 1].set_xlabel('time [s]')
    axes[1, 1].grid(True)

    axes[2, 1].plot(t, vel_error[:, 2], 'r-', linewidth=2)
    axes[2, 1].axhline(0, color='k', linestyle=':', linewidth=1)
    axes[2, 1].set_ylabel(r'$e_{v_z}$ [m/s]')
    axes[2, 1].set_xlabel('time [s]')
    axes[2, 1].grid(True)

    # Norm of position error
    axes[0, 2].plot(t, norm_pos_error, 'r-', linewidth=2)
    axes[0, 2].set_ylabel(r'$\|\mathbf{e}_p\|$ [m]')
    axes[0, 2].set_xlabel('time [s]')
    axes[0, 2].grid(True)

    # Remove unused subplots
    axes[1, 2].axis('off')
    axes[2, 2].axis('off')

    fig.tight_layout()
    return fig


def create_rotor_figure(t, w_rotor, alpha_rotor, title_str):
    """Figure 3: Rotor Dynamics (각 로터 개별)"""
    n_rotors = w_rotor.shape[1]

    # Figure 3-1: Rotor Speed
    fig_speed, axes_speed = plt.subplots(3, 2, figsize=(14, 12))
    fig_speed.suptitle(f'{title_str} - Rotor Speed', fontsize=18, fontweight='bold')

    # Figure 3-2: Rotor Acceleration
    fig_accel, axes_accel = plt.subplots(3, 2, figsize=(14, 12))
    fig_accel.suptitle(f'{title_str} - Rotor Acceleration', fontsize=18, fontweight='bold')

    # Flatten axes for easier indexing
    axes_speed = axes_speed.flatten()
    axes_accel = axes_accel.flatten()

    for i in range(n_rotors):
        # Rotor speed
        axes_speed[i].plot(t, w_rotor[:, i], 'b-', linewidth=2)
        axes_speed[i].set_ylabel(rf'$\omega_{i + 1}$ [RPM]')
        axes_speed[i].set_xlabel('time [s]')
        axes_speed[i].set_title(f'Motor {i + 1}', fontsize=14)
        axes_speed[i].grid(True)

        # Rotor acceleration
        axes_accel[i].plot(t, alpha_rotor[:, i], 'r-', linewidth=2)
        axes_accel[i].set_ylabel(rf'$\alpha_{i + 1}$ [RPM/s]')
        axes_accel[i].set_xlabel('time [s]')
        axes_accel[i].set_title(f'Motor {i + 1}', fontsize=14)
        axes_accel[i].grid(True)

    fig_speed.tight_layout()
    fig_accel.tight_layout()

    return fig_speed, fig_accel


def create_disturbance_figure(t, f_est, tau_est, title_str):
    """Figure 4: DOB Disturbance Estimates"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f'{title_str} - Disturbance Estimates', fontsize=18, fontweight='bold')

    # Force estimates
    axes[0, 0].plot(t, f_est[:, 0], 'b-', linewidth=2)
    axes[0, 0].set_ylabel(r'$\hat{f}_x$ [N]')
    axes[0, 0].set_xlabel('time [s]')
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, f_est[:, 1], 'g-', linewidth=2)
    axes[0, 1].set_ylabel(r'$\hat{f}_y$ [N]')
    axes[0, 1].set_xlabel('time [s]')
    axes[0, 1].grid(True)

    axes[0, 2].plot(t, f_est[:, 2], 'r-', linewidth=2)
    axes[0, 2].set_ylabel(r'$\hat{f}_z$ [N]')
    axes[0, 2].set_xlabel('time [s]')
    axes[0, 2].grid(True)

    # Torque estimates
    axes[1, 0].plot(t, tau_est[:, 0], 'b-', linewidth=2)
    axes[1, 0].set_ylabel(r'$\hat{\tau}_x$ [Nm]')
    axes[1, 0].set_xlabel('time [s]')
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, tau_est[:, 1], 'g-', linewidth=2)
    axes[1, 1].set_ylabel(r'$\hat{\tau}_y$ [Nm]')
    axes[1, 1].set_xlabel('time [s]')
    axes[1, 1].grid(True)

    axes[1, 2].plot(t, tau_est[:, 2], 'r-', linewidth=2)
    axes[1, 2].set_ylabel(r'$\hat{\tau}_z$ [Nm]')
    axes[1, 2].set_xlabel('time [s]')
    axes[1, 2].grid(True)

    fig.tight_layout()
    return fig


def create_parameter_figure(t, m_est, com_x_est, com_y_est, true_dynamic_params, title_str):
    """Figure 5: Dynamic Parameter Estimates"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{title_str} - Parameter Estimates', fontsize=18, fontweight='bold')

    true_mass = true_dynamic_params['m']
    true_com_x = true_dynamic_params['com_offset'][0]
    true_com_y = true_dynamic_params['com_offset'][1]

    # Mass estimate
    axes[0].plot(t, m_est, 'b-', label=r'$\hat{m}$', linewidth=2)
    axes[0].axhline(y=true_mass, color='r', linestyle='--', label='True mass', linewidth=2)
    axes[0].set_ylabel(r'$\hat{m}$ [kg]')
    axes[0].set_xlabel('time [s]')
    axes[0].legend()
    axes[0].grid(True)

    # CoM x offset
    axes[1].plot(t, com_x_est, 'b-', label=r'$\hat{x}_{\mathrm{offset}}$', linewidth=2)
    axes[1].axhline(y=true_com_x, color='r', linestyle='--', label=r'$x_{\mathrm{offset,true}}$', linewidth=2)
    axes[1].set_ylabel(r'$\hat{x}_{\mathrm{CoM}}$ [m]')
    axes[1].set_xlabel('time [s]')
    axes[1].legend()
    axes[1].grid(True)

    # CoM y offset
    axes[2].plot(t, com_y_est, 'b-', label=r'$\hat{y}_{\mathrm{offset}}$', linewidth=2)
    axes[2].axhline(y=true_com_y, color='r', linestyle='--', label=r'$y_{\mathrm{offset,true}}$', linewidth=2)
    axes[2].set_ylabel(r'$\hat{y}_{\mathrm{CoM}}$ [m]')
    axes[2].set_xlabel('time [s]')
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    return fig