"""
Test script to compare original minimum snap vs QP-based minimum snap
"""
import numpy as np
import matplotlib.pyplot as plt
from ref_generation.minimum_snap import create_trajectory_from_waypoints as create_traj_original
from ref_generation.minimum_snap_qp import create_trajectory_from_waypoints as create_traj_qp

# Define waypoints (same as in ref_generator.py)
WAYPOINTS = [
    [0.0, 0.0, 0.0],      # Start at origin
    [0.0, 0.0, 1.0],      # Takeoff to 1m
    [0.5, 0.5, 1.5],      # Move to [0.5, 0.5, 1.5]
    [0.5, -0.5, 1.0],     # Move to [0.5, -0.5, 1.0]
    [0.0, 0.0, 1.0],      # Return to hover position
]

# Create trajectories with both methods
print("Creating trajectories...")
print("1. Original method (5th order polynomial)")
traj_original = create_traj_original(WAYPOINTS, max_velocity=0.3, time_scale=5.0)

print("2. QP method (7th order polynomial with snap minimization)")
traj_qp = create_traj_qp(WAYPOINTS, max_velocity=0.3, time_scale=5.0)

# Sample both trajectories
t_total = traj_original.times[-1]
dt = 0.01
times = np.arange(0, t_total + dt, dt)

# Storage for trajectory data
pos_original = []
vel_original = []
acc_original = []
snap_original = []

pos_qp = []
vel_qp = []
acc_qp = []
snap_qp = []

print(f"\nSampling trajectories over {t_total:.2f} seconds...")
for t in times:
    # Original trajectory
    p_o, v_o, a_o, _, _ = traj_original.get_state(t)
    pos_original.append(p_o)
    vel_original.append(v_o)
    acc_original.append(a_o)

    # QP trajectory
    p_q, v_q, a_q, _, _ = traj_qp.get_state(t)
    pos_qp.append(p_q)
    vel_qp.append(v_q)
    acc_qp.append(a_q)

# Convert to arrays
pos_original = np.array(pos_original)
vel_original = np.array(vel_original)
acc_original = np.array(acc_original)

pos_qp = np.array(pos_qp)
vel_qp = np.array(vel_qp)
acc_qp = np.array(acc_qp)

# Compute snap numerically (d^4 position / dt^4)
snap_original = np.gradient(np.gradient(acc_original, dt, axis=0), dt, axis=0)
snap_qp = np.gradient(np.gradient(acc_qp, dt, axis=0), dt, axis=0)

# Compute snap magnitude
snap_mag_original = np.linalg.norm(snap_original, axis=1)
snap_mag_qp = np.linalg.norm(snap_qp, axis=1)

# Compute total snap cost (integral of squared snap)
snap_cost_original = np.sum(snap_mag_original**2) * dt
snap_cost_qp = np.sum(snap_mag_qp**2) * dt

print(f"\n{'='*60}")
print(f"SNAP COST COMPARISON")
print(f"{'='*60}")
print(f"Original method: {snap_cost_original:.4f}")
print(f"QP method:       {snap_cost_qp:.4f}")
print(f"Reduction:       {(1 - snap_cost_qp/snap_cost_original)*100:.2f}%")
print(f"{'='*60}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Position
ax = axes[0, 0]
ax.plot(times, pos_original[:, 0], 'r-', label='Original X', alpha=0.7)
ax.plot(times, pos_original[:, 1], 'g-', label='Original Y', alpha=0.7)
ax.plot(times, pos_original[:, 2], 'b-', label='Original Z', alpha=0.7)
ax.plot(times, pos_qp[:, 0], 'r--', label='QP X', linewidth=2)
ax.plot(times, pos_qp[:, 1], 'g--', label='QP Y', linewidth=2)
ax.plot(times, pos_qp[:, 2], 'b--', label='QP Z', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [m]')
ax.set_title('Position Comparison')
ax.legend()
ax.grid(True)

# Velocity
ax = axes[0, 1]
vel_mag_original = np.linalg.norm(vel_original, axis=1)
vel_mag_qp = np.linalg.norm(vel_qp, axis=1)
ax.plot(times, vel_mag_original, 'b-', label='Original', alpha=0.7)
ax.plot(times, vel_mag_qp, 'r--', label='QP', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocity Magnitude [m/s]')
ax.set_title('Velocity Comparison')
ax.legend()
ax.grid(True)

# Acceleration
ax = axes[1, 0]
acc_mag_original = np.linalg.norm(acc_original, axis=1)
acc_mag_qp = np.linalg.norm(acc_qp, axis=1)
ax.plot(times, acc_mag_original, 'b-', label='Original', alpha=0.7)
ax.plot(times, acc_mag_qp, 'r--', label='QP', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Acceleration Magnitude [m/s²]')
ax.set_title('Acceleration Comparison')
ax.legend()
ax.grid(True)

# Snap
ax = axes[1, 1]
ax.plot(times, snap_mag_original, 'b-', label=f'Original (Cost={snap_cost_original:.2f})', alpha=0.7)
ax.plot(times, snap_mag_qp, 'r--', label=f'QP (Cost={snap_cost_qp:.2f})', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Snap Magnitude [m/s⁴]')
ax.set_title('Snap Comparison (Lower is Better)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('trajectory_comparison.png', dpi=150)
print(f"\nPlot saved to: trajectory_comparison.png")

# 3D trajectory plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(pos_original[:, 0], pos_original[:, 1], pos_original[:, 2],
        'b-', label='Original', alpha=0.6, linewidth=2)
ax.plot(pos_qp[:, 0], pos_qp[:, 1], pos_qp[:, 2],
        'r--', label='QP', linewidth=2)

# Plot waypoints
waypoints = np.array(WAYPOINTS)
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
          c='black', s=100, marker='o', label='Waypoints')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Trajectory Comparison')
ax.legend()
ax.grid(True)

plt.savefig('trajectory_3d_comparison.png', dpi=150)
print(f"3D plot saved to: trajectory_3d_comparison.png")

plt.show()

print("\nTest completed!")
