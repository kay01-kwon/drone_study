"""
Actuator constraint analysis for S550 hexarotor NMPC.

Diamond-shaped feasible region with:
- Lower triangle: M_y = 2*l_eff*(f - f_min)/3
- Upper triangle: M_y = l_eff*(f_max - f)/3  (used for takeoff: high thrust)

Plots:
1. f vs M_y with f_max - f constraint
2. theta vs M_y relationship
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================
# S550 Drone Parameters
# ==========================
m = 2.9           # Mass [kg]
l = 0.265         # Arm length [m]
C_T = 1.465e-7    # Motor thrust constant
w_min = 2000.0    # Min rotor speed [RPM]
w_max = 7200.0    # Max rotor speed [RPM]
g = 9.81          # Gravity [m/s^2]

mg = m * g
l_eff = l * np.sqrt(3)  # Effective arm length for hexarotor

# Per-motor thrust limits
u_min = C_T * w_min**2
u_max = C_T * w_max**2

# Total thrust limits (6 motors)
f_min = 6 * u_min
f_max = 6 * u_max

print("="*60)
print("S550 Hexarotor Actuator Parameters")
print("="*60)
print(f"Mass: m = {m} kg")
print(f"Weight: mg = {mg:.2f} N")
print(f"Arm length: l = {l*1000:.1f} mm")
print(f"Effective arm: l_eff = {l_eff*1000:.2f} mm")
print()
print(f"Per-motor thrust: u_min = {u_min:.4f} N, u_max = {u_max:.4f} N")
print(f"Total thrust: f_min = {f_min:.2f} N, f_max = {f_max:.2f} N")
print(f"f_min/mg = {f_min/mg:.3f}, f_max/mg = {f_max/mg:.3f}")
print()

# ==========================
# h_eff parameter (CoG height)
# ==========================
h_eff = 0.36  # [m]

# ==========================
# Diamond constraint lines
# ==========================
# Lower triangle (low thrust): M_y = 2*l_eff*(f - f_min)/3
# Upper triangle (high thrust): M_y = l_eff*(f_max - f)/3

# For f range
f_array = np.linspace(f_min, f_max, 500)

# Lower boundary (from f_min)
M_y_lower = 2 * l_eff * (f_array - f_min) / 3

# Upper boundary (from f_max) - using this for takeoff
M_y_upper = l_eff * (f_max - f_array) / 3

# ==========================
# Plot 1: f vs M_y with f_max - f constraint
# ==========================
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Plot diamond boundaries
ax1.plot(f_array, M_y_lower, 'b-', linewidth=2,
         label=r'Lower: $M_y = \frac{2}{3} l_{eff} (f - f_{min})$')
ax1.plot(f_array, M_y_upper, 'r-', linewidth=2,
         label=r'Upper: $M_y = \frac{1}{3} l_{eff} (f_{max} - f)$')
ax1.plot(f_array, -M_y_lower, 'b-', linewidth=2)
ax1.plot(f_array, -M_y_upper, 'r-', linewidth=2)

# Fill feasible region (diamond)
# Positive M_y side
f_fill_lower = f_array[f_array <= (f_min + f_max)/2]
f_fill_upper = f_array[f_array >= (f_min + f_max)/2]
M_y_fill_lower = 2 * l_eff * (f_fill_lower - f_min) / 3
M_y_fill_upper = l_eff * (f_max - f_fill_upper) / 3

# Create diamond polygon
diamond_f = np.concatenate([
    [f_min], f_fill_lower, f_fill_upper, [f_max],
    f_fill_upper[::-1], f_fill_lower[::-1], [f_min]
])
diamond_M = np.concatenate([
    [0], M_y_fill_lower, M_y_fill_upper, [0],
    -M_y_fill_upper[::-1], -M_y_fill_lower[::-1], [0]
])

ax1.fill(diamond_f, diamond_M, color='lightgreen', alpha=0.3, label='Feasible Region')

# Mark hover point (f = mg)
ax1.axvline(x=mg, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'Hover: f = mg = {mg:.1f} N')

# Mark f_min and f_max
ax1.axvline(x=f_min, color='blue', linestyle=':', linewidth=1, alpha=0.5)
ax1.axvline(x=f_max, color='red', linestyle=':', linewidth=1, alpha=0.5)

# Calculate M_y available at hover
M_y_at_hover_upper = l_eff * (f_max - mg) / 3
M_y_at_hover_lower = 2 * l_eff * (mg - f_min) / 3
ax1.plot(mg, M_y_at_hover_upper, 'ro', markersize=10, markeredgecolor='black',
         label=f'$M_y$ available (upper) = {M_y_at_hover_upper:.3f} N·m')
ax1.plot(mg, M_y_at_hover_lower, 'bo', markersize=10, markeredgecolor='black',
         label=f'$M_y$ available (lower) = {M_y_at_hover_lower:.3f} N·m')

ax1.set_xlabel('Total Thrust $f$ [N]', fontsize=13)
ax1.set_ylabel('Pitch Moment $M_y$ [N·m]', fontsize=13)
ax1.set_title('Hexarotor Actuator Constraints: f vs $M_y$ (Diamond Region)', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, f_max * 1.1])

# Add annotations
ax1.annotate(f'$f_{{min}}$ = {f_min:.1f} N',
             xy=(f_min, 0), xytext=(f_min + 5, 0.8),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'))
ax1.annotate(f'$f_{{max}}$ = {f_max:.1f} N',
             xy=(f_max, 0), xytext=(f_max - 20, 0.8),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('analysis/f_vs_My_diamond.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis/f_vs_My_diamond.pdf', bbox_inches='tight')
print("Saved: analysis/f_vs_My_diamond.png")

# ==========================
# Plot 2: theta vs M_y relationship
# ==========================
# Stability condition: M_y >= mg * h_eff * sin(theta)
# Required moment increases with theta

fig2, ax2 = plt.subplots(figsize=(10, 7))

theta_deg = np.linspace(0, 45, 200)
theta_rad = np.deg2rad(theta_deg)

# Required moment for stability
M_y_required = mg * h_eff * np.sin(theta_rad)

# Available M_y using upper constraint (f_max - f)
# At takeoff, f varies with theta: f = mg / cos(theta)
f_at_theta = mg / np.cos(theta_rad)

# Clip to valid range
f_clipped = np.clip(f_at_theta, f_min, f_max)

# Available M_y from upper constraint
M_y_available_upper = l_eff * (f_max - f_clipped) / 3

# Available M_y from lower constraint (for comparison)
M_y_available_lower = 2 * l_eff * (f_clipped - f_min) / 3

# Plot
ax2.plot(theta_deg, M_y_required, 'k-', linewidth=2.5,
         label=r'Required: $M_y = mg \cdot h_{eff} \cdot \sin(\theta)$')
ax2.plot(theta_deg, M_y_available_upper, 'r-', linewidth=2.5,
         label=r'Available (upper): $M_y = \frac{1}{3} l_{eff} (f_{max} - f)$')
ax2.plot(theta_deg, M_y_available_lower, 'b--', linewidth=2, alpha=0.7,
         label=r'Available (lower): $M_y = \frac{2}{3} l_{eff} (f - f_{min})$')

# Find critical theta where available = required
# Using upper constraint
diff_upper = M_y_available_upper - M_y_required
idx_cross = np.where(diff_upper[:-1] * diff_upper[1:] < 0)[0]
if len(idx_cross) > 0:
    theta_crit = theta_deg[idx_cross[0]]
    M_y_crit = M_y_required[idx_cross[0]]
    ax2.plot(theta_crit, M_y_crit, 'ro', markersize=12, markeredgecolor='black',
             markeredgewidth=2, zorder=5, label=f'Critical: $\\theta$ = {theta_crit:.1f}°')
    ax2.axvline(x=theta_crit, color='red', linestyle=':', alpha=0.5)

# Shade stable region
stable_mask = M_y_available_upper >= M_y_required
ax2.fill_between(theta_deg, M_y_required, M_y_available_upper,
                 where=stable_mask, color='lightgreen', alpha=0.4, label='Stable Margin')
ax2.fill_between(theta_deg, M_y_required, M_y_available_upper,
                 where=~stable_mask, color='lightcoral', alpha=0.4, label='Unstable')

ax2.set_xlabel(r'Pitch Angle $\theta$ [deg]', fontsize=13)
ax2.set_ylabel('Pitch Moment $M_y$ [N·m]', fontsize=13)
ax2.set_title(f'Pitch Stability: $\\theta$ vs $M_y$ ($h_{{eff}}$ = {h_eff*1000:.0f} mm, upper constraint)', fontsize=14)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 45])
ax2.set_ylim([0, max(M_y_required.max(), M_y_available_upper.max()) * 1.1])

# Add stability text
ax2.text(5, M_y_available_upper[20], 'STABLE', fontsize=14, fontweight='bold',
         color='darkgreen', ha='left')
if len(idx_cross) > 0:
    ax2.text(theta_crit + 5, M_y_required[idx_cross[0] + 20] * 0.8, 'UNSTABLE',
             fontsize=14, fontweight='bold', color='darkred', ha='left')

plt.tight_layout()
plt.savefig('analysis/theta_vs_My.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis/theta_vs_My.pdf', bbox_inches='tight')
print("Saved: analysis/theta_vs_My.png")

# ==========================
# Print key results
# ==========================
print("\n" + "="*60)
print("Key Results (using upper constraint: f_max - f)")
print("="*60)
print(f"h_eff = {h_eff*1000:.0f} mm")
print()
print("Upper constraint: M_y = l_eff * (f_max - f) / 3")
print(f"  At hover (f = mg = {mg:.1f} N):")
print(f"    M_y_available = {M_y_at_hover_upper:.4f} N·m")
print()
print("Lower constraint: M_y = 2 * l_eff * (f - f_min) / 3")
print(f"  At hover (f = mg = {mg:.1f} N):")
print(f"    M_y_available = {M_y_at_hover_lower:.4f} N·m")
print()
if len(idx_cross) > 0:
    print(f"Critical theta (upper constraint): {theta_crit:.1f}°")
    print(f"  At this angle, required M_y = {M_y_crit:.4f} N·m")

plt.show()
