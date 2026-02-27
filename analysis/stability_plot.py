"""
Stability region plot for S550 hexarotor.
Plot stability condition as function of f_ratio and theta_crit.
"""
import numpy as np
import matplotlib.pyplot as plt

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
l_eff = l * np.sqrt(3)
u_min = C_T * w_min**2
u_max = C_T * w_max**2

# Maximum moment (constant for this drone)
M_y_max = l_eff * (u_max - u_min)

# ==========================
# h_eff parameter
# ==========================
h_eff = 0.36  # [m]

# ==========================
# Create grid
# ==========================
theta_deg = np.linspace(5, 35, 200)
f_ratio = np.linspace(0.8, 1.8, 200)

THETA, F_RATIO = np.meshgrid(theta_deg, f_ratio)
THETA_RAD = np.deg2rad(THETA)

# ==========================
# Compute stability condition (x_g = 0 case)
# ==========================
# RHS = mg * h_eff * sin(theta)
RHS = mg * h_eff * np.sin(THETA_RAD)

# Margin = M_y_max - RHS
MARGIN = M_y_max - RHS

# Stability region: MARGIN >= 0
STABLE = MARGIN >= 0

# ==========================
# Plot
# ==========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Stability region (binary)
ax1 = axes[0]
contour1 = ax1.contourf(THETA, F_RATIO, STABLE.astype(float),
                         levels=[0, 0.5, 1], colors=['#ffcccc', '#ccffcc'], alpha=0.8)
ax1.contour(THETA, F_RATIO, STABLE.astype(float), levels=[0.5], colors='k', linewidths=2)

# Mark the user's original point
ax1.plot(20, 1.2, 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2,
         label=f'Original (θ=20°, f=1.2mg)')

# Mark boundary at theta=18
ax1.axvline(x=18, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='θ=18° boundary')

ax1.set_xlabel('θ_crit [deg]', fontsize=12)
ax1.set_ylabel('f / mg', fontsize=12)
ax1.set_title(f'Stability Region (h_eff = {h_eff*1000:.0f} mm, x_g = 0)', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([5, 35])
ax1.set_ylim([0.8, 1.8])

# Add text annotations
ax1.text(10, 1.5, 'STABLE', fontsize=14, fontweight='bold', color='green')
ax1.text(28, 1.5, 'UNSTABLE', fontsize=14, fontweight='bold', color='red')

# Plot 2: Margin contour
ax2 = axes[1]
levels = np.linspace(-2, 2, 21)
contour2 = ax2.contourf(THETA, F_RATIO, MARGIN, levels=levels, cmap='RdYlGn')
ax2.contour(THETA, F_RATIO, MARGIN, levels=[0], colors='k', linewidths=2)
cbar = plt.colorbar(contour2, ax=ax2)
cbar.set_label('Margin [N·m]', fontsize=11)

# Mark the user's original point
ax2.plot(20, 1.2, 'ko', markersize=12, markerfacecolor='white', markeredgewidth=2,
         label=f'Original (θ=20°, f=1.2mg)')

ax2.set_xlabel('θ_crit [deg]', fontsize=12)
ax2.set_ylabel('f / mg', fontsize=12)
ax2.set_title(f'Stability Margin (h_eff = {h_eff*1000:.0f} mm)', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([5, 35])
ax2.set_ylim([0.8, 1.8])

plt.tight_layout()
plt.savefig('analysis/stability_region.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis/stability_region.pdf', bbox_inches='tight')
print("Saved: analysis/stability_region.png, analysis/stability_region.pdf")

# ==========================
# Additional: h_eff sensitivity plot
# ==========================
fig2, ax3 = plt.subplots(figsize=(8, 6))

h_eff_values = [0.28, 0.32, 0.36, 0.40]
colors = ['green', 'blue', 'orange', 'red']

for h_eff_i, color in zip(h_eff_values, colors):
    RHS_i = mg * h_eff_i * np.sin(THETA_RAD)
    MARGIN_i = M_y_max - RHS_i
    ax3.contour(THETA, F_RATIO, MARGIN_i, levels=[0], colors=color, linewidths=2,
                label=f'h_eff = {h_eff_i*1000:.0f} mm')

ax3.plot(20, 1.2, 'ko', markersize=12, markerfacecolor='red', markeredgewidth=2,
         label='Target (θ=20°, f=1.2mg)')

ax3.set_xlabel('θ_crit [deg]', fontsize=12)
ax3.set_ylabel('f / mg', fontsize=12)
ax3.set_title('Stability Boundaries for Different h_eff', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([5, 35])
ax3.set_ylim([0.8, 1.8])

# Shade stable region for h_eff = 320mm
h_eff_320 = 0.32
RHS_320 = mg * h_eff_320 * np.sin(THETA_RAD)
STABLE_320 = (M_y_max - RHS_320) >= 0
ax3.contourf(THETA, F_RATIO, STABLE_320.astype(float), levels=[0.5, 1],
             colors=['lightgreen'], alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/stability_heff_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: analysis/stability_heff_comparison.png")

# ==========================
# Print key findings
# ==========================
print("\n" + "="*60)
print("Key Findings")
print("="*60)
print(f"M_y_max = {M_y_max:.4f} N·m (constant)")
print(f"h_eff = {h_eff*1000:.0f} mm")
print()
print("Stability condition (x_g = 0):")
print("  M_y_max >= mg * h_eff * sin(θ)")
print(f"  {M_y_max:.4f} >= {mg:.2f} * {h_eff:.3f} * sin(θ)")
print()

# Critical theta for given h_eff
theta_crit_max = np.rad2deg(np.arcsin(M_y_max / (mg * h_eff)))
print(f"Critical theta (h_eff={h_eff*1000:.0f}mm): θ_max = {theta_crit_max:.1f}°")

# For theta = 20, what h_eff is needed?
h_eff_needed = M_y_max / (mg * np.sin(np.deg2rad(20)))
print(f"For θ=20°: max h_eff = {h_eff_needed*1000:.1f} mm")

plt.show()
