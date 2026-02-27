"""
Thrust ratio analysis for S550 hexarotor.

Given x_g = 0.255/2.0 m, find required total thrust f as function of theta.
From force equilibrium: f * cos(theta) = mg
Therefore: f/mg = 1/cos(theta) = sec(theta)
"""
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# S550 Drone Parameters
# ==========================
m = 2.9           # Mass [kg]
l = 0.265         # Arm length [m]
g = 9.81          # Gravity [m/s^2]

mg = m * g

# ==========================
# Given parameters
# ==========================
x_g = 0.255 / 2.0  # [m] = 0.1275 m

# ==========================
# Theta range: 0 to 20 degrees
# ==========================
theta_deg = np.linspace(0, 20, 100)
theta_rad = np.deg2rad(theta_deg)

# ==========================
# Force equilibrium: f * cos(theta) = mg
# ==========================
# f = mg / cos(theta)
# f/mg = 1/cos(theta) = sec(theta)

f_ratio = 1.0 / np.cos(theta_rad)

# ==========================
# Plot
# ==========================
fig, ax = plt.subplots(figsize=(10, 6))

# Define y-axis limits
y_min, y_max = 0.98, 1.10

# Fill STABLE region (above the curve)
ax.fill_between(theta_deg, f_ratio, y_max,
                color='lightgreen', alpha=0.5, label='Stable')

# Fill UNSTABLE region (below the curve)
ax.fill_between(theta_deg, y_min, f_ratio,
                color='lightcoral', alpha=0.5, label='Unstable')

# Plot the equilibrium curve
ax.plot(theta_deg, f_ratio, 'b-', linewidth=2.5, label=r'Equilibrium: $f/mg = 1/\cos(\theta)$')

# Mark specific points
theta_marks = [0, 5, 10, 15, 20]
for theta in theta_marks:
    f_val = 1.0 / np.cos(np.deg2rad(theta))
    ax.plot(theta, f_val, 'ko', markersize=6)
    ax.annotate(f'{f_val:.4f}', (theta, f_val),
                textcoords='offset points', xytext=(5, 10), fontsize=10)

# Add Stable/Unstable text labels
ax.text(10, 1.075, 'STABLE', fontsize=16, fontweight='bold',
        color='darkgreen', ha='center', va='center')
ax.text(10, 1.005, 'UNSTABLE', fontsize=16, fontweight='bold',
        color='darkred', ha='center', va='center')

ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
ax.set_ylabel(r'$f / mg$', fontsize=14)
ax.set_title(r'Moment Stability: Thrust Ratio vs Pitch Angle ($x_g$ = %.4f m)' % x_g, fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, zorder=5)
ax.set_xlim([0, 20])
ax.set_ylim([y_min, y_max])

# Add horizontal line at f/mg = 1
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig('analysis/thrust_vs_theta.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis/thrust_vs_theta.pdf', bbox_inches='tight')
print("Saved: analysis/thrust_vs_theta.png, analysis/thrust_vs_theta.pdf")

# ==========================
# Print values
# ==========================
print("\n" + "="*50)
print(f"x_g = {x_g:.4f} m = {x_g*1000:.2f} mm")
print("="*50)
print(f"{'theta [deg]':>12} | {'cos(theta)':>12} | {'f/mg':>12} | {'f [N]':>12}")
print("-"*50)
for theta in [0, 5, 10, 15, 20]:
    cos_val = np.cos(np.deg2rad(theta))
    f_ratio_val = 1.0 / cos_val
    f_val = f_ratio_val * mg
    print(f"{theta:>12.0f} | {cos_val:>12.6f} | {f_ratio_val:>12.6f} | {f_val:>12.4f}")

plt.show()
