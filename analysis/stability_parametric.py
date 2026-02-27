"""
Parametric stability analysis for S550 hexarotor.

Find feasible parameter combinations for stability.
"""
import numpy as np

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


def check_stability(theta_crit_deg, f_ratio, h_eff, x_g=0.0):
    """
    Check stability condition.
    Returns: (is_stable, M_y_max, RHS, margin, u2_feasible)
    """
    theta_crit = np.deg2rad(theta_crit_deg)
    f_total = f_ratio * mg

    # RHS of stability condition
    term1 = mg * x_g * np.cos(theta_crit)
    term2 = x_g * f_total
    term3 = mg * h_eff * np.sin(theta_crit)
    RHS = np.abs(term1 - (term2 + term3))

    # Check thrust allocation feasibility
    u_sum = f_total / 2
    u2_required = u_sum - u_min - u_max
    u2_feasible = (u_min <= u2_required <= u_max)

    if u2_feasible:
        M_y_max = l_eff * (u_max - u_min)
    else:
        # Constrained case
        if u2_required < u_min:
            u3_max = u_sum - 2 * u_min
            u3_max = min(u3_max, u_max)
            M_y_max = l_eff * (u3_max - u_min)
        else:  # u2_required > u_max
            M_y_max = l_eff * (u_max - u_min)

    margin = M_y_max - RHS
    is_stable = margin >= 0

    return is_stable, M_y_max, RHS, margin, u2_feasible


# ==========================
# Analysis 1: f_ratio sweep (fixed theta=20, h_eff=360mm)
# ==========================
print("=" * 70)
print("Analysis 1: f_ratio sweep (theta_crit=20 deg, h_eff=360 mm)")
print("=" * 70)
print(f"{'f_ratio':>8} | {'f [N]':>8} | {'M_y_max':>10} | {'RHS':>10} | {'Margin':>10} | {'u2 OK':>6} | {'Status':>10}")
print("-" * 70)

for f_ratio in np.arange(1.0, 1.51, 0.05):
    stable, My_max, RHS, margin, u2_ok = check_stability(20.0, f_ratio, 0.36)
    status = "STABLE" if stable else "UNSTABLE"
    u2_str = "Yes" if u2_ok else "No"
    print(f"{f_ratio:>8.2f} | {f_ratio*mg:>8.2f} | {My_max:>10.4f} | {RHS:>10.4f} | {margin:>10.4f} | {u2_str:>6} | {status:>10}")


# ==========================
# Analysis 2: theta_crit sweep (fixed f=1.2mg, h_eff=360mm)
# ==========================
print()
print("=" * 70)
print("Analysis 2: theta_crit sweep (f=1.2*mg, h_eff=360 mm)")
print("=" * 70)
print(f"{'theta':>8} | {'sin(th)':>8} | {'M_y_max':>10} | {'RHS':>10} | {'Margin':>10} | {'Status':>10}")
print("-" * 70)

for theta in np.arange(10, 31, 2):
    stable, My_max, RHS, margin, _ = check_stability(theta, 1.2, 0.36)
    status = "STABLE" if stable else "UNSTABLE"
    print(f"{theta:>8.0f} | {np.sin(np.deg2rad(theta)):>8.4f} | {My_max:>10.4f} | {RHS:>10.4f} | {margin:>10.4f} | {status:>10}")


# ==========================
# Analysis 3: h_eff sweep (fixed theta=20, f=1.2mg)
# ==========================
print()
print("=" * 70)
print("Analysis 3: h_eff sweep (theta_crit=20 deg, f=1.2*mg)")
print("=" * 70)
print(f"{'h_eff [mm]':>10} | {'M_y_max':>10} | {'RHS':>10} | {'Margin':>10} | {'Status':>10}")
print("-" * 70)

for h_eff_mm in np.arange(200, 420, 20):
    h_eff = h_eff_mm / 1000
    stable, My_max, RHS, margin, _ = check_stability(20.0, 1.2, h_eff)
    status = "STABLE" if stable else "UNSTABLE"
    print(f"{h_eff_mm:>10.0f} | {My_max:>10.4f} | {RHS:>10.4f} | {margin:>10.4f} | {status:>10}")


# ==========================
# Analysis 4: 2D sweep (theta vs f_ratio) with h_eff=360mm
# ==========================
print()
print("=" * 70)
print("Analysis 4: Stability Map (theta_crit vs f_ratio), h_eff=360 mm")
print("=" * 70)

theta_range = np.arange(10, 26, 2)
f_ratio_range = np.arange(1.0, 1.41, 0.05)

# Header
print(f"{'':>8}", end="")
for f_ratio in f_ratio_range:
    print(f" {f_ratio:>5.2f}", end="")
print()
print("-" * (8 + len(f_ratio_range) * 6))

for theta in theta_range:
    print(f"{theta:>6.0f}° ", end="")
    for f_ratio in f_ratio_range:
        stable, _, _, margin, _ = check_stability(theta, f_ratio, 0.36)
        if stable:
            print(f"   ✓ ", end="")
        else:
            print(f"   ✗ ", end="")
    print()


# ==========================
# Find boundary conditions
# ==========================
print()
print("=" * 70)
print("Boundary Analysis: Find critical parameters")
print("=" * 70)

# Find max f_ratio for theta=20, h_eff=360
for f_ratio in np.arange(1.0, 1.5, 0.01):
    stable, _, _, _, _ = check_stability(20.0, f_ratio, 0.36)
    if not stable:
        print(f"theta=20°, h_eff=360mm: max f_ratio = {f_ratio-0.01:.2f}")
        break

# Find max theta for f=1.2mg, h_eff=360
for theta in np.arange(10, 30, 1):
    stable, _, _, _, _ = check_stability(theta, 1.2, 0.36)
    if not stable:
        print(f"f=1.2*mg, h_eff=360mm: max theta = {theta-1:.0f}°")
        break

# Find max h_eff for theta=20, f=1.2mg
for h_eff_mm in np.arange(100, 500, 5):
    stable, _, _, _, _ = check_stability(20.0, 1.2, h_eff_mm/1000)
    if not stable:
        print(f"theta=20°, f=1.2*mg: max h_eff = {h_eff_mm-5:.0f} mm")
        break


# ==========================
# Recommended parameter set
# ==========================
print()
print("=" * 70)
print("Recommended Parameter Sets (all stable)")
print("=" * 70)

recommendations = [
    ("Conservative", 15, 1.2, 360),
    ("Moderate", 20, 1.1, 360),
    ("Lower h_eff", 20, 1.2, 320),
    ("Balanced", 18, 1.15, 340),
]

print(f"{'Name':>15} | {'theta':>6} | {'f_ratio':>8} | {'h_eff':>8} | {'Margin':>10} | {'Status':>10}")
print("-" * 70)

for name, theta, f_ratio, h_eff_mm in recommendations:
    stable, My_max, RHS, margin, _ = check_stability(theta, f_ratio, h_eff_mm/1000)
    status = "STABLE" if stable else "UNSTABLE"
    print(f"{name:>15} | {theta:>5}° | {f_ratio:>8.2f} | {h_eff_mm:>6} mm | {margin:>10.4f} | {status:>10}")
