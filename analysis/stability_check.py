"""
Stability condition check for S550 hexarotor with actuator.

Parameters from user:
- theta_crit = 20 deg
- f = 1.2 * mg (total thrust)
- h_eff = 360 mm

Stability condition (conservative):
|M_y| >= |mg(x_g - x_off)cos(theta_crit) - (x_g*f + mg*h_eff*sin(theta_crit))|

For hexarotor with 3 thrust groups:
M_y_max = l_eff * (f - f_min) / 3
where l_eff = l * sqrt(3)
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

# ==========================
# User-specified Parameters
# ==========================
theta_crit_deg = 20.0                    # Critical angle [deg]
theta_crit = np.deg2rad(theta_crit_deg)  # Critical angle [rad]
h_eff = 0.36                             # Effective height [m]
f_ratio = 1.2                            # f = f_ratio * mg

# ==========================
# Derived Parameters
# ==========================
mg = m * g
f_total = f_ratio * mg                   # Total thrust [N]

# Hexarotor geometry
l_eff = l * np.sqrt(3)                   # Effective moment arm [m]

# Thrust limits per group (2 motors per group)
u_min = C_T * w_min**2                   # Min thrust per motor [N]
u_max = C_T * w_max**2                   # Max thrust per motor [N]
f_min_group = 2 * u_min                  # Min thrust per group [N]
f_max_group = 2 * u_max                  # Max thrust per group [N]
f_min_total = 3 * f_min_group            # Total min thrust [N]
f_max_total = 3 * f_max_group            # Total max thrust [N]

# ==========================
# Analysis
# ==========================
print("=" * 60)
print("S550 Hexarotor Stability Analysis")
print("=" * 60)
print()
print("[ Drone Parameters ]")
print(f"  Mass (m)         : {m:.2f} kg")
print(f"  Arm length (l)   : {l*1000:.1f} mm")
print(f"  l_eff            : {l_eff*1000:.1f} mm")
print(f"  mg               : {mg:.2f} N")
print()
print("[ Thrust Limits ]")
print(f"  u_min (per motor): {u_min:.4f} N")
print(f"  u_max (per motor): {u_max:.4f} N")
print(f"  f_min (total)    : {f_min_total:.4f} N")
print(f"  f_max (total)    : {f_max_total:.4f} N")
print(f"  f_max / mg       : {f_max_total/mg:.2f}")
print()
print("[ User Parameters ]")
print(f"  theta_crit       : {theta_crit_deg:.1f} deg")
print(f"  h_eff            : {h_eff*1000:.1f} mm")
print(f"  f = {f_ratio:.1f} * mg     : {f_total:.2f} N")
print()

# ==========================
# Stability Condition Check
# ==========================
# Assume x_g = 0 (CoM at center), x_off = 0
x_g = 0.0
x_off = 0.0

# RHS of stability condition
# |mg(x_g - x_off)cos(theta_crit) - (x_g*f + mg*h_eff*sin(theta_crit))|
term1 = mg * (x_g - x_off) * np.cos(theta_crit)
term2 = x_g * f_total
term3 = mg * h_eff * np.sin(theta_crit)

RHS = np.abs(term1 - (term2 + term3))

print("[ Stability Condition Analysis ]")
print(f"  term1 = mg*(x_g - x_off)*cos(theta) = {term1:.4f} N*m")
print(f"  term2 = x_g * f                     = {term2:.4f} N*m")
print(f"  term3 = mg*h_eff*sin(theta)         = {term3:.4f} N*m")
print(f"  RHS = |term1 - (term2 + term3)|     = {RHS:.4f} N*m")
print()

# Maximum available moment for stability
# M_y_max = l_eff * (f - f_min_total) / 3
# This is when one group is at minimum thrust, others compensate
# Actually for hexarotor pitch control: M_y = l_eff * (u3 - u1)
# At max moment: u3 = (f/6 + delta), u1 = (f/6 - delta)

# More accurate: Given total f, what's the max |M_y|?
# u1 + u2 + u3 = f/2 (each group has 2 motors, so group thrust sum = f/2)
# M_y = l_eff * (u3 - u1)
# With u1 >= u_min, u3 <= u_max:
# Max M_y: u1 = u_min, u3 = u_max (if feasible)
# Then u2 = f/2 - u_min - u_max

f_per_2 = f_total / 2  # Sum of 3 group thrusts (not per motor)
# Actually: f_total = 2*(u1 + u2 + u3) where u1,u2,u3 are per-motor thrusts
# So u1 + u2 + u3 = f_total/2

u_sum = f_total / 2  # Per-motor thrust sum for 3 groups (one motor each)

# For max moment: u1 -> min, u3 -> max
# u2 = u_sum - u_min - u_max
u2_for_max = u_sum - u_min - u_max

print("[ Maximum Moment Analysis ]")
print(f"  u_sum (per motor, 3 groups) = f/2  : {u_sum:.4f} N")
print(f"  For max M_y: u1={u_min:.4f}, u3={u_max:.4f}")
print(f"  Required u2 = u_sum - u1 - u3      : {u2_for_max:.4f} N")

if u2_for_max < u_min:
    print(f"  WARNING: u2 < u_min ({u_min:.4f}), not feasible!")
    # Adjust: u2 = u_min, u3 = u_sum - u1 - u2 = u_sum - 2*u_min
    u3_adjusted = u_sum - 2*u_min
    print(f"  Adjusted: u1=u2={u_min:.4f}, u3={u3_adjusted:.4f}")
    if u3_adjusted > u_max:
        print(f"  ERROR: u3 > u_max, thrust ratio too high!")
        M_y_max = l_eff * (u_max - u_min)
    else:
        M_y_max = l_eff * (u3_adjusted - u_min)
elif u2_for_max > u_max:
    print(f"  WARNING: u2 > u_max ({u_max:.4f}), not feasible!")
    M_y_max = l_eff * (u_max - u_min)
else:
    print(f"  Feasible: u2 in [{u_min:.4f}, {u_max:.4f}]")
    M_y_max = l_eff * (u_max - u_min)

print()
print(f"  M_y_max = l_eff * (u3 - u1)        : {M_y_max:.4f} N*m")
print()

# ==========================
# Final Check
# ==========================
print("=" * 60)
print("STABILITY CHECK RESULT")
print("=" * 60)
print(f"  Required |M_y| (RHS)   : {RHS:.4f} N*m")
print(f"  Available M_y_max      : {M_y_max:.4f} N*m")
print(f"  Margin                 : {M_y_max - RHS:.4f} N*m")
print(f"  Ratio (M_y_max / RHS)  : {M_y_max / RHS:.2f}")
print()

if M_y_max >= RHS:
    print("  *** STABLE: M_y_max >= RHS ***")
else:
    print("  *** UNSTABLE: M_y_max < RHS ***")

# ==========================
# Sensitivity Analysis
# ==========================
print()
print("=" * 60)
print("Sensitivity Analysis: x_g offset")
print("=" * 60)

x_g_values = np.linspace(-0.05, 0.05, 11)

print(f"{'x_g [mm]':>10} | {'RHS [N*m]':>12} | {'Margin [N*m]':>12} | {'Status':>10}")
print("-" * 50)

for x_g_test in x_g_values:
    term1_t = mg * (x_g_test - x_off) * np.cos(theta_crit)
    term2_t = x_g_test * f_total
    term3_t = mg * h_eff * np.sin(theta_crit)
    RHS_t = np.abs(term1_t - (term2_t + term3_t))
    margin_t = M_y_max - RHS_t
    status = "STABLE" if margin_t >= 0 else "UNSTABLE"
    print(f"{x_g_test*1000:>10.1f} | {RHS_t:>12.4f} | {margin_t:>12.4f} | {status:>10}")
