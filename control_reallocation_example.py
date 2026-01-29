import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# ==========================================
# 1. 고정 파라미터 및 한계치 설정 (From Identified Model)
# ==========================================
# 식별된 2차 동역학 파라미터
P1, P2, P3 = 25.16687, 0.003933, 515.605

# 물리적 제약 조건 (Saturation Limits)
W_MIN, W_MAX = 2000.0, 7300.0  # RPM (Bounds)
ALPHA_MAX = 15000.0  # RPM/s (State Constraint)
J_MAX = 250000.0  # RPM/s^2 (Path Constraint)

# 기체 제원
C_T = 1.465e-7  # Thrust coeff
k_m = 0.01569  # Moment coeff
l = 0.265  # Arm length
DT = 0.01  # Control loop step (s)


# ==========================================
# 2. 물리 모델 및 예측 함수
# ==========================================
def get_jerk_vec(w, alpha, w_cmd):
    """Jerk: j = -(p1 + p2*w)*alpha + p3*(w_cmd - w)"""
    return -(P1 + P2 * w) * alpha + P3 * (w_cmd - w)


def compute_wrench(w_rot):
    """로터 속도로부터 6개 로터의 총 추력 및 모멘트 계산"""
    T = C_T * (w_rot ** 2)
    cp3, sp3 = np.cos(np.pi / 3), np.sin(np.pi / 3)

    f = np.sum(T)
    mx = l * ((T[0] + T[2]) * cp3 + T[1] - (T[3] + T[5]) * cp3 - T[4])
    my = l * (-(T[0] + T[5]) * sp3 + (T[2] + T[3]) * sp3)
    mz = k_m * (-T[0] + T[1] - T[2] + T[3] - T[4] + T[5])
    return np.array([f, mx, my, mz])


def soft_saturation(x, limit, stiffness = 1.0):
    return limit/stiffness * np.tanh(stiffness*x/limit)

# ==========================================
# 3. 최적 할당 클래스 (Allocator)
# ==========================================
class ActuatorAwareAllocator:
    def __init__(self, AllocParams):

        # Weight for thrust and moment along x, y, z
        self.Q = np.diag(AllocParams['Q'])

        # Regularization weight
        self.R = AllocParams['R']*np.eye(6)
        # To activate warm start
        self.last_w_cmd = np.array([5860.0] * 6)

    def allocate(self, w_curr, alpha_curr, u_des):
        """
        :param w_curr: current rotor speed (6,)
        :param alpha_curr: current rotor acceleration (6,)
        :param u_des: Desired control input (Thrust, Moment)
        :return: Optimal w_cmd (6,)
        """

        # [A] 목적 함수: 렌치 오차 최소화
        def objective(w_cmd):
            # 2차 모델 기반 다음 스텝 속도 예측
            jerk = get_jerk_vec(w_curr, alpha_curr, w_cmd)

            jerk_sat = soft_saturation(jerk, J_MAX, stiffness=1.0)

            alpha_raw = alpha_curr + jerk_sat*DT
            alpha_sat = soft_saturation(alpha_raw, ALPHA_MAX, stiffness=1.0)

            w_next = w_curr + alpha_sat * DT + 0.5 * jerk_sat * DT ** 2

            u_achieved = compute_wrench(w_next)
            # L2 norm bewteen optimal control input and desired control input
            l2_norm = (u_achieved - u_des).T @ self.Q @ (u_achieved - u_des)

            # To prevent too fast change for rotor speed
            residual_reg = (w_cmd - w_curr).T @ self.R @ (w_cmd - w_curr)
            return l2_norm + residual_reg

        # lb/ub setup: [-alpha_max, -j_max] ~ [alpha_max, j_max]
        lb = np.concatenate([[-ALPHA_MAX] * 6, [-J_MAX] * 6])
        ub = np.concatenate([[ALPHA_MAX] * 6, [J_MAX] * 6])

        # [C] 최적화 실행 (SLSQP 알고리즘 사용)
        res = minimize(
            objective,
            x0=self.last_w_cmd,
            method='SLSQP',
            bounds=[(W_MIN, W_MAX)] * 6,  # RPM 박스 제약
            options={'ftol': 1e-20, 'disp': False}
        )

        if res.success:
            self.last_w_cmd = res.x
            return res.x
        else:
            # 실패 시 안전을 위해 현재 속도 유지 또는 경고
            return self.last_w_cmd


# ==========================================
# 4. 실행 예제 (User Scenario)
# ==========================================
Q = np.array([100.0, 1.0, 1.0, 1.0])
R = 0.0

alloc_params = {'Q': Q, 'R': R}
allocator = ActuatorAwareAllocator(AllocParams=alloc_params)

# 현재 상태: 5860 RPM, 가속도 0
w_now = np.array([5860.0] * 6)
alpha_now = np.zeros(6)

# NMPC 목표: 30N 추력, Mx 0.1Nm
u_target = np.array([30.0, 0.3, 0.005, 0.0])

w_cmd_optimal = allocator.allocate(w_now, alpha_now, u_target)

print(f"--- Result of control allocation ---")
print(f"Commanded RPM:\n{w_cmd_optimal}")
print(f"\nDesired wrench: {u_target} (F, Mx, My, Mz)\n)")
# 검증을 위한 예측 결과 계산
j_pred = get_jerk_vec(w_now, alpha_now, w_cmd_optimal)
w_next_pred = w_now + alpha_now * DT + 0.5 * j_pred * DT ** 2
u_cmd = compute_wrench(w_cmd_optimal)
u_final = compute_wrench(w_next_pred)


print(f"cmd F: {u_cmd[0]:.4f}, Mx: {u_cmd[1]:.4f},"
      f"My: {u_cmd[2]:.4f}, Mz: {u_cmd[3]:.4f}\n")

print(f"Predicted F: {u_final[0]:.4f}, Mx: {u_final[1]:.4f},"
      f"My: {u_final[2]:.4f}, Mz: {u_final[3]:.4f}\n")