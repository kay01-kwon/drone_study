import numpy as np

def reference_packer(p_des, v_des, yaw_des, yaw_des_dot):
    return np.concatenate([p_des, v_des, yaw_des, yaw_des_dot])