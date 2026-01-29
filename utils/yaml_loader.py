import yaml
import numpy as np
import os

def load_yaml(config_path):

    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_path = os.path.join(project_root, config_path)

    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def get_dynamic_params(config):
    # Convert lists to numpy arrays where needed
    dynamic_params = {
        'm': config['dynamic_params']['m'],
        'MoiArray': np.array(config['dynamic_params']['MoiArray']),
        'com_offset': np.array(config['dynamic_params']['com_offset'])
    }
    return dynamic_params

def get_drone_params(config):
    drone_params = {
        'arm_length': config['drone_params']['arm_length'],
        'motor_const': config['drone_params']['motor_const'],
        'moment_const': config['drone_params']['moment_const']
    }
    return drone_params

def get_rotor_params(config):
    p_array = config['rotor_params']['p']
    rotor_params = {
        'p': np.array(p_array),
        'p1': p_array[0],
        'p2': p_array[1],
        'p3': p_array[2],
        'w_rotor_min': config['rotor_params']['w_rotor_min'],
        'w_rotor_max': config['rotor_params']['w_rotor_max'],
        'alpha_rotor_max': config['rotor_params']['alpha_rotor_max'],
        'alpha_max': config['rotor_params']['alpha_rotor_max'],  # alias for allocator
        'jerk_rotor_max': config['rotor_params']['jerk_rotor_max'],
        'j_max': config['rotor_params']['jerk_rotor_max'],  # alias for allocator
        'num_rotors': config['rotor_params']['num_rotors']
    }
    return rotor_params

def get_pd_gain_params(config):
    gain_params = {
        'KpPosArray': np.array(config['gain_params']['KpPosArray']),
        'KdPosArray': np.array(config['gain_params']['KdPosArray']),
        'KiPosArray': np.array(config['gain_params']['KiPosArray']),
        'ITransLimit': np.array(config['gain_params']['ITransLimit']),
        'KpOriArray': np.array(config['gain_params']['KpOriArray']),
        'KdOriArray': np.array(config['gain_params']['KdOriArray']),
        'KiOriArray': np.array(config['gain_params']['KiOriArray']),
        'IOriLimit': np.array(config['gain_params']['IOriLimit']),
        'AccelMax': np.array(config['gain_params']['AccelMax'])
    }
    return gain_params

def get_nmpc_params(config):
    # Handle both 'R' and 'RArray' keys for backwards compatibility
    if 'R' in config['nmpc_params']:
        r_value = float(config['nmpc_params']['R'])
    else:
        r_value = float(config['nmpc_params']['RArray'])

    nmpc_params = {
        't_horizon': config['nmpc_params']['t_horizon'],
        'n_nodes': config['nmpc_params']['n_nodes'],
        'QArray': [float(q) for q in config['nmpc_params']['QArray']],
        'R': r_value,
    }
    # Optional actuator state weights (single scalar each)
    if 'Q_w_rot' in config['nmpc_params']:
        nmpc_params['Q_w_rot'] = float(config['nmpc_params']['Q_w_rot'])
    if 'Q_alpha_rot' in config['nmpc_params']:
        nmpc_params['Q_alpha_rot'] = float(config['nmpc_params']['Q_alpha_rot'])
    return nmpc_params

def get_allocator_params(config):
    allocator_params = {
        'Q': np.array([float(q) for q in config['allocator_params']['Q']]),
        'R': float(config['allocator_params']['R']),
    }
    return allocator_params

def get_trajectory_params(config):
    trajectory_params = {
        'max_velocity': config['trajectory']['max_velocity'],
        'max_yaw_rate': config['trajectory']['max_yaw_rate'],
        'time_scale': config['trajectory']['time_scale']
    }
    return trajectory_params

def get_regulation_params(config):
    regulation_params = {
        'setpoint_position': np.array(config['regulation']['setpoint_position']),
        'setpoint_yaw': config['regulation']['setpoint_yaw']
    }
    return regulation_params

def get_sim_params(config):
    sim_params = {
        'w_rotor_idle': config['simulation']['w_rotor_idle'],
        'tf': config['simulation']['tf'],
        'dt': config['simulation']['dt']
    }
    return sim_params

def get_hgdo_params(config):
    hgdo_params = {
        'eps_f': config['dob']['eps_f'],
        'eps_tau': config['dob']['eps_tau'],
        'lin_vel_cutoff': config['lpf']['lin_vel_cutoff'],
        'ang_vel_cutoff': config['lpf']['ang_vel_cutoff'],
        'disturbance_force_cutoff': config['lpf']['disturbance_force_cutoff'],
        'disturbance_torque_cutoff': config['lpf']['disturbance_torque_cutoff']
    }
    return hgdo_params

def get_l1_adaptation_params(config):
    l1_adaptation_params = {
        'As_array': config['l1_adaptive']['As_array'],
        'freq_cutoff_trans': config['l1_adaptive']['freq_cutoff_trans'],
        'freq_cutoff_rot': config['l1_adaptive']['freq_cutoff_rot'],
        'lin_vel_cutoff': config['lpf']['lin_vel_cutoff'],
        'ang_vel_cutoff': config['lpf']['ang_vel_cutoff']
    }
    return l1_adaptation_params

def get_rls_parameters(config):
    rls_params = {'P_m': float(config['rls']['m']['P_m']),
                  'R_m': float(config['rls']['m']['R_m']),
                  'P_com': float(config['rls']['com']['P_com']),
                  'R_com': float(config['rls']['com']['R_com'])}

    # Add optional forgetting factors if present
    if 'lambda_m' in config['rls']['m']:
        rls_params['lambda_m'] = float(config['rls']['m']['lambda_m'])
    if 'lambda_com' in config['rls']['com']:
        rls_params['lambda_com'] = float(config['rls']['com']['lambda_com'])

    return rls_params