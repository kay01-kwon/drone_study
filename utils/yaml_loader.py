import yaml
import numpy as np
import os

def load_yaml(config_path='config/pd_params.yaml'):
    """

    :param config_path:
    :return:
    """

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
    rotor_params = {
        'p': np.array(config['rotor_params']['p']),
        'w_rotor_min': config['rotor_params']['w_rotor_min'],
        'w_rotor_max': config['rotor_params']['w_rotor_max'],
        'alpha_rotor_max': config['rotor_params']['alpha_rotor_max'],
        'jerk_rotor_max': config['rotor_params']['jerk_rotor_max'],
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
    nmpc_params = {
        't_horizon': config['nmpc_params']['t_horizon'],
        'n_nodes': config['nmpc_params']['n_nodes'],
        'QArray': config['nmpc_params']['QArray'],
        'RArray': config['nmpc_params']['RArray'],
    }
    return nmpc_params

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
