from utils import yaml_loader

def load_parameters(control_type, dob_type):
    """
    Load all necessary parameters from split config files

    IMPORTANT: Separates True parameters (for simulator) from NOMINAL parameters (for control/DOB)
    - TRUE parameters: Used in simulator to represent the actual system
    - NOMINAL parameters: Used in controller and DOB (share between them)
    This separation allows testing robustness to model uncertainty.

    Args:
         control_type: Type of controller ('nmpc', 'nmpc_actuator', 'pd', or 'pd_actuator')
         dob_type: Type of DOB ('none', 'hgdo', or 'l1')

    Five steps - 1. Simulation
                 2. Control
                 3. DOB
                 4. Parameter estimation
                 5. Control estimation

    Returns:
        Dictionary containing all loaded parameters with keys:
        - true_*: True parameters for simulator
        - nominal_*: Nominal parameters for controller and DOB
        - sim_params, trajectory_params, control-specific params, dob_params
    """
    params = {}

    # 1. Load True parameters from simulator config (used for simulation model)
    config_sim = yaml_loader.load_yaml('config/simulator/simulator.yaml')
    params['true_dynamic_params'] = yaml_loader.get_dynamic_params(config_sim)
    params['true_drone_params'] = yaml_loader.get_drone_params(config_sim)
    params['true_rotor_params'] = yaml_loader.get_rotor_params(config_sim)
    params['sim_params'] = yaml_loader.get_sim_params(config_sim)

    # 2. Load NOMINAL and control related parameters from control config (used for controller AND DOB)
    if control_type == 'nmpc':
        config_control = yaml_loader.load_yaml('config/control/nmpc/nmpc_params.yaml')
        params['nmpc_params'] = yaml_loader.get_nmpc_params(config_control)

        # Load nominal dynamic params from control config
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)

    elif control_type == 'nmpc_actuator':
        config_control = yaml_loader.load_yaml('config/control/nmpc/nmpc_actuator.yaml')
        params['nmpc_params'] = yaml_loader.get_nmpc_params(config_control)

        # Load nominal dynamic params from control config
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)

        # Load allocator weight params from separate config
        config_allocator = yaml_loader.load_yaml('config/control/allocator/allocator.yaml')
        params['allocator_params'] = yaml_loader.get_allocator_params(config_allocator)

    elif control_type == "pd":
        config_control = yaml_loader.load_yaml('config/control/pd/pd_params.yaml')
        params['gain_params'] = yaml_loader.get_pd_gain_params(config_control)

        # Load nominal dynamic params from control config
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)

    elif control_type == "pd_actuator":
        config_control = yaml_loader.load_yaml('config/control/pd/pd_params.yaml')
        params['gain_params'] = yaml_loader.get_pd_gain_params(config_control)

        # Load nominal dynamic params from control config
        params['nominal_dynamic_params'] = yaml_loader.get_dynamic_params(config_control)
        params['nominal_drone_params'] = yaml_loader.get_drone_params(config_control)
        params['nominal_rotor_params'] = yaml_loader.get_rotor_params(config_control)

        # Load allocator weight params from separate config
        config_allocator = yaml_loader.load_yaml('config/control/allocator/allocator.yaml')
        params['allocator_params'] = yaml_loader.get_allocator_params(config_allocator)

    # 3. Load DOB-specific parameters (DOB will use nominal params from control config)
    if dob_type == "hgdo":
        config_dob = yaml_loader.load_yaml('config/estimator/dob/hgdo.yaml')
        params['dob_params'] = yaml_loader.get_hgdo_params(config_dob)
    elif dob_type == "l1":
        config_dob = yaml_loader.load_yaml('config/estimator/dob/l1_adaptive.yaml')
        params['dob_params'] = yaml_loader.get_l1_adaptation_params(config_dob)

    # 4. Load RLS parameters for dynamic parameter estimator
    config_rls = yaml_loader.load_yaml('config/estimator/rls/rls_param.yaml')
    params['rls_params'] = yaml_loader.get_rls_parameters(config_rls)

    # 5. Load trajectory and regulation parameters
    config_traj = yaml_loader.load_yaml('config/trajectory/trajectory.yaml')
    params['trajectory_params'] = yaml_loader.get_trajectory_params(config_traj)
    params['regulation_params'] = yaml_loader.get_regulation_params(config_traj)

    return params


def setup_controller(control_type, dob_type, params):
    """
    Setup the appropriate controller and disturbance observer

    IMPORTANT: Both controller and DOB use NOMINAL parameters (not true parameters)
    This ensures they share the same system model.

    Args:
        control_type: Type of controller ('nmpc_comp','nmpc_param', or 'pd')
        dob_type: Type of DOB ('none', 'hgdo', or 'l1')
        params: Dictionary of parameters (must contain nominal_* parameters)

    Returns:
        controller: The control object (uses nominal parameters)
        dob: Disturbance observer (uses nominal parameters, or None)
    """
    dob = None

    # Setup controller using NOMINAL parameters
    if control_type == 'nmpc' or control_type == 'nmpc_actuator':
        from control.nmpc.ocp.S550_ocp import S550Ocp

        controller = S550Ocp(DynParam=params['nominal_dynamic_params'],
                                   DroneParam=params['nominal_drone_params'],
                                   MpcParam=params['nmpc_params'])

    elif control_type == 'pd' or control_type == 'pd_actuator':
        from control.PID.geometric_control import GeometricControl

        # DobMode = True if dob_type is not 'none'
        use_dob = (dob_type != 'none')
        controller = GeometricControl(DynamicParams=params['nominal_dynamic_params'],
                                      GainParams=params['gain_params'],
                                      DobMode=use_dob)
    else:
        raise ValueError(f"Unknown control type: {control_type}")

    # Setup disturbance observer using NOMINAL parameters (same as controller)
    if dob_type == 'hgdo':
        from estimator.dob.hgdo.hgdo import HGDO

        dob = HGDO(DynParam=params['nominal_dynamic_params'],
                   DroneParam=params['nominal_drone_params'],
                   RotorParam=params['nominal_rotor_params'],
                   DobParam=params['dob_params'])

    elif dob_type == 'l1':
        from estimator.dob.l1_adaptation.l1_adaptation import L1Adaptation

        dob = L1Adaptation(DynParam=params['nominal_dynamic_params'],
                           DroneParam=params['nominal_drone_params'],
                           RotorParam=params['nominal_rotor_params'],
                           DobParam=params['dob_params'])

    elif dob_type == 'none':
        dob = None
    else:
        raise ValueError(f"Unknown DOB type: {dob_type}")

    return controller, dob