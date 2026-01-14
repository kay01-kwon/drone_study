# drone_study

## Overview

Previously, there were 4 separate main files with duplicate code:
- `main_mpc.py` - NMPC trajectory tracking
- `main_mpc_regulation.py` - NMPC regulation control
- `main_pd_hgdo.py` - PD control with HGDO
- `main_pd_l1_adaptation.py` - PD control with L1 Adaptation

Now all functionality is **unified into a single `main.py`** that allows you to select the control method via command-line arguments.

## Usage

```bash
# NMPC trajectory tracking (default)
python3 main.py --control nmpc

# NMPC regulation control
python3 main.py --control nmpc_regulation

# PD control with HGDO (High Gain Disturbance Observer)
python3 main.py --control pd_hgdo

# PD control with L1 Adaptation
python3 main.py --control pd_l1
```

## Architecture

The unified main program:

1. **Eliminates code duplication** - Common code (simulation loop, plotting, statistics) is now in shared functions
2. **Modular controller setup** - Each controller is initialized based on the `--control` argument
3. **Consistent interface** - All control methods use the same simulation framework
4. **Easy to extend** - Adding new control methods only requires updating the `setup_controller()` function

## Key Functions

### `state_initialize(w_rotor_idle, initial_offset=None)`
Initializes drone and rotor states. Supports optional initial position offset for regulation tests.

### `setup_controller(control_type, ...)`
Creates the appropriate controller object based on the selected control type.

### `compute_control(control_type, controller, ...)`
Computes control input using the selected controller.

### `plot_results(...)`
Generates standardized plots for all control methods.

### `print_statistics(...)`
Prints performance statistics at the end of simulation.

## Configuration Files

Each control method uses its own YAML configuration file:

- `nmpc` → `config/nmpc_params.yaml`
- `nmpc_regulation` → `config/nmpc_regulation_params.yaml`
- `pd_hgdo` → `config/pd_params.yaml` + `config/hgdo.yaml`
- `pd_l1` → `config/pd_params.yaml` + `config/l1_adaptive.yaml`

## Benefits of Unified Architecture

1. **Maintainability** - Bug fixes and improvements only need to be made once
2. **Consistency** - All simulations use identical plotting and data collection
3. **Flexibility** - Easy to compare different control methods
4. **Reduced code** - ~1000+ lines of duplicate code eliminated

## Old Files

The original main files are preserved for reference:
- `main_mpc.py`
- `main_mpc_regulation.py`
- `main_pd_hgdo.py`
- `main_pd_l1_adaptation.py`

These can be removed once the unified version is fully validated.