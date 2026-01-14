# Unified Drone Simulation Main Program

## Overview

Previously, there were 4 separate main files with duplicate code:
- `main_mpc.py` - NMPC trajectory tracking
- `main_mpc_regulation.py` - NMPC regulation control
- `main_pd_hgdo.py` - PD control with HGDO
- `main_pd_l1_adaptation.py` - PD control with L1 Adaptation

Now all functionality is **unified into a single `main.py`** with:
- **Separate control and DOB selection** via command-line arguments
- **Modular config file system** for easy parameter management
- **Flexible combinations** (e.g., NMPC + no DOB, PD + HGDO, PD + L1, etc.)

## Usage

```bash
# NMPC trajectory tracking without DOB (default)
python3 main.py --control nmpc --dob none

# PD control with HGDO (High Gain Disturbance Observer)
python3 main.py --control pd --dob hgdo

# PD control with L1 Adaptation
python3 main.py --control pd --dob l1

# PD control without DOB
python3 main.py --control pd --dob none

# NMPC with DOB (unusual but supported)
python3 main.py --control nmpc --dob hgdo
```

### Arguments

- `--control {nmpc, pd}`: Control method
  - `nmpc`: Nonlinear Model Predictive Control
  - `pd`: PD/Geometric Control

- `--dob {none, hgdo, l1}`: Disturbance observer type
  - `none`: No disturbance observer
  - `hgdo`: High Gain Disturbance Observer
  - `l1`: L1 Adaptive Control

## Architecture

The unified main program:

1. **Eliminates code duplication** - Common code (simulation loop, plotting, statistics) is now in shared functions
2. **Modular controller setup** - Each controller is initialized based on the `--control` argument
3. **Consistent interface** - All control methods use the same simulation framework
4. **Easy to extend** - Adding new control methods only requires updating the `setup_controller()` function

## Key Functions

### `state_initialize(w_rotor_idle, initial_offset=None)`
Initializes drone and rotor states. Supports optional initial position offset.

### `load_parameters(control_type, dob_type)`
Loads all necessary parameters from split config files:
- Reads common parameters from `simulator.yaml` and `trajectory.yaml`
- Loads control-specific parameters (NMPC or PD)
- Loads DOB-specific parameters if applicable
- Returns a dictionary containing all loaded parameters

### `setup_controller(control_type, dob_type, params)`
Creates the appropriate controller and disturbance observer:
- Initializes controller based on `control_type`
- Initializes DOB based on `dob_type` (if not 'none')
- Returns controller and dob objects

### `plot_results(control_type, dob_type, ...)`
Generates standardized plots with control/DOB info in title.

### `print_statistics(control_type, dob_type, ...)`
Prints performance statistics including control and DOB types.

## Configuration Files

Config files are now **modularly organized** in subdirectories:

### Common Parameters
- `config/simulator/simulator.yaml` - Dynamic parameters, drone parameters, rotor parameters, simulation settings
- `config/trajectory/trajectory.yaml` - Trajectory generation parameters

### Control-Specific Parameters
- `config/control/nmpc/nmpc_params.yaml` - NMPC controller parameters
- `config/control/pd/pd_params.yaml` - PD/Geometric controller gains

### Disturbance Observer Parameters
- `config/estimator/dob/hgdo.yaml` - High Gain DOB parameters
- `config/estimator/dob/l1_adaptive.yaml` - L1 Adaptive Control parameters

### Parameter Loading Logic
1. Load common parameters from `simulator.yaml` and `trajectory.yaml`
2. Load control-specific parameters and override if present
3. Load DOB-specific parameters if DOB is enabled

## Benefits of Unified Architecture

### Code Organization
1. **Eliminates duplication** - ~1000+ lines of duplicate code removed
2. **Single source of truth** - Bug fixes only need to be made once
3. **Consistent interface** - All simulations use identical framework

### Flexibility
4. **Independent selection** - Control and DOB can be selected separately
5. **Easy comparison** - Test different combinations without code changes
6. **Extensible** - Adding new controllers or DOBs is straightforward

### Configuration
7. **Modular config files** - Parameters organized by function
8. **Easy tuning** - Change parameters without touching code
9. **Parameter override** - Control-specific params can override common params

## Modular Config System Benefits

The split config file structure provides:

1. **Separation of concerns**
   - Simulator parameters separate from control parameters
   - DOB parameters independent of control parameters

2. **Reusability**
   - Same simulator config can be used across all controllers
   - Trajectory parameters shared between NMPC and PD

3. **Maintainability**
   - Easy to find and modify specific parameters
   - Reduced risk of parameter conflicts

4. **Scalability**
   - Simple to add new control methods (add new file in `config/control/`)
   - Easy to add new DOB types (add new file in `config/estimator/dob/`)

## Migration from Old Structure

The old main files have been **deleted** as the unified version supersedes them:
- ~~`main_mpc.py`~~
- ~~`main_mpc_regulation.py`~~
- ~~`main_pd_hgdo.py`~~
- ~~`main_pd_l1_adaptation.py`~~

To reproduce old behavior:
- `main_mpc.py` → `python3 main.py --control nmpc --dob none`
- `main_pd_hgdo.py` → `python3 main.py --control pd --dob hgdo`
- `main_pd_l1_adaptation.py` → `python3 main.py --control pd --dob l1`
