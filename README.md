# Unified Drone Simulation Main Program

## Overview

Now all functionality is **unified into a single `main_control.py`** with:
- **Separate control and DOB selection** via command-line arguments
- **Modular config file system** for easy parameter management
- **Flexible combinations** (e.g., NMPC + no DOB, PD + HGDO, PD + L1, etc.)

## Usage

```bash
# NMPC trajectory tracking without DOB (default)
python3 main_control.py --control nmpc --dob none

# PD control with HGDO (High Gain Disturbance Observer)
python3 main_control.py --control pd --dob hgdo

# PD control with L1 Adaptation
python3 main_control.py --control pd --dob l1

# PD control without DOB
python3 main_control.py --control pd --dob none

# NMPC_param with DOB (unusual but supported)
python3 main_control.py --control nmpc --dob hgdo
```

### Arguments

- `--control {nmpc, pd}`: Control method
  - `nmpc`: Nonlinear Model Predictive Control (Direct compensation with DOB)
  - `pd`: Geometric Control

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
Loads all necessary parameters from split config files with TRUE/NOMINAL separation:
- Loads **TRUE** parameters from `simulator.yaml` for simulation
- Loads **NOMINAL** parameters from control config for controller & DOB
- Loads trajectory parameters and DOB-specific tuning parameters
- Returns dictionary with `true_*` and `nominal_*` parameter sets

### `setup_controller(control_type, dob_type, params)`
Creates the appropriate controller and disturbance observer using **NOMINAL** parameters:
- Initializes controller with `nominal_dynamic_params`
- Initializes DOB with the **same** `nominal_dynamic_params` (shared model)
- Returns controller and dob objects (both using identical nominal model)

### `plot_results(control_type, dob_type, ...)`
Generates standardized plots with control/DOB info in title.

### `print_statistics(control_type, dob_type, ...)`
Prints performance statistics including control and DOB types.

## Configuration Files

Config files are now **modularly organized** in subdirectories:

### TRUE Parameters (Simulator)
- `config/simulator/simulator.yaml` - **TRUE** dynamic parameters for actual system simulation
  - Dynamic parameters (mass, inertia, COM offset)
  - Drone parameters (arm length, motor constants)
  - Rotor parameters (model, limits)
  - Simulation settings (time step, duration)

### NOMINAL Parameters (Control & DOB)
- `config/control/nmpc/nmpc_params.yaml` - **NOMINAL** parameters for NMPC controller
  - Dynamic, drone, and rotor parameters (controller's model)
  - NMPC-specific parameters (horizon, nodes, weights)
- `config/control/pd/pd_params.yaml` - **NOMINAL** parameters for PD controller
  - Dynamic, drone, and rotor parameters (controller's model)
  - PD gains (Kp, Kd, Ki)

### Disturbance Observer Parameters
- `config/estimator/dob/hgdo.yaml` - High Gain DOB parameters (uses nominal params from control)
- `config/estimator/dob/l1_adaptive.yaml` - L1 Adaptive Control parameters (uses nominal params from control)

### Trajectory Parameters
- `config/trajectory/trajectory.yaml` - Trajectory generation parameters

## TRUE vs NOMINAL Parameters

**CRITICAL DISTINCTION:**

### TRUE Parameters (Simulator)
- **Used by:** `S550_Sim_Model`, `S550_Param_Model`, `RotorModel`, `HexaConverter`
- **Purpose:** Represent the **actual physical system** being simulated
- **Source:** `config/simulator/simulator.yaml`
- **Can differ from nominal** to test robustness to model uncertainty

### NOMINAL Parameters (Controller & DOB)
- **Used by:** Controller (NMPC/PD) **AND** DOB (HGDO/L1)
- **Purpose:** Represent the **controller's model** of the system
- **Source:** `config/control/{nmpc,pd}/*.yaml`
- **MUST be identical** for both controller and DOB (shared model)

### Why This Matters

This separation allows testing:
- **Model uncertainty** - When true ≠ nominal parameters
- **Robustness** - Controller performance under parameter mismatch
- **DOB effectiveness** - Ability to compensate for modeling errors

### Parameter Loading Logic
1. Load **TRUE** parameters from `simulator.yaml` → used for simulation models
2. Load **NOMINAL** parameters from control config → used for controller **AND** DOB
3. Load DOB-specific tuning parameters (cutoff frequencies, gains, etc.)
4. Controller and DOB share the same nominal model (consistency guarantee)

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

# To do list

- [ ] Control reallocation : minimize collective thrust and moment s.t. actuator model