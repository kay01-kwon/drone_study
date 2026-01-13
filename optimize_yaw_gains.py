"""
Yaw gain optimization script

This script tests different yaw gain combinations and evaluates their performance.
"""
import numpy as np
import yaml
import subprocess
import re
from pathlib import Path

# Define test ranges for yaw gains
YAW_KP_RANGE = np.linspace(1.5, 6.0, 10)  # KpOriArray[2]
YAW_KD_RANGE = np.linspace(0.5, 3.0, 10)  # KdOriArray[2]

# Configuration file path
CONFIG_PATH = Path('config/pd_params.yaml')

def load_current_config():
    """Load current configuration from YAML"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Save configuration to YAML"""
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_simulation_and_extract_yaw_error():
    """
    Run main_pd.py and extract yaw tracking error from output

    Returns: (mean_yaw_error, max_yaw_error, final_yaw_error) in degrees
    """
    try:
        result = subprocess.run(
            ['python', 'main_pd.py'],
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout

        # Parse yaw error from output
        # Look for lines like "Yaw Error: Mean = X.XX deg, Max = X.XX deg"
        yaw_pattern = r"Yaw.*Mean.*?=\s*([0-9.]+).*Max.*?=\s*([0-9.]+)"
        yaw_match = re.search(yaw_pattern, output, re.IGNORECASE)

        if yaw_match:
            mean_yaw = float(yaw_match.group(1))
            max_yaw = float(yaw_match.group(2))

            # Try to get final yaw error
            final_pattern = r"Final.*yaw.*?([0-9.]+)"
            final_match = re.search(final_pattern, output, re.IGNORECASE)
            final_yaw = float(final_match.group(1)) if final_match else max_yaw

            return mean_yaw, max_yaw, final_yaw
        else:
            # If pattern not found, return None to indicate failure
            return None, None, None

    except subprocess.TimeoutExpired:
        print("  Simulation timed out")
        return None, None, None
    except Exception as e:
        print(f"  Error running simulation: {e}")
        return None, None, None

def main():
    """Main optimization loop"""
    print("="*60)
    print("YAW GAIN OPTIMIZATION")
    print("="*60)
    print(f"Testing {len(YAW_KP_RANGE)} x {len(YAW_KD_RANGE)} = {len(YAW_KP_RANGE) * len(YAW_KD_RANGE)} combinations")
    print()

    # Save original configuration
    original_config = load_current_config()

    # Store results
    results = []

    total_tests = len(YAW_KP_RANGE) * len(YAW_KD_RANGE)
    test_count = 0

    try:
        for kp in YAW_KP_RANGE:
            for kd in YAW_KD_RANGE:
                test_count += 1

                # Update configuration
                config = load_current_config()
                config['gain_params']['KpOriArray'][2] = float(kp)
                config['gain_params']['KdOriArray'][2] = float(kd)
                save_config(config)

                print(f"[{test_count}/{total_tests}] Testing Kp_yaw={kp:.2f}, Kd_yaw={kd:.2f}...", end=" ")

                # Run simulation
                mean_yaw, max_yaw, final_yaw = run_simulation_and_extract_yaw_error()

                if mean_yaw is not None:
                    print(f"Mean={mean_yaw:.2f}°, Max={max_yaw:.2f}°, Final={final_yaw:.2f}°")

                    results.append({
                        'kp': kp,
                        'kd': kd,
                        'mean_yaw_error': mean_yaw,
                        'max_yaw_error': max_yaw,
                        'final_yaw_error': final_yaw,
                        'score': mean_yaw + 0.5 * max_yaw + 0.3 * final_yaw  # Weighted score
                    })
                else:
                    print("Failed")

    finally:
        # Restore original configuration
        save_config(original_config)
        print("\nOriginal configuration restored")

    # Analyze results
    if results:
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)

        # Sort by score (lower is better)
        results.sort(key=lambda x: x['score'])

        print("\nTop 10 configurations:")
        print(f"{'Rank':<6} {'Kp_yaw':<8} {'Kd_yaw':<8} {'Mean°':<8} {'Max°':<8} {'Final°':<8} {'Score':<8}")
        print("-" * 60)

        for i, r in enumerate(results[:10]):
            print(f"{i+1:<6} {r['kp']:<8.2f} {r['kd']:<8.2f} {r['mean_yaw_error']:<8.2f} "
                  f"{r['max_yaw_error']:<8.2f} {r['final_yaw_error']:<8.2f} {r['score']:<8.2f}")

        # Best configuration
        best = results[0]
        print("\n" + "="*60)
        print("BEST CONFIGURATION:")
        print(f"  Kp_yaw = {best['kp']:.2f}")
        print(f"  Kd_yaw = {best['kd']:.2f}")
        print(f"  Mean yaw error = {best['mean_yaw_error']:.2f}°")
        print(f"  Max yaw error = {best['max_yaw_error']:.2f}°")
        print(f"  Final yaw error = {best['final_yaw_error']:.2f}°")
        print("="*60)

        # Ask user if they want to apply best configuration
        print("\nDo you want to apply the best configuration? (y/n): ", end="")
        choice = input().strip().lower()

        if choice == 'y':
            config = load_current_config()
            config['gain_params']['KpOriArray'][2] = float(best['kp'])
            config['gain_params']['KdOriArray'][2] = float(best['kd'])
            save_config(config)
            print("Best configuration applied!")
        else:
            print("Configuration not changed.")
    else:
        print("\nNo valid results obtained. Check simulation output.")

if __name__ == "__main__":
    main()
