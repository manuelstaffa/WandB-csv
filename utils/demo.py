#!/usr/bin/env python3
"""
Demo script for WandB CSV to Graph Converter
Showcases different features of the tool
"""

import subprocess
import sys
import time
from pathlib import Path

def run_demo(description, command):
    """Run a demo command with description"""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print("Running...")
    
    try:
        result = subprocess.run(command, timeout=60)
        if result.returncode == 0:
            print("✓ Success!")
        else:
            print("✗ Failed!")
        time.sleep(2)
    except subprocess.TimeoutExpired:
        print("✗ Timed out!")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    """Run demonstration of different features"""
    csv_file = "wandb_export_2025-09-19T15_36_24.891+02_00.csv"
    
    if not Path(csv_file).exists():
        print(f"Demo CSV file not found: {csv_file}")
        return
    
    print("WandB CSV to Graph Converter - Feature Demonstration")
    print("This demo will create several graphs showcasing different features")
    
    base_cmd = [sys.executable, "main.py", "--csv-path", csv_file, "--atari-game", "kangaroo", "--resolution-dpi", "200"]
    
    # Demo 1: Basic grouping with default settings
    run_demo(
        "Basic Grouping with Default Settings",
        base_cmd + ["--title", "Demo 1: Default Grouping"]
    )
    
    # Demo 2: Individual runs (no grouping)
    run_demo(
        "Individual Runs (No Grouping)",
        base_cmd + ["--no-group-runs", "--title", "Demo 2: Individual Runs"]
    )
    
    # Demo 3: High smoothing
    run_demo(
        "High Smoothing (EMA = 0.98)",
        base_cmd + ["--ema-smoothing", "0.98", "--title", "Demo 3: Heavy Smoothing"]
    )
    
    # Demo 4: Low smoothing
    run_demo(
        "Low Smoothing (EMA = 0.7)",
        base_cmd + ["--ema-smoothing", "0.7", "--title", "Demo 4: Light Smoothing"]
    )
    
    # Demo 5: Custom grouping - combine reward functions
    run_demo(
        "Custom Grouping (RF13 + RF14)",
        base_cmd + ["--custom-groups", "rf13", "rf14", "--title", "Demo 5: Combined RF13 & RF14"]
    )
    
    # Demo 6: No envelope, high opacity individual runs
    run_demo(
        "No Envelope with Visible Individual Runs",
        base_cmd + ["--no-show-envelope", "--opacity", "0.8", "--title", "Demo 6: High Opacity Runs"]
    )
    
    # Demo 7: Custom labels and styling
    run_demo(
        "Custom Labels and Styling",
        base_cmd + [
            "--x-axis-label", "Training Episodes",
            "--y-axis-label", "Average Score",
            "--title", "Demo 7: Kangaroo Performance Analysis",
            "--opacity", "0.2"
        ]
    )
    
    # Demo 8: Using color configuration
    run_demo(
        "Custom Color Configuration",
        base_cmd + [
            "--config-file", "config/colors.toml",
            "--title", "Demo 8: Custom Colors"
        ]
    )
    
    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("All generated graphs are saved in the 'output' directory.")
    print("Each graph demonstrates different features of the tool:")
    print("- Grouping vs individual runs")
    print("- Different smoothing levels")  
    print("- Custom grouping")
    print("- Envelope visualization")
    print("- Custom labels and styling")
    print("- Color configuration")

if __name__ == "__main__":
    main()
