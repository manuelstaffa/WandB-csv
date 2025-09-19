#!/usr/bin/env python3
"""
Batch processing utility for WandB CSV to Graph Converter
Process multiple CSV files or create multiple variations of the same data
"""

import subprocess
import sys
from pathlib import Path
import argparse

def process_batch(csv_file, game_name, variations):
    """Process a CSV file with different parameter variations"""
    base_cmd = [
        sys.executable, "main.py", 
        "--csv-path", csv_file,
        "--atari-game", game_name
    ]
    
    for i, variation in enumerate(variations, 1):
        print(f"Processing variation {i}/{len(variations)}: {variation['name']}")
        
        cmd = base_cmd + variation['params']
        
        try:
            result = subprocess.run(cmd, timeout=120)
            if result.returncode == 0:
                print(f"✓ Successfully created: {variation['name']}")
            else:
                print(f"✗ Failed to create: {variation['name']}")
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout creating: {variation['name']}")
        except Exception as e:
            print(f"✗ Error creating {variation['name']}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch process WandB CSV files")
    parser.add_argument("csv_file", help="Path to CSV file")
    parser.add_argument("game_name", help="Atari game name")
    parser.add_argument("--preset", choices=["analysis", "publication", "comparison"], 
                       default="analysis", help="Preset variation type")
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return 1
    
    # Define preset variations
    presets = {
        "analysis": [
            {
                "name": "Default Grouping",
                "params": ["--group", "default", "--title", "Training Analysis - Grouped Runs"]
            },
            {
                "name": "Individual Runs", 
                "params": ["--group", "none", "--title", "Training Analysis - Individual Runs"]
            },
            {
                "name": "High Smoothing",
                "params": ["--ema-smoothing", "0.95", "--title", "Training Analysis - Smoothed"]
            },
            {
                "name": "Envelope Only",
                "params": ["--no-show-original-when-grouped", "--title", "Training Analysis - Envelope Only"]
            }
        ],
        
        "publication": [
            {
                "name": "Publication Figure",
                "params": [
                    "--resolution-dpi", "600",
                    "--ema-smoothing", "0.95", 
                    "--line-weight", "2.5",
                    "--title", f"{args.game_name.title()} Training Performance",
                    "--x-axis-label", "Training Episodes",
                    "--y-axis-label", "Average Episodic Reward"
                ]
            },
            {
                "name": "Publication Figure (Envelope Only)",
                "params": [
                    "--resolution-dpi", "600",
                    "--ema-smoothing", "0.95",
                    "--line-weight", "3.0",
                    "--no-show-original-when-grouped",
                    "--title", f"{args.game_name.title()} Training Performance (Clean)"
                ]
            }
        ],
        
        "comparison": [
            {
                "name": "Baseline vs Reward Functions",
                "params": [
                    "--group", "custom", 
                    "--group-config", "config/groups.toml",
                    "--title", "Baseline Performance"
                ]
            },
            {
                "name": "Early Reward Functions",
                "params": [
                    "--group", "custom",
                    "--group-config", "config/advanced_groups.toml", 
                    "--title", "Early Reward Functions"
                ]
            },
            {
                "name": "Late Reward Functions", 
                "params": [
                    "--group", "custom",
                    "--group-config", "config/advanced_groups.toml",
                    "--title", "Advanced Reward Functions"
                ]
            }
        ]
    }
    
    variations = presets[args.preset]
    print(f"Processing {len(variations)} variations using '{args.preset}' preset")
    
    process_batch(args.csv_file, args.game_name, variations)
    
    print("\nBatch processing complete!")
    print("Check the 'output' directory for all generated graphs.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
