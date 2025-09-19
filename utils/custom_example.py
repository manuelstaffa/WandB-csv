#!/usr/bin/env python3
"""
Example of how to customize the run name extraction logic for different naming schemes.

This shows how to modify the WandBVisualizer class to handle different CSV column naming patterns.
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))
from main import WandBVisualizer, Config


class CustomWandBVisualizer(WandBVisualizer):
    """Custom visualizer with different run name extraction logic."""

    def _extract_run_info(self, column_name: str) -> Optional[Tuple[str, str]]:
        """
        Custom run name extraction for different naming schemes.

        Override this method to handle different column naming patterns.
        """

        # Example 1: Original kangaroo pattern
        # ALE/Kangaroo-v5__19-09-kangaroo-rf15__237__1758314296 - charts/...
        kangaroo_pattern = r".*?kangaroo-([^_]+)__.*"
        match = re.search(kangaroo_pattern, column_name)
        if match:
            group_name = match.group(1)
            return column_name, group_name

        # Example 2: Different environment pattern
        # ALE/Breakout-v5__experiment-type-A__run123__timestamp - metrics/...
        breakout_pattern = r".*?experiment-([^_]+)__.*"
        match = re.search(breakout_pattern, column_name)
        if match:
            group_name = match.group(1)
            return column_name, group_name

        # Example 3: Simple pattern with method names
        # run_dqn_seed1, run_ppo_seed2, run_a3c_seed3
        simple_pattern = r"run_([^_]+)_.*"
        match = re.search(simple_pattern, column_name)
        if match:
            group_name = match.group(1)
            return column_name, group_name

        # Example 4: Pattern with environment and algorithm
        # CartPole_DQN_trial1, CartPole_PPO_trial2
        env_algo_pattern = r"([^_]+)_([^_]+)_.*"
        match = re.search(env_algo_pattern, column_name)
        if match:
            env, algo = match.groups()
            group_name = f"{env}_{algo}"
            return column_name, group_name

        # Fallback: return None if no pattern matches
        return None


def main():
    """Example usage of custom visualizer."""
    import tyro

    # Use the custom visualizer instead of the default one
    config = tyro.cli(Config, description="Custom WandB Visualizer Example")

    # Create custom visualizer
    visualizer = CustomWandBVisualizer(config)

    # Process data with custom extraction logic
    runs = visualizer.parse_csv()

    if not runs:
        print("No valid runs found with custom extraction logic.")
        return

    print(f"Found {len(runs)} runs with custom extraction")

    # Show what groups were extracted
    groups = set(run.group for run in runs)
    print(f"Extracted groups: {sorted(groups)}")

    # Apply smoothing and create visualization
    smoothed_runs = visualizer.smooth_data(runs)
    visualizer.create_visualization(smoothed_runs)


if __name__ == "__main__":
    main()
