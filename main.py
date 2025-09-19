#!/usr/bin/env python3
"""
WandB CSV to Graph Converter

A tool to convert CSV files exported from WandB into visualized graphs with
smoothing, grouping, and customization options.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tyro
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

try:
    import toml
except ImportError:
    print("Warning: toml package not found. Config file support disabled.")
    toml = None


@dataclass
class GraphConfig:
    """Configuration for graph generation"""

    csv_path: str  # Required: Path to the CSV file
    atari_game: str  # Required: Atari game/environment name (e.g., "kangaroo")

    # Optional parameters with defaults
    ema_smoothing: float = 0.9  # EMA smoothing factor (0.0 to 1.0)
    group: str = "default"  # Grouping mode: none/default/custom
    group_config: Optional[str] = None  # Path to custom group config file
    opacity: float = 0.3  # Opacity for individual runs when grouped (0.0 to 1.0)
    show_envelope: bool = True  # Show envelope for groups
    smooth_envelope: bool = False  # Apply EMA smoothing to envelope bounds
    show_original_when_grouped: bool = True  # Show original runs when grouping
    line_weight: float = 2.0  # Line weight for graphs
    x_axis_label: str = "Steps"  # X-axis label
    y_axis_label: str = "Episodic Reward"  # Y-axis label
    title: str = ""  # Graph title (auto-generated if empty)
    show_legend: bool = True  # Whether to show legend
    resolution_dpi: int = 300  # Resolution for output image
    output_format: str = "png"  # Output format: png or svg
    output_dir: str = "output"  # Output directory
    config_file: Optional[str] = None  # Path to custom config TOML file
    custom_groups: Optional[List[str]] = (
        None  # Custom group identifiers to combine (deprecated)
    )


class WandBGrapher:
    """Main class for converting WandB CSV to graphs"""

    def __init__(self, config: GraphConfig):
        self.config = config
        self.colors = self._load_colors()
        self.df = None
        self.groups = {}

    def _load_colors(self) -> List[str]:
        """Load colors from config file or use defaults"""
        if self.config.config_file and Path(self.config.config_file).exists() and toml:
            try:
                config_data = toml.load(self.config.config_file)
                return config_data.get("colors", self._default_colors())
            except Exception as e:
                print(
                    f"Warning: Could not load config file {self.config.config_file}: {e}"
                )
                return self._default_colors()
        else:
            return self._default_colors()

    def _default_colors(self) -> List[str]:
        """Default color palette"""
        return [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d3",
            "#c7c7c7",
            "#dbdb8d",
            "#9edae5",
            "#393b79",
            "#637939",
            "#8c6d31",
            "#843c39",
            "#7b4173",
            "#5254a3",
            "#8ca252",
            "#bd9e39",
            "#ad494a",
            "#a55194",
        ]

    def load_data(self):
        """Load and preprocess the CSV data"""
        print(f"Loading data from {self.config.csv_path}")
        self.df = pd.read_csv(self.config.csv_path)

        # Filter columns to remove MIN/MAX and keep only relevant ones
        columns_to_keep = ["Step"]
        run_columns = []

        for col in self.df.columns[1:]:  # Skip 'Step' column
            if "__MIN" in col or "__MAX" in col:
                continue
            if self.config.atari_game.lower() in col.lower():
                columns_to_keep.append(col)
                run_columns.append(col)

        self.df = self.df[columns_to_keep]

        # Convert to numeric, replacing empty strings with NaN
        for col in run_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        print(f"Loaded {len(run_columns)} runs for game '{self.config.atari_game}'")
        return run_columns

    def extract_run_identifier(self, column_name: str) -> str:
        """Extract run identifier from column name"""
        # Pattern: ALE/Game-v5__date-game-identifier__number__timestamp
        pattern = rf"ALE/{re.escape(self.config.atari_game)}-v\d+__.*?-{re.escape(self.config.atari_game)}-([^_]+)__"
        match = re.search(pattern, column_name, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: look for any identifier after game name
        fallback_pattern = rf"{re.escape(self.config.atari_game)}-([^_]+)"
        match = re.search(fallback_pattern, column_name, re.IGNORECASE)
        if match:
            return match.group(1)

        return "unknown"

    def group_runs(self, run_columns: List[str]) -> Dict[str, List[str]]:
        """Group runs by their identifiers based on grouping mode"""

        if self.config.group == "none":
            # No grouping - each run is its own group
            return {col: [col] for col in run_columns}

        elif self.config.group == "custom":
            # Custom grouping from config file
            if not self.config.group_config:
                print(
                    "Warning: Custom grouping requested but no group config file provided. Using default grouping."
                )
                return self._default_grouping(run_columns)

            return self._load_custom_groups(run_columns)

        else:  # default
            # Default grouping by identifier
            return self._default_grouping(run_columns)

    def _default_grouping(self, run_columns: List[str]) -> Dict[str, List[str]]:
        """Default grouping by run identifier"""
        groups = {}

        for col in run_columns:
            identifier = self.extract_run_identifier(col)
            if identifier not in groups:
                groups[identifier] = []
            groups[identifier].append(col)

        # Handle legacy custom_groups parameter (deprecated)
        if self.config.custom_groups:
            print(
                "Warning: --custom-groups is deprecated. Use --group custom with --group-config instead."
            )
            combined_group = {}
            remaining_groups = {}

            for group_name, columns in groups.items():
                if group_name in self.config.custom_groups:
                    for col in columns:
                        if "combined" not in combined_group:
                            combined_group["combined"] = []
                        combined_group["combined"].append(col)
                else:
                    remaining_groups[group_name] = columns

            groups = {**combined_group, **remaining_groups}

        return groups

    def _load_custom_groups(self, run_columns: List[str]) -> Dict[str, List[str]]:
        """Load custom groups from config file"""
        if not toml:
            print(
                "Error: toml package required for custom grouping. Install with: pip install toml"
            )
            return self._default_grouping(run_columns)

        try:
            if self.config.group_config is None:
                print("Error: Group config path is None")
                return self._default_grouping(run_columns)

            if not Path(self.config.group_config).exists():
                print(f"Error: Group config file not found: {self.config.group_config}")
                return self._default_grouping(run_columns)

            config_data = toml.load(self.config.group_config)
            custom_groups = config_data.get("groups", {})

            if not custom_groups:
                print(
                    f"Warning: No groups found in config file {self.config.group_config}. Using default grouping."
                )
                return self._default_grouping(run_columns)

            # Create mapping from run identifier to columns
            identifier_to_columns = {}
            for col in run_columns:
                identifier = self.extract_run_identifier(col)
                if identifier not in identifier_to_columns:
                    identifier_to_columns[identifier] = []
                identifier_to_columns[identifier].append(col)

            # Build groups based on config
            result_groups = {}
            used_columns = set()

            for group_name, identifiers in custom_groups.items():
                group_columns = []
                for identifier in identifiers:
                    if identifier in identifier_to_columns:
                        group_columns.extend(identifier_to_columns[identifier])
                        used_columns.update(identifier_to_columns[identifier])

                if group_columns:
                    result_groups[group_name] = group_columns

            # Add any remaining ungrouped runs
            for col in run_columns:
                if col not in used_columns:
                    identifier = self.extract_run_identifier(col)
                    if identifier not in result_groups:
                        result_groups[identifier] = []
                    result_groups[identifier].append(col)

            print(
                f"Loaded {len(custom_groups)} custom groups from {self.config.group_config}"
            )
            return result_groups

        except Exception as e:
            print(f"Error loading custom groups from {self.config.group_config}: {e}")
            return self._default_grouping(run_columns)

    def apply_ema_smoothing(self, data: pd.Series) -> pd.Series:
        """Apply Exponential Moving Average smoothing"""
        if self.config.ema_smoothing == 0:
            return data

        smoothed = data.copy()
        alpha = 1 - self.config.ema_smoothing

        for i in range(1, len(smoothed)):
            if not pd.isna(smoothed.iloc[i]) and not pd.isna(smoothed.iloc[i - 1]):
                smoothed.iloc[i] = (
                    alpha * smoothed.iloc[i] + (1 - alpha) * smoothed.iloc[i - 1]
                )

        return smoothed

    def create_graph(self):
        """Create the main graph"""
        run_columns = self.load_data()
        groups = self.group_runs(run_columns)

        # Determine if we're doing grouping (more than one run per group)
        is_grouping = any(len(columns) > 1 for columns in groups.values())

        # Set up the plot
        plt.figure(figsize=(12, 8))

        color_idx = 0

        for group_name, columns in groups.items():
            color = self.colors[color_idx % len(self.colors)]
            color_idx += 1

            all_data = []
            steps = None

            for col in columns:
                if self.df is None:
                    continue

                data = self.df[["Step", col]].dropna()
                if len(data) == 0:
                    continue

                # Apply smoothing
                smoothed_data = self.apply_ema_smoothing(data[col])
                steps = np.array(data["Step"].values)
                smoothed_values = np.array(smoothed_data.values)

                if is_grouping and len(columns) > 1:
                    # We're grouping and this group has multiple runs
                    if self.config.show_original_when_grouped:
                        # Plot individual runs with low opacity
                        plt.plot(
                            steps,
                            smoothed_values,
                            color=color,
                            alpha=self.config.opacity,
                            linewidth=self.config.line_weight * 0.7,
                        )
                    all_data.append(smoothed_values)
                else:
                    # Single run or no grouping - plot normally
                    plt.plot(
                        steps,
                        smoothed_values,
                        color=color,
                        label=group_name,
                        linewidth=self.config.line_weight,
                    )

            # Plot group mean and envelope if we're grouping and have multiple runs
            if is_grouping and len(columns) > 1 and all_data and steps is not None:
                # Align all data to the same length (use shortest)
                min_length = min(len(d) for d in all_data)
                aligned_data = np.array([d[:min_length] for d in all_data])

                # Calculate statistics
                mean_data = np.mean(aligned_data, axis=0)
                steps_aligned = np.array(steps[:min_length])

                # Calculate envelope bounds using min/max of all runs
                lower_bound = np.min(aligned_data, axis=0)
                upper_bound = np.max(aligned_data, axis=0)

                # Apply EMA smoothing to envelope bounds if requested
                if self.config.smooth_envelope:
                    alpha = self.config.ema_smoothing
                    # Smooth lower bound
                    smoothed_lower = np.zeros_like(lower_bound)
                    smoothed_lower[0] = lower_bound[0]
                    for i in range(1, len(lower_bound)):
                        smoothed_lower[i] = (
                            alpha * smoothed_lower[i - 1] + (1 - alpha) * lower_bound[i]
                        )

                    # Smooth upper bound
                    smoothed_upper = np.zeros_like(upper_bound)
                    smoothed_upper[0] = upper_bound[0]
                    for i in range(1, len(upper_bound)):
                        smoothed_upper[i] = (
                            alpha * smoothed_upper[i - 1] + (1 - alpha) * upper_bound[i]
                        )

                    lower_bound = smoothed_lower
                    upper_bound = smoothed_upper

                # Plot mean line
                plt.plot(
                    steps_aligned,
                    mean_data,
                    color=color,
                    label=group_name,
                    linewidth=self.config.line_weight,
                )

                # Plot envelope if requested
                if self.config.show_envelope:
                    plt.fill_between(
                        steps_aligned, lower_bound, upper_bound, color=color, alpha=0.2
                    )

        # Customize the plot
        plt.xlabel(self.config.x_axis_label, fontsize=12)
        plt.ylabel(self.config.y_axis_label, fontsize=12)

        title = (
            self.config.title
            if self.config.title
            else f"{self.config.atari_game.title()} - Training Progress"
        )
        plt.title(title, fontsize=14, fontweight="bold")

        if self.config.show_legend:
            plt.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.atari_game}_{timestamp}.{self.config.output_format}"
        output_path = output_dir / filename

        if self.config.output_format == "svg":
            plt.savefig(output_path, format="svg", bbox_inches="tight")
        else:  # png
            plt.savefig(
                output_path, dpi=self.config.resolution_dpi, bbox_inches="tight"
            )

        print(f"Graph saved to: {output_path}")

        plt.show()


def main():
    """Main entry point"""
    config = tyro.cli(GraphConfig)

    # Validate inputs
    if not Path(config.csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {config.csv_path}")

    if not (0 <= config.ema_smoothing <= 1):
        raise ValueError("EMA smoothing must be between 0 and 1")

    if not (0 <= config.opacity <= 1):
        raise ValueError("Opacity must be between 0 and 1")

    if config.group not in ["none", "default", "custom"]:
        raise ValueError("Group mode must be 'none', 'default', or 'custom'")

    if config.group == "custom" and not config.group_config:
        raise ValueError("Custom grouping requires --group-config to be specified")

    if config.line_weight <= 0:
        raise ValueError("Line weight must be positive")

    if config.output_format not in ["png", "svg"]:
        raise ValueError("Output format must be 'png' or 'svg'")

    grapher = WandBGrapher(config)
    grapher.create_graph()


if __name__ == "__main__":
    main()
