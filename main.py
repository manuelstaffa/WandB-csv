#!/usr/bin/env python3
"""
WandB CSV to Graph Converter

A tool to convert CSV files from wandb.ai to graphs with customizable grouping,
smoothing, and visualization options.
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tyro
import toml


class OutputType(Enum):
    PDF = "pdf"
    PNG = "png"
    SVG = "svg"
    CSV = "csv"


class SmoothingType(Enum):
    NONE = "none"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    EMA = "ema"  # Exponential Moving Average (time weighted EMA)


class EnvelopeType(Enum):
    MINMAX = "minmax"
    STD_DEV = "std-dev"
    STD_ERR = "std-err"


class LegendPosition(Enum):
    INSIDE = "inside"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Config:
    """Main configuration for the visualizer."""

    env_id: str
    csv_file: str
    type: OutputType = OutputType.PNG
    title: str = ""
    dpi: int = 300
    custom_groups: Optional[str] = "config/groups.toml"
    smoothing: SmoothingType = SmoothingType.EMA
    smoothing_amount: float = 0.9
    graph_envelope: EnvelopeType = EnvelopeType.STD_DEV
    envelope_smoothing: bool = True
    show_original_graph: bool = False
    original_graph_smoothing: bool = False
    x_axis_field: str = "global_step"
    y_axis_field: str = "Episodic_Original_Reward"


@dataclass
class GraphSettings:
    """Graph styling settings from graph.toml"""

    figure_size: Tuple[float, float] = (9, 6)
    x_axis_name: str = "Step"
    y_axis_name: str = "Episodic Original Reward"
    envelope_opacity: float = 0.2
    font_color: str = "#000000"
    font_size: int = 12
    font_weight: str = "normal"
    box_color: str = "#FFFFFF"
    line_thickness: float = 2.0
    grid_color: str = "#CCCCCC"
    grid_thickness: float = 0.5
    line_width: float = 2.0
    original_line_thickness: float = 0.5
    legend_position: LegendPosition = LegendPosition.INSIDE
    legend_box: bool = False
    legend_pattern: bool = True
    legend_pattern_fade: float = 0.8
    envelope_patterns: bool = True
    envelope_pattern_scale: float = 0.25


@dataclass
class RunData:
    """Data for a single run"""

    name: str
    group: str
    steps: np.ndarray
    values: np.ndarray
    is_dotted: bool = False


class WandBVisualizer:
    """Main visualizer class for converting WandB CSV files to graphs."""

    # Constants for customizable pattern matching
    MATCH_START = (
        "{env_id}-"  # Pattern that marks the start of the group name extraction
    )
    MATCH_END = (
        "__"  # Character sequence that marks the end of the group name extraction
    )

    def __init__(self, config: Config):
        self.config = config
        self.graph_settings = GraphSettings()
        self.colors = self._load_colors()
        self.custom_groups = self._load_custom_groups()
        self.group_colors = {}

    def _get_config_name(self) -> str:
        """Get the configuration name for filename generation."""
        if not self.config.custom_groups:
            return "default"

        # Extract filename from path and remove extension
        config_path = Path(self.config.custom_groups)
        return config_path.stem  # This gets the filename without extension

    def _adjust_color_brightness(self, color: str, factor: float = 0.7) -> str:
        """Adjust the brightness of a hex color. Factor < 1 makes it darker, > 1 makes it lighter."""
        import matplotlib.colors as mcolors

        # Convert hex to RGB
        rgb = mcolors.hex2color(color)

        # Adjust brightness
        adjusted_rgb = (
            min(1.0, max(0.0, rgb[0] * factor)),
            min(1.0, max(0.0, rgb[1] * factor)),
            min(1.0, max(0.0, rgb[2] * factor)),
        )

        # Convert back to hex
        return mcolors.rgb2hex(adjusted_rgb)

    def _load_colors(self) -> List[str]:
        """Load colors from colors.toml"""
        colors_toml_path = Path("config/colors.toml")
        if colors_toml_path.exists():
            try:
                data = toml.load(colors_toml_path)
                return data.get("colors", self._default_colors())
            except Exception as e:
                warnings.warn(f"Error loading config/colors.toml: {e}. Using defaults.")
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
        ]

    def _load_custom_groups(self) -> Optional[Dict[str, Dict]]:
        """Load custom groups from groups.toml"""
        if not self.config.custom_groups:
            return None

        groups_path = Path(self.config.custom_groups)
        if not groups_path.exists():
            return None

        try:
            data = toml.load(groups_path)
            return data
        except Exception as e:
            warnings.warn(f"Error loading {self.config.custom_groups}: {e}")
            return None

    def parse_csv(self) -> List[RunData]:
        """Parse the WandB CSV file and extract run data."""
        csv_path = Path(self.config.csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.config.csv_file}")

        df = pd.read_csv(csv_path)
        runs = []

        # Get column names and filter out MIN/MAX columns
        columns = [col for col in df.columns if not col.endswith(("__MIN", "__MAX"))]
        # Filter columns to only include those containing the y_axis_field and exclude x_axis_field
        data_columns = [
            col
            for col in columns
            if col != self.config.x_axis_field and self.config.y_axis_field in col
        ]

        for col in data_columns:
            # Extract run information from column name
            run_info = self._extract_run_info(col)
            if run_info:
                run_name, group_name = run_info

                # Extract data (remove NaN values)
                mask = ~df[col].isna()
                steps = df.loc[mask, self.config.x_axis_field].values
                values = df.loc[mask, col].values

                if len(steps) > 0:  # Only add runs with data
                    runs.append(
                        RunData(
                            name=run_name,
                            group=group_name,
                            steps=np.array(steps),
                            values=np.array(values),
                        )
                    )

        return self._apply_grouping(runs)

    def _extract_run_info(self, column_name: str) -> Optional[Tuple[str, str]]:
        """Extract run name and group from column header."""
        # Pattern to match: ALE/Kangaroo-v5__date-env_id-GROUP__number__timestamp - charts/...
        # We want to extract the GROUP part between MATCH_START and MATCH_END
        # Build the pattern using the customizable constants

        # Handle MATCH_START: substitute {env_id} placeholder and handle None
        if self.MATCH_START is not None:
            start_pattern = self.MATCH_START.format(env_id=self.config.env_id)
            escaped_start = re.escape(start_pattern)
            start_regex = f".*?{escaped_start}"
        else:
            # If MATCH_START is None, match from the beginning of string
            start_regex = ""

        # Handle MATCH_END: handle None case
        if self.MATCH_END is not None:
            escaped_end = re.escape(self.MATCH_END)
            end_regex = f"{escaped_end}.*"
            # Create character class for stopping the capture group
            if len(self.MATCH_END) > 0:
                stop_char = re.escape(self.MATCH_END[0])
            else:
                stop_char = "_"  # fallback
        else:
            # If MATCH_END is None, match until the end of string
            end_regex = "$"
            stop_char = ""  # capture everything

        # Build the complete pattern
        if stop_char:
            pattern = f"{start_regex}([^{stop_char}]+){end_regex}"
        else:
            pattern = f"{start_regex}(.+){end_regex}"

        match = re.search(pattern, column_name)

        if match:
            group_name = match.group(1)
            return column_name, group_name

        return None

    def _apply_grouping(self, runs: List[RunData]) -> List[RunData]:
        """Apply custom grouping if specified, otherwise use default grouping."""
        if not self.custom_groups:
            return runs

        # Create a mapping of group names to new group names
        group_mapping = {}
        dotted_groups = set()
        self.group_colors = {}  # Store custom colors for groups

        for group_name, group_config in self.custom_groups.items():
            if isinstance(group_config, dict):
                members = group_config.get("members", [])
                is_dotted = group_config.get("dotted", False)
                custom_color = group_config.get("color")

                if is_dotted:
                    dotted_groups.add(group_name)

                # Store custom color if specified
                if custom_color:
                    self.group_colors[group_name] = custom_color
            else:
                # Legacy format: just a list
                members = group_config

            for member in members:
                group_mapping[member] = group_name

        # Apply grouping
        grouped_runs = []
        for run in runs:
            new_group = group_mapping.get(run.group, run.group)
            is_dotted = new_group in dotted_groups

            # Only include runs that are in the custom groups (if custom groups are specified)
            if run.group in group_mapping:
                grouped_runs.append(
                    RunData(
                        name=run.name,
                        group=new_group,
                        steps=run.steps,
                        values=run.values,
                        is_dotted=is_dotted,
                    )
                )

        return grouped_runs

    def smooth_data(self, runs: List[RunData]) -> List[RunData]:
        """Apply smoothing to the run data."""
        if self.config.smoothing == SmoothingType.NONE:
            return runs

        smoothed_runs = []
        for run in runs:
            smoothed_values = self._apply_smoothing(
                run.values,
                self.config.smoothing,
                self.config.smoothing_amount,
                run.steps,
            )
            smoothed_runs.append(
                RunData(
                    name=run.name,
                    group=run.group,
                    steps=run.steps,
                    values=smoothed_values,
                    is_dotted=run.is_dotted,
                )
            )

        return smoothed_runs

    def _apply_smoothing(
        self,
        values: np.ndarray,
        smoothing_type: SmoothingType,
        amount: float,
        steps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply the specified smoothing algorithm."""
        if smoothing_type == SmoothingType.NONE or amount == 0:
            return values

        if smoothing_type == SmoothingType.EMA:
            return self._exponential_moving_average(values, amount, steps)
        elif smoothing_type == SmoothingType.AVERAGE:
            window_size = max(1, int(len(values) * (1 - amount)))
            return self._moving_average(values, window_size)
        elif smoothing_type == SmoothingType.MIN:
            window_size = max(1, int(len(values) * (1 - amount)))
            return self._moving_min(values, window_size)
        elif smoothing_type == SmoothingType.MAX:
            window_size = max(1, int(len(values) * (1 - amount)))
            return self._moving_max(values, window_size)
        elif smoothing_type == SmoothingType.MEAN:
            # Same as average for now
            window_size = max(1, int(len(values) * (1 - amount)))
            return self._moving_average(values, window_size)

        return values

    def _exponential_moving_average(
        self, values: np.ndarray, alpha: float, steps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply exponential moving average smoothing with adaptive alpha based on step intervals."""
        if len(values) == 0:
            return values

        smoothed = np.zeros_like(values, dtype=np.float64)
        smoothed[0] = values[0]

        # If steps are provided, use adaptive alpha based on step intervals
        if steps is not None and len(steps) == len(values):
            # Calculate a reference interval (median or mean of intervals)
            intervals = np.diff(steps)
            if len(intervals) > 0:
                # Use median interval as reference to avoid outliers
                ref_interval = np.median(intervals)

                for i in range(1, len(values)):
                    # Calculate interval between current and previous step
                    current_interval = steps[i] - steps[i - 1]

                    # Adjust alpha based on the ratio of current interval to reference interval
                    # If interval is larger, reduce the effect of previous value (lower alpha)
                    # If interval is smaller, increase the effect of previous value (higher alpha)
                    if ref_interval > 0:
                        interval_ratio = current_interval / ref_interval
                        # Clamp the ratio to reasonable bounds to avoid extreme values
                        interval_ratio = np.clip(interval_ratio, 0.1, 10.0)

                        # Adaptive alpha: larger intervals -> smaller alpha (less smoothing from previous)
                        adaptive_alpha = alpha**interval_ratio
                    else:
                        adaptive_alpha = alpha

                    smoothed[i] = (
                        adaptive_alpha * smoothed[i - 1]
                        + (1 - adaptive_alpha) * values[i]
                    )
            else:
                # Fallback to standard EMA if no intervals can be calculated
                for i in range(1, len(values)):
                    smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
        else:
            # Standard EMA when no steps are provided
            for i in range(1, len(values)):
                smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]

        return smoothed

    def _moving_average(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if window_size <= 1:
            return values

        return np.convolve(values, np.ones(window_size) / window_size, mode="same")

    def _moving_min(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving minimum smoothing."""
        if window_size <= 1:
            return values

        result = np.zeros_like(values)
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            result[i] = np.min(values[start:end])

        return result

    def _moving_max(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving maximum smoothing."""
        if window_size <= 1:
            return values

        result = np.zeros_like(values)
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            result[i] = np.max(values[start:end])

        return result

    def create_visualization(self, runs: List[RunData]) -> None:
        """Create the visualization based on the configuration."""
        if self.config.type == OutputType.CSV:
            self._save_as_csv(runs)
        else:
            self._create_plot(runs)

    def _save_as_csv(self, runs: List[RunData]) -> None:
        """Save processed data as CSV."""
        # Group runs by group name
        grouped_data = {}
        for run in runs:
            if run.group not in grouped_data:
                grouped_data[run.group] = []
            grouped_data[run.group].append(run)

        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = self._get_config_name()
        output_path = (
            Path("out")
            / f"{self.config.env_id}_{config_name}_{timestamp}_processed.csv"
        )

        # Combine all data into a single DataFrame
        all_data = []
        for group_name, group_runs in grouped_data.items():
            for run in group_runs:
                df_run = pd.DataFrame(
                    {
                        "Step": run.steps,
                        "Value": run.values,
                        "Group": group_name,
                        "Run": run.name,
                    }
                )
                all_data.append(df_run)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(output_path, index=False)
            print(f"Processed data saved to: {output_path}")

    def _create_plot(self, runs: List[RunData]) -> None:
        """Create matplotlib plot."""
        # Group runs by group name, maintaining order from TOML config
        grouped_data = {}
        for run in runs:
            if run.group not in grouped_data:
                grouped_data[run.group] = []
            grouped_data[run.group].append(run)

        # Reorder grouped_data to match the order in custom_groups if available
        if self.custom_groups:
            ordered_grouped_data = {}
            # First, add groups in the order they appear in the TOML config
            for group_name in self.custom_groups.keys():
                if group_name in grouped_data:
                    ordered_grouped_data[group_name] = grouped_data[group_name]
            # Then add any remaining groups that weren't in the config
            for group_name, group_runs in grouped_data.items():
                if group_name not in ordered_grouped_data:
                    ordered_grouped_data[group_name] = group_runs
            grouped_data = ordered_grouped_data

        # Set up the plot
        plt.style.use("default")
        fig, ax = plt.subplots(
            figsize=self.graph_settings.figure_size, dpi=self.config.dpi
        )

        # Apply graph settings
        ax.set_xlabel(
            self.graph_settings.x_axis_name,
            fontsize=self.graph_settings.font_size,
            color=self.graph_settings.font_color,
            weight=self.graph_settings.font_weight,
        )
        ax.set_ylabel(
            self.graph_settings.y_axis_name,
            fontsize=self.graph_settings.font_size,
            color=self.graph_settings.font_color,
            weight=self.graph_settings.font_weight,
        )

        if self.config.title:
            ax.set_title(
                self.config.title,
                fontsize=self.graph_settings.font_size + 2,
                color=self.graph_settings.font_color,
                weight=self.graph_settings.font_weight,
            )

        # Grid settings
        ax.grid(
            True,
            color=self.graph_settings.grid_color,
            linewidth=self.graph_settings.grid_thickness,
        )
        ax.set_facecolor(self.graph_settings.box_color)

        # Plot each group
        legend_elements = []
        color_idx = 0
        pattern_idx = 0
        # Define hatch patterns for envelope differentiation
        hatch_patterns = ["", "///", "\\\\\\", "|||", "---", "+++", "xxx", "..."]

        for group_name, group_runs in grouped_data.items():
            # Check if group has a custom color, otherwise use default palette
            if group_name in self.group_colors:
                color = self.group_colors[group_name]
            else:
                color = self.colors[color_idx % len(self.colors)]
                color_idx += 1

            # Aggregate data for the group
            aggregated_data = self._aggregate_group_data(group_runs)

            if aggregated_data is None:
                continue

            steps, mean_values, envelope_data = aggregated_data

            # Determine line style
            linestyle = "--" if any(run.is_dotted for run in group_runs) else "-"

            # Determine hatch pattern if patterns are enabled (before plotting envelope)
            hatch = None
            if self.graph_settings.envelope_patterns:
                base_pattern = hatch_patterns[pattern_idx % len(hatch_patterns)]
                # Apply pattern scale by repeating the pattern
                if base_pattern and self.graph_settings.envelope_pattern_scale != 1.0:
                    # Scale the pattern by repeating characters
                    scale_factor = max(0.1, self.graph_settings.envelope_pattern_scale)
                    if scale_factor < 1.0:
                        # For scale < 1, use fewer pattern repetitions
                        pattern_length = max(1, int(len(base_pattern) * scale_factor))
                        hatch = base_pattern[:pattern_length]
                    else:
                        # For scale > 1, repeat the pattern
                        repetitions = int(scale_factor)
                        hatch = base_pattern * repetitions
                else:
                    hatch = base_pattern

            # Plot the envelope if requested
            if envelope_data is not None:
                lower, upper = envelope_data

                ax.fill_between(
                    steps,
                    lower,
                    upper,
                    alpha=self.graph_settings.envelope_opacity,
                    color=color,
                    hatch=hatch,
                )

            # Increment pattern index for next group
            pattern_idx += 1

            # Plot original graphs if requested
            if self.config.show_original_graph:
                for run in group_runs:
                    # Apply smoothing to original graph if requested
                    if self.config.original_graph_smoothing:
                        smoothed_values = self._apply_smoothing(
                            run.values,
                            self.config.smoothing,
                            self.config.smoothing_amount,
                            run.steps,
                        )
                        ax.plot(
                            run.steps,
                            smoothed_values,
                            color=color,
                            alpha=0.3,
                            linewidth=self.graph_settings.original_line_thickness,
                        )
                    else:
                        ax.plot(
                            run.steps,
                            run.values,
                            color=color,
                            alpha=0.3,
                            linewidth=self.graph_settings.original_line_thickness,
                        )  # Plot the main line
            ax.plot(
                steps,
                mean_values,
                color=color,
                linewidth=self.graph_settings.line_width,
                linestyle=linestyle,
                label=group_name,
            )

            # Create legend element
            legend_hatch = None
            if (
                self.graph_settings.legend_pattern
                and self.graph_settings.envelope_patterns
            ):
                # Create optimal density hatch for legend visibility
                if hatch and hatch.strip():  # Only if hatch is not empty
                    # Make legend hatch with moderate density for visibility
                    base_char = hatch[0] if hatch else ""
                    if base_char in ["/", "\\", "|", "-", "+", "x", "."]:
                        # Use moderate density (40% of previous 8 = ~3) for better visibility
                        legend_hatch = base_char * 3
                    else:
                        # For already complex patterns, use original pattern
                        legend_hatch = hatch
                else:
                    legend_hatch = hatch

            legend_elements.append(
                patches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=color,
                    hatch=legend_hatch,
                    edgecolor=(
                        self._adjust_color_brightness(
                            color, factor=self.graph_settings.legend_pattern_fade
                        )
                        if legend_hatch
                        else None
                    ),
                    label=group_name,
                )
            )

        # Add legend
        self._add_legend(ax, legend_elements)

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = self._get_config_name()
        output_path = (
            Path("out")
            / f"{self.config.env_id}_{config_name}_{timestamp}_graph.{self.config.type.value}"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        print(f"Graph saved to: {output_path}")

    def _aggregate_group_data(
        self, group_runs: List[RunData]
    ) -> Optional[
        Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]
    ]:
        """Aggregate data for a group of runs."""
        if not group_runs:
            return None

        # Find common step range
        min_step = max(run.steps.min() for run in group_runs)
        max_step = min(run.steps.max() for run in group_runs)

        if min_step >= max_step:
            return None

        # Create interpolated data on common grid
        common_steps = np.linspace(min_step, max_step, 1000)
        interpolated_values = []

        for run in group_runs:
            interpolated = np.interp(common_steps, run.steps, run.values)
            interpolated_values.append(interpolated)

        if not interpolated_values:
            return None

        values_array = np.array(interpolated_values)

        # Calculate mean
        mean_values = np.mean(values_array, axis=0)

        # Apply smoothing to mean if requested
        if self.config.envelope_smoothing:
            mean_values = self._apply_smoothing(
                mean_values,
                self.config.smoothing,
                self.config.smoothing_amount,
                common_steps,
            )

        # Calculate envelope
        envelope_data = None
        if len(group_runs) > 1:
            lower = None
            upper = None

            if self.config.graph_envelope == EnvelopeType.MINMAX:
                lower = np.min(values_array, axis=0)
                upper = np.max(values_array, axis=0)
            elif self.config.graph_envelope == EnvelopeType.STD_DEV:
                std = np.std(values_array, axis=0)
                lower = mean_values - std
                upper = mean_values + std
            elif self.config.graph_envelope == EnvelopeType.STD_ERR:
                std_err = np.std(values_array, axis=0) / np.sqrt(len(group_runs))
                lower = mean_values - std_err
                upper = mean_values + std_err

            # Apply smoothing to envelope if requested
            if (
                self.config.envelope_smoothing
                and lower is not None
                and upper is not None
            ):
                lower = self._apply_smoothing(
                    lower,
                    self.config.smoothing,
                    self.config.smoothing_amount,
                    common_steps,
                )
                upper = self._apply_smoothing(
                    upper,
                    self.config.smoothing,
                    self.config.smoothing_amount,
                    common_steps,
                )

            if lower is not None and upper is not None:
                envelope_data = (lower, upper)

        return common_steps, mean_values, envelope_data

    def _add_legend(self, ax, legend_elements: List) -> None:
        """Add legend to the plot."""
        if not legend_elements:
            return

        if self.graph_settings.legend_position == LegendPosition.INSIDE:
            ax.legend(
                handles=legend_elements,
                loc="best",
                frameon=self.graph_settings.legend_box,
            )
        elif self.graph_settings.legend_position == LegendPosition.TOP:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(0.5, 1.02),
                loc="lower center",
                ncol=len(legend_elements),
                frameon=self.graph_settings.legend_box,
            )
        elif self.graph_settings.legend_position == LegendPosition.BOTTOM:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(0.5, -0.05),
                loc="upper center",
                ncol=len(legend_elements),
                frameon=self.graph_settings.legend_box,
            )
        elif self.graph_settings.legend_position == LegendPosition.LEFT:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(-0.05, 0.5),
                loc="center right",
                frameon=self.graph_settings.legend_box,
            )
        elif self.graph_settings.legend_position == LegendPosition.RIGHT:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 0.5),
                loc="center left",
                frameon=self.graph_settings.legend_box,
            )


def main():
    """Main entry point."""
    config = tyro.cli(Config, description=__doc__, use_underscores=False)

    # Create default config files if they don't exist
    create_default_configs()

    # Create visualizer and process data
    visualizer = WandBVisualizer(config)
    runs = visualizer.parse_csv()

    if not runs:
        print("No valid runs found in the CSV file.")
        return

    print(f"Found {len(runs)} runs")

    # Apply smoothing
    smoothed_runs = visualizer.smooth_data(runs)

    # Create visualization
    visualizer.create_visualization(smoothed_runs)


def create_default_configs():
    """Create default configuration files if they don't exist."""
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Create output directory
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    # Create default groups.toml
    if not (config_dir / "groups.toml").exists():
        default_groups = {
            "Baseline": {"members": ["baseline"], "dotted": False, "color": "#ff0000"},
            "RF Group 1": {"members": ["rf1", "rf2", "rf3"], "dotted": False},
            "RF Group 2": {"members": ["rf4", "rf5", "rf6"], "dotted": True},
        }
        with open(config_dir / "groups.toml", "w") as f:
            toml.dump(default_groups, f)

    # Create default colors.toml
    if not (config_dir / "colors.toml").exists():
        default_colors = {
            "colors": [
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
            ]
        }
        with open(config_dir / "colors.toml", "w") as f:
            toml.dump(default_colors, f)


if __name__ == "__main__":
    main()
