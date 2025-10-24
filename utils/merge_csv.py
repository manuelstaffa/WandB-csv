#!/usr/bin/env python3
"""
CSV Merger utility for merging multiple WandB export CSV files.

This script combines two or more CSV files with the same structure,
merging data based on the global_step column and combining experiment columns.
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional
import sys


class CSVMerger:
    """Handles merging of multiple CSV files based on global_step."""

    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize the CSV merger.

        Args:
            output_path: Optional path for output file. If None, will be auto-generated.
        """
        self.output_path = output_path

    def merge_files(self, file_paths: List[Path]) -> pd.DataFrame:
        """
        Merge multiple CSV files based on global_step column.

        Args:
            file_paths: List of paths to CSV files to merge

        Returns:
            Merged pandas DataFrame

        Raises:
            ValueError: If files have incompatible structures or no global_step column
            FileNotFoundError: If any input file doesn't exist
        """
        if len(file_paths) < 2:
            raise ValueError("At least two files are required for merging")

        # Validate all files exist
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Merging {len(file_paths)} files...")

        # Load first file as base
        base_df = pd.read_csv(file_paths[0])
        print(
            f"Loaded base file: {file_paths[0]} ({len(base_df)} rows, {len(base_df.columns)} columns)"
        )

        # Validate base file has global_step column
        if "global_step" not in base_df.columns:
            raise ValueError(f"Base file {file_paths[0]} missing 'global_step' column")

        # Merge with remaining files
        for i, file_path in enumerate(file_paths[1:], 1):
            print(f"Merging file {i+1}/{len(file_paths)}: {file_path}")

            # Load next file
            next_df = pd.read_csv(file_path)
            print(f"  Loaded: {len(next_df)} rows, {len(next_df.columns)} columns")

            # Validate has global_step column
            if "global_step" not in next_df.columns:
                raise ValueError(f"File {file_path} missing 'global_step' column")

            # Merge on global_step column
            # Use outer join to include all global_step values from both files
            merged_df = pd.merge(
                base_df,
                next_df,
                on="global_step",
                how="outer",
                suffixes=("", f"_file{i+1}"),
            )

            print(
                f"  After merge: {len(merged_df)} rows, {len(merged_df.columns)} columns"
            )
            base_df = merged_df

        # Sort by global_step for consistency
        base_df = base_df.sort_values("global_step").reset_index(drop=True)

        print(
            f"Final merged dataset: {len(base_df)} rows, {len(base_df.columns)} columns"
        )
        return base_df

    def save_merged_data(self, merged_df: pd.DataFrame, file_paths: List[Path]) -> Path:
        """
        Save merged DataFrame to CSV file.

        Args:
            merged_df: Merged DataFrame to save
            file_paths: Original file paths (used for auto-naming if output_path not set)

        Returns:
            Path to the saved file
        """
        if self.output_path:
            output_path = self.output_path
        else:
            # Auto-generate output filename
            base_names = [fp.stem for fp in file_paths]
            output_name = f"merged_{'_'.join(base_names)}.csv"
            output_path = file_paths[0].parent / output_name

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving merged data to: {output_path}")
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(merged_df)} rows to {output_path}")

        return output_path

    def get_merge_summary(
        self, merged_df: pd.DataFrame, file_paths: List[Path]
    ) -> dict:
        """
        Generate summary statistics about the merge operation.

        Args:
            merged_df: The merged DataFrame
            file_paths: Original file paths

        Returns:
            Dictionary with summary information
        """
        # Count experiment columns (exclude global_step)
        experiment_cols = [col for col in merged_df.columns if col != "global_step"]

        # Count non-null values per original file (approximate)
        file_contributions = {}
        for i, file_path in enumerate(file_paths):
            if i == 0:
                # For base file, count columns that don't have suffix
                base_cols = [
                    col
                    for col in experiment_cols
                    if not any(
                        col.endswith(f"_file{j+1}") for j in range(1, len(file_paths))
                    )
                ]
                file_contributions[file_path.name] = len(base_cols)
            else:
                # For other files, count columns with their suffix
                suffix_cols = [
                    col for col in experiment_cols if col.endswith(f"_file{i+1}")
                ]
                file_contributions[file_path.name] = len(suffix_cols)

        summary = {
            "total_rows": len(merged_df),
            "total_columns": len(merged_df.columns),
            "experiment_columns": len(experiment_cols),
            "global_step_range": (
                merged_df["global_step"].min(),
                merged_df["global_step"].max(),
            ),
            "file_contributions": file_contributions,
            "files_merged": len(file_paths),
        }

        return summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Merge multiple WandB CSV export files based on global_step column",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two files
  python merge_csv.py file1.csv file2.csv
  
  # Merge multiple files with custom output
  python merge_csv.py file1.csv file2.csv file3.csv -o merged_results.csv
  
  # Merge files from input directory
  python merge_csv.py in/kangaroo.csv in/seaquest.csv -o out/combined_experiments.csv
        """,
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="CSV files to merge (minimum 2 files required)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (if not specified, auto-generated name will be used)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed merge information"
    )

    args = parser.parse_args()

    try:
        # Create merger
        merger = CSVMerger(output_path=args.output)

        # Perform merge
        merged_df = merger.merge_files(args.files)

        # Save results
        output_path = merger.save_merged_data(merged_df, args.files)

        # Print summary
        if args.verbose:
            summary = merger.get_merge_summary(merged_df, args.files)
            print("\n=== Merge Summary ===")
            print(f"Files merged: {summary['files_merged']}")
            print(f"Total rows: {summary['total_rows']:,}")
            print(f"Total columns: {summary['total_columns']:,}")
            print(f"Experiment columns: {summary['experiment_columns']:,}")
            print(
                f"Global step range: {summary['global_step_range'][0]:,} to {summary['global_step_range'][1]:,}"
            )
            print("\nFile contributions:")
            for filename, col_count in summary["file_contributions"].items():
                print(f"  {filename}: {col_count} columns")

        print(f"\n✅ Successfully merged {len(args.files)} files into: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
