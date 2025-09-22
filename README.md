# WandB CSV to Graph Converter

A comprehensive tool to convert CSV files exported from wandb.ai to publication-ready graphs with customizable grouping, smoothing, and visualization options.

## Features

- **Flexible Data Parsing**: Automatically parses WandB CSV files and extracts run information
- **Customizable Pattern Matching**: Configurable pattern extraction for run names with `MATCH_START` and `MATCH_END` constants
- **Smart Grouping**: Groups runs by name patterns (e.g., `rf14`, `baseline`) with optional custom grouping via TOML configuration
- **Custom Group Colors**: Specify custom colors for individual groups in groups.toml
- **Multiple Smoothing Options**: Support for various smoothing algorithms including EMA, moving average, min/max
- **Customizable Envelopes**: Choose between min-max, standard deviation, or standard error envelopes
- **Publication Ready**: Export to PDF, PNG, or processed CSV with customizable styling
- **Configurable Visualization**: Full control over colors, fonts, legends, and layout via dataclass settings

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage - create a PNG graph (default)
python main.py --env-id kangaroo --csv-file in/kangaroo.csv --title "Kangaroo Training Results"

# Export as PDF with higher DPI
python main.py --env-id kangaroo --csv-file in/kangaroo.csv --type PDF --dpi 600

# Export as SVG (vector graphics)
python main.py --env-id kangaroo --csv-file in/kangaroo.csv --type SVG

# Use custom grouping and smoothing
python main.py --env-id kangaroo --csv-file in/kangaroo.csv --custom-groups config/my_groups.toml --smoothing average --smoothing-amount 0.8
```

## Command Line Arguments

### Required Arguments

- `--env-id`: Atari environment ID (e.g., `kangaroo`)
- `--csv-file`: Path to the WandB CSV file

### Optional Arguments

- `--type`: Output format - `PNG` (default), `PDF`, `SVG`, or `CSV`
- `--title`: Graph title (default: empty)
- `--dpi`: Output DPI for images (default: 300)
- `--custom-groups`: Path to custom groups TOML file (default: `config/groups.toml`)
- `--smoothing`: Smoothing algorithm - `ema` (default), `average`, `min`, `max`, `mean`, `none`
- `--smoothing-amount`: Smoothing strength from 0 (none) to 1 (max) (default: 0.95)
- `--graph-envelope`: Envelope type - `std-dev` (default), `minmax`, `std-err`
- `--envelope-smoothing`: Apply smoothing to envelope (default: true)
- `--show-original-lines`: Show faded original lines behind aggregated data (default: false)
- `--legend-position`: Legend placement - `inside` (default), `top`, `bottom`, `left`, `right`

## Project Structure

```
WandB-csv/
├── main.py                      # Main script
├── requirements.txt             # Dependencies
├── README.md                   # Documentation
├── in/                         # Input CSV files
│   └── kangaroo.csv
├── out/                        # Generated output files (with timestamps)
│   ├── kangaroo_20250920_120000_graph.png
│   ├── kangaroo_20250920_120000_graph.pdf
│   ├── kangaroo_20250920_120000_graph.svg
│   └── kangaroo_20250920_120000_processed.csv
├── config/                     # Configuration files
│   ├── groups.toml            # Run grouping definitions
│   ├── colors.toml            # Color palette
│   └── graph.toml             # Graph styling
└── utils/                      # Utility scripts
    ├── test_visualizer.py     # Test suite
    └── custom_example.py      # Extension example
```

## Configuration Files

The tool uses three TOML configuration files in the `config/` directory for customization:

### config/groups.toml - Run Grouping

Define how runs should be grouped together:

```toml
[Baseline]
members = ["baseline"]
dotted = false
color = "#ff0000"  # Optional custom color

["RF Group 1-5"]
members = ["rf1", "rf2", "rf3", "rf4", "rf5"]
dotted = false

["RF Group 6-10"] 
members = ["rf6", "rf7", "rf8", "rf9", "rf10"]
dotted = true  # Use dashed lines
color = "#00aa00"  # Optional custom color
```

- `members`: List of run name patterns to include in this group
- `dotted`: Use dashed lines for this group (default: false)
- `color`: Optional custom color in hex format (e.g., "#ff0000" for red). If not specified, uses default color palette

## Customizable Pattern Matching

The tool uses configurable constants in the `WandBVisualizer` class to extract group names from CSV column headers:

```python
MATCH_START = "{env_id}-"  # Pattern that marks the start of the group name extraction
MATCH_END = "__"           # Character sequence that marks the end of the group name extraction
```

### How Pattern Matching Works

For a CSV column header like:

```
"ALE/Kangaroo-v5__19-09-kangaroo-rf21__745__1758407694 - charts/Episodic_Original_Reward"
```

With `--env-id kangaroo`:

1. `MATCH_START = "{env_id}-"` becomes `"kangaroo-"`
2. The tool finds `"kangaroo-rf21__"` in the header
3. Extracts `"rf21"` as the group name (between `"kangaroo-"` and `"__"`)

### Customizing Pattern Matching

To change how group names are extracted, modify the class constants:

```python
# Example: Extract from different positions
MATCH_START = "/"          # Extract after forward slash  
MATCH_END = "-"            # Extract before dash

# Example: Extract from start of string
MATCH_START = None         # Start from beginning of string
MATCH_END = "__"           # Extract until double underscore

# Example: Extract until end of string  
MATCH_START = "{env_id}-"  # Start after env_id and dash
MATCH_END = None           # Extract until end of string
```

**Note**: When `MATCH_START` or `MATCH_END` is `None`, the pattern will match from the start or until the end of the string respectively.

### config/colors.toml - Color Palette

Customize the colors used for different groups:

```toml
colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    # ... add more as needed
]
```

## Data Processing Pipeline

1. **CSV Parsing**: Reads WandB CSV and identifies run columns (excludes `_MIN` and `_MAX` columns)
2. **Run Extraction**: Extracts run names from column headers using pattern matching
3. **Grouping**: Groups runs either by default naming or custom TOML configuration
4. **Data Alignment**: Interpolates all runs to a common step grid for aggregation
5. **Smoothing**: Applies selected smoothing algorithm to individual runs or aggregated data
6. **Envelope Calculation**: Computes confidence bands using selected method
7. **Visualization**: Renders final graph with all styling options

## Smoothing Algorithms

- **EMA (Exponential Moving Average)**: Time-weighted smoothing that gives more weight to recent values
- **Average**: Simple moving average over a window
- **Min/Max**: Moving minimum/maximum over a window
- **Mean**: Same as average (alias)
- **None**: No smoothing applied

The `smoothing-amount` parameter controls the strength:

- `0.0`: No smoothing
- `0.95`: Strong smoothing (default for EMA)
- `1.0`: Maximum smoothing

## Envelope Types

- **std-dev**: Mean ± standard deviation
- **std-err**: Mean ± standard error
- **minmax**: Minimum and maximum values across runs

## CSV Input Format

The tool expects WandB CSV exports with the following structure:

```csv
"Step","ALE/Kangaroo-v5__19-09-kangaroo-rf15__237__1758314296 - charts/Episodic_Original_Reward","ALE/Kangaroo-v5__19-09-kangaroo-rf15__237__1758314296 - charts/Episodic_Original_Reward__MIN",...
```

- First column: `Step`
- Data columns: Run identifiers with metrics
- MIN/MAX columns: Automatically filtered out
- Run name extraction: Uses customizable pattern matching to identify group names (e.g., `rf15`, `baseline`)
  - Default pattern extracts names between `{env_id}-` and `__`
  - Pattern can be customized by modifying `MATCH_START` and `MATCH_END` class constants

## Examples

### Basic PNG Export (Default)

```bash
python main.py --env-id kangaroo --csv-file data/kangaroo.csv --title "Kangaroo Training Performance"
```

### High-Resolution PDF with Custom Smoothing

```bash
python main.py --env-id kangaroo --csv-file data/kangaroo.csv --type PDF --dpi 600 --smoothing average --smoothing-amount 0.7
```

### SVG Export for Vector Graphics

```bash
python main.py --env-id kangaroo --csv-file data/kangaroo.csv --type SVG --title "Vector Graphics Export"
```

### Show Individual Runs with Envelope

```bash
python main.py --env-id kangaroo --csv-file data/kangaroo.csv --show-original-lines --graph-envelope minmax
```

### Export Processed Data as CSV

```bash
python main.py --env-id kangaroo --csv-file data/kangaroo.csv --type CSV
```

### Custom Grouping with External Legend

```bash
python main.py --env-id kangaroo --csv-file data/kangaroo.csv --custom-groups config/experiment_groups.toml --legend-position top
```

## Output Files

All output files are saved to the `out/` directory with timestamps:

- **PNG/PDF/SVG**: `out/<input_filename>_YYYYMMDD_HHMMSS_graph.png/pdf/svg`
- **CSV**: `out/<input_filename>_YYYYMMDD_HHMMSS_processed.csv` with columns: Step, Value, Group, Run

## Troubleshooting

### No runs found

- Check that your CSV file contains data columns with the expected naming pattern
- Verify that the run names contain the expected group identifiers between `{env_id}-` and `__` (default pattern)
- If using a different naming convention, modify the `MATCH_START` and `MATCH_END` constants in the `WandBVisualizer` class

### Import errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For TOML import issues, try: `pip install toml`

### Empty graphs

- Check that your custom groups TOML file correctly references the run names in your data
- Use `--type csv` to export processed data and verify grouping is working correctly

### Configuration not applied

- Ensure TOML files are in the `config/` directory
- Check TOML syntax using an online validator

## Contributing

Feel free to submit issues and enhancement requests! The tool is designed to be easily extensible for different naming schemes and visualization requirements.
