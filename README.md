# WandB CSV to Graph Converter
### Required Parameters

```bash
python main.py --csv-path in/your_file.csv --atari-game kangaroo
```werful tool to convert CSV files exported from Weights & Biases (WandB) into beautiful, customizable graphs with smoothing, grouping, and visualization options.

## F### Batch Processing

For processing multiple games or configurations, you can use the batch utility:

```bash
python utils/batch_process.py wandb_data.csv kangaroo --preset analysis
``` **EMA Smoothing**: Apply Exponential Moving Average smoothing to reduce noise
- **Flexible Grouping**: Three grouping modes - none, default (by identifier), or custom
- **Custom Group Configuration**: Define complex groups via TOML configuration files
- **Envelope Visualization**: Show min/max bounds of grouped runs with optional smoothing
- **Original Run Visibility**: Control whether to show individual runs when grouping
- **Line Weight Control**: Customize line thickness for better visibility
- **Color Customization**: Use custom color palettes via TOML configuration
- **Flexible Output**: Save high-resolution images with timestamps
- **Command Line Interface**: Easy-to-use CLI with sensible defaults

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Place your WandB CSV export files in the `in/` directory

## Project Structure

```
WandB-csv/
├── main.py                    # Main converter tool
├── requirements.txt           # Dependencies  
├── README.md                  # This guide
├── config/
│   ├── colors.toml           # Color configuration
│   └── groups.toml           # Custom grouping configuration
├── in/                       # Input CSV files (place your files here)
├── output/                   # Generated graphs
└── utils/
    ├── batch_process.py      # Batch processing utility
    ├── demo.py               # Feature demonstration
    └── test.py               # Test suite
```

## Basic Usage

### Required Parameters

```bash
python main.py --csv-path path/to/your/file.csv --atari-game kangaroo
```

### Common Usage Examples

```bash
# Basic usage with default grouping
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo

# No grouping - show all individual runs
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo --group none

# Custom grouping with configuration file
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo \
    --group custom --group-config config/groups.toml

# Envelope only (hide individual runs when grouped)
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo \
    --no-show-original-when-grouped

# Custom smoothing and line weight
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo \
    --ema-smoothing 0.95 --line-weight 3.0

# Custom labels and title with thick lines
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo \
    --x-axis-label "Training Steps" \
    --y-axis-label "Average Reward" \
    --title "Kangaroo Training Results" \
    --line-weight 2.5

# High resolution output
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo --resolution-dpi 600

# SVG output format
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo --output-format svg

# Smoothed envelope visualization
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo \
    --smooth-envelope --ema-smoothing 0.95

# Combined features: SVG output with smoothed envelope
python main.py --csv-path in/wandb_export.csv --atari-game kangaroo \
    --output-format svg --smooth-envelope --title "Smoothed Envelope Analysis"
```

## Configuration Files

### Color Configuration

Create custom color palettes by editing `config/colors.toml`:

```toml
colors = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    # ... add more colors
]
```

Use your custom colors:
```bash
python main.py --csv-path in/data.csv --atari-game kangaroo --config-file config/colors.toml
```

### Custom Grouping

The most powerful feature is custom grouping via TOML configuration files. Create a file like `config/groups.toml`:

```toml
[groups]
reward_functions = ["rf13", "rf14"]
baseline_comparison = ["baseline", "rf1"]
early_experiments = ["rf4", "rf5", "rf6"]
advanced_experiments = ["rf10", "rf11", "rf12"]
```

Use your custom groups:
```bash
python main.py --csv-path in/data.csv --atari-game kangaroo \
    --group custom --group-config config/groups.toml
```

**Grouping Modes:**
- `--group none`: No grouping, show all runs individually
- `--group default`: Group by run identifier (rf1, rf2, baseline, etc.)
- `--group custom`: Use custom grouping from config file

## Command Line Parameters

### Required
- `--csv-path`: Path to the WandB CSV export file
- `--atari-game`: Name of the Atari game/environment (e.g., "kangaroo")

### Optional (with defaults)
- `--ema-smoothing`: EMA smoothing factor (0.0-1.0, default: 0.9)
- `--group`: Grouping mode - none/default/custom (default: default)
- `--group-config`: Path to custom group configuration file (required for custom mode)
- `--opacity`: Opacity for individual runs when grouped (0.0-1.0, default: 0.3)
- `--show-envelope`: Show min/max envelope for groups (default: True)
- `--smooth-envelope`: Apply EMA smoothing to envelope bounds (default: False)
- `--show-original-when-grouped`: Show individual runs when grouping (default: True)
- `--line-weight`: Line thickness for graphs (default: 2.0)
- `--x-axis-label`: X-axis label (default: "Steps")
- `--y-axis-label`: Y-axis label (default: "Episodic Reward")
- `--title`: Graph title (default: auto-generated)
- `--show-legend`: Show legend (default: True)
- `--resolution-dpi`: Output resolution (default: 300)
- `--output-format`: Output format - png or svg (default: png)
- `--output-dir`: Output directory (default: "output")
- `--config-file`: Path to custom configuration TOML file
- `--custom-groups`: List of custom group identifiers (deprecated - use custom grouping instead)

### Boolean Flags
Use `--no-` prefix to disable boolean options:
- `--no-show-envelope`: Hide envelope visualization
- `--no-show-original-when-grouped`: Hide individual runs when grouping (envelope only)
- `--no-show-legend`: Hide legend

## CSV File Format

The tool expects CSV files exported from WandB with the following structure:

```csv
Step,ALE/Game-v5__date-game-identifier__number__timestamp - charts/Metric,ALE/Game-v5__date-game-identifier__number__timestamp - charts/Metric__MIN,ALE/Game-v5__date-game-identifier__number__timestamp - charts/Metric__MAX
```

- **Step**: Training step/episode number
- **Metric columns**: Main data columns (MIN/MAX columns are automatically ignored)
- **Run identifiers**: Extracted from column names (e.g., rf1, rf2, baseline)

## Output

- Graphs are saved as PNG files in the specified output directory
- Filename format: `{game_name}_{timestamp}.png`
- The graph is also displayed on screen (if display is available)

## Advanced Features

### Multiple Group Combination

Create sophisticated group combinations:

```bash
# Custom grouping with envelope-only visualization
python main.py --csv-path in/data.csv --atari-game kangaroo \
    --group custom --group-config config/groups.toml \
    --no-show-original-when-grouped

# Thick lines with custom grouping
python main.py --csv-path in/data.csv --atari-game kangaroo \
    --group custom --group-config config/groups.toml \
    --line-weight 3.5 --opacity 0.2
```

### High-Quality Publication Figures

```bash
python main.py --csv-path in/data.csv --atari-game kangaroo \
    --resolution-dpi 600 \
    --ema-smoothing 0.95 \
    --line-weight 2.5 \
    --group custom --group-config config/groups.toml \
    --title "Kangaroo: Reward Function Comparison" \
    --x-axis-label "Training Episodes" \
    --y-axis-label "Cumulative Reward"
```

### Batch Processing

For processing multiple graphs with different settings, use the batch processing utility:

```bash
# Create analysis variations
python batch_process.py wandb_export.csv kangaroo --preset analysis

# Create publication-ready figures
python batch_process.py wandb_export.csv kangaroo --preset publication

# Create comparison graphs  
python batch_process.py wandb_export.csv kangaroo --preset comparison
```

### Manual Batch Processing

For shell-based batch processing:

```bash
#!/bin/bash
for game in kangaroo pong breakout; do
    python main.py --csv-path "in/wandb_${game}.csv" --atari-game "$game"
done
```

## Additional Utilities

### Test Suite
Run the test suite to verify functionality:
```bash
python utils/test.py
```

### Demo Script  
See all features in action:
```bash
python utils/demo.py
```

### Batch Processing
Process multiple variations automatically:
```bash
python utils/batch_process.py --help
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install all requirements with `pip install -r requirements.txt`
2. **File not found**: Check the CSV file path and ensure it exists
3. **No data for game**: Verify the game name matches the column names in your CSV
4. **Empty graphs**: Check if your CSV contains valid numeric data (not all empty strings)

### Debug Tips

- Use `--group none` to see individual runs without any grouping
- Use `--no-show-original-when-grouped` to see only group means and envelopes
- Check the console output for data loading and grouping information
- Verify your CSV file structure matches the expected format
- For custom grouping, check that your group config file has the correct format

## Contributing

Feel free to contribute improvements, bug fixes, or new features. The code is designed to be modular and extensible.

## License

This tool is provided as-is for research and educational purposes.