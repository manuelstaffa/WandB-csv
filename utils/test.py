#!/usr/bin/env python3
"""
Test script for the WandB CSV to Graph Converter
"""

import subprocess
import sys
from pathlib import Path


def test_basic_functionality():
    """Test basic functionality with the provided CSV file"""
    csv_path = "wandb_export_2025-09-19T15_36_24.891+02_00.csv"

    # Test 1: Basic functionality
    print("Test 1: Basic functionality")
    cmd = [
        sys.executable,
        "main.py",
        "--csv-path",
        csv_path,
        "--atari-game",
        "kangaroo",
        "--ema-smoothing",
        "0.9",
        "--resolution-dpi",
        "150",  # Lower resolution for faster testing
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ Basic functionality test passed")
        else:
            print(f"✗ Basic functionality test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Basic functionality test timed out")
        return False
    except Exception as e:
        print(f"✗ Basic functionality test error: {e}")
        return False

    # Test 2: No grouping
    print("\nTest 2: No grouping")
    cmd = [
        sys.executable,
        "main.py",
        "--csv-path",
        csv_path,
        "--atari-game",
        "kangaroo",
        "--group",
        "none",
        "--resolution-dpi",
        "150",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ No grouping test passed")
        else:
            print(f"✗ No grouping test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ No grouping test error: {e}")
        return False

    # Test 3: Custom parameters
    print("\nTest 3: Custom parameters")
    cmd = [
        sys.executable,
        "main.py",
        "--csv-path",
        csv_path,
        "--atari-game",
        "kangaroo",
        "--ema-smoothing",
        "0.95",
        "--opacity",
        "0.2",
        "--title",
        "Test Graph",
        "--x-axis-label",
        "Episodes",
        "--y-axis-label",
        "Score",
        "--resolution-dpi",
        "150",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ Custom parameters test passed")
        else:
            print(f"✗ Custom parameters test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Custom parameters test error: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("WandB CSV to Graph Converter - Test Suite")
    print("=" * 50)

    # Check if CSV file exists
    csv_path = Path("wandb_export_2025-09-19T15_36_24.891+02_00.csv")
    if not csv_path.exists():
        print(f"✗ Test CSV file not found: {csv_path}")
        print("Please ensure the CSV file is in the current directory")
        return False

    # Check if main script exists
    main_script = Path("main.py")
    if not main_script.exists():
        print(f"✗ Main script not found: {main_script}")
        return False

    # Run tests
    if test_basic_functionality():
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("Check the 'output' directory for generated graphs.")
        return True
    else:
        print("\n" + "=" * 50)
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
