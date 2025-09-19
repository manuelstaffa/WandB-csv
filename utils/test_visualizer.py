#!/usr/bin/env python3
"""
Test script to validate different features of the WandB visualizer.
"""

import os
import sys
from pathlib import Path


def run_test(description, command):
    """Run a test command and report results."""
    print(f"\n🧪 Testing: {description}")
    print(f"Command: {command}")
    result = os.system(command)
    if result == 0:
        print("✅ Success")
    else:
        print("❌ Failed")
    return result == 0


def main():
    """Run various tests of the visualizer."""

    base_cmd = "python ../main.py --env-id kangaroo --csv-file ../in/kangaroo.csv"

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("🚀 Starting WandB Visualizer Tests")

    tests = [
        ("Basic PNG generation", f"{base_cmd} --title 'Test Graph'"),
        (
            "PDF output with custom DPI",
            f"{base_cmd} --type PDF --dpi 150 --title 'PDF Test'",
        ),
        ("SVG output", f"{base_cmd} --type SVG --title 'SVG Test'"),
        ("CSV export", f"{base_cmd} --type CSV"),
        ("Custom smoothing", f"{base_cmd} --smoothing AVERAGE --smoothing-amount 0.7"),
        ("Min-max envelope", f"{base_cmd} --graph-envelope MINMAX"),
        ("Show original lines", f"{base_cmd} --show-original-lines"),
        ("Top legend position", f"{base_cmd} --legend-position TOP"),
        ("No smoothing", f"{base_cmd} --smoothing NONE"),
    ]

    passed = 0
    total = len(tests)

    for description, command in tests:
        if run_test(description, command):
            passed += 1

    print(f"\n📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
