#!/usr/bin/env python3
"""
XPU/Triton Op Analysis Tool

This script analyzes PyTorch ATen operators to identify which ones need
XPU and/or Triton implementations. It processes native_functions.yaml to:
1. Extract ops with pointwise or reduction tags
2. Check XPU implementation status
3. Check Triton/Inductor coverage

Usage:
    python tools/xpu_triton_analysis/analyze_ops.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directories to path for imports
PYTORCH_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PYTORCH_ROOT))

from tools.xpu_triton_analysis.yaml_parser import parse_native_functions
from tools.xpu_triton_analysis.xpu_checker import check_xpu_implementation
from tools.xpu_triton_analysis.triton_checker import check_triton_coverage
from tools.xpu_triton_analysis.output_formatter import (
    write_json_output,
    write_csv_output,
)
from tools.xpu_triton_analysis.summary_report import generate_summary_report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch ops for XPU/Triton support"
    )
    parser.add_argument(
        "--native-functions",
        type=str,
        default="aten/src/ATen/native/native_functions.yaml",
        help="Path to native_functions.yaml (relative to PyTorch root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tools/xpu_triton_analysis/output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    # Resolve paths
    native_functions_path = PYTORCH_ROOT / args.native_functions
    output_dir = PYTORCH_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("XPU/Triton Op Analysis Tool")
    print("=" * 80)
    print(f"PyTorch Root: {PYTORCH_ROOT}")
    print(f"Native Functions: {native_functions_path}")
    print(f"Output Directory: {output_dir}")
    print()

    # Stage 1: Parse native_functions.yaml
    print("Stage 1: Parsing native_functions.yaml for pointwise/reduction ops...")
    ops = parse_native_functions(native_functions_path, verbose=args.verbose)
    print(f"  Found {len(ops)} ops with pointwise/reduction tags")
    print()

    # Stage 2: Check XPU implementations
    print("Stage 2: Checking XPU implementation status...")
    ops = check_xpu_implementation(PYTORCH_ROOT, ops, verbose=args.verbose)
    xpu_count = sum(1 for op in ops if op["has_xpu_impl"]["final"])
    print(f"  Found {xpu_count} ops with XPU implementation")
    print()

    # Stage 3: Check Triton/Inductor coverage
    print("Stage 3: Checking Triton/Inductor coverage...")
    ops = check_triton_coverage(PYTORCH_ROOT, ops, verbose=args.verbose)
    triton_count = sum(
        1
        for op in ops
        if op["triton_coverage"]["status"] in ["yes", "likely_yes"]
    )
    print(f"  Found {triton_count} ops with Triton coverage")
    print()

    # Write outputs
    print("Writing output files...")
    if args.format in ["json", "both"]:
        json_path = output_dir / "ops_analysis.json"
        write_json_output(ops, json_path)
        print(f"  JSON: {json_path}")

    if args.format in ["csv", "both"]:
        csv_path = output_dir / "ops_analysis.csv"
        write_csv_output(ops, csv_path)
        print(f"  CSV: {csv_path}")

    # Always generate summary report
    summary_path = output_dir / "summary_report.md"
    generate_summary_report(ops, summary_path)
    print(f"  Summary: {summary_path}")

    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Total ops analyzed: {len(ops)}")
    print(f"  - Pointwise: {sum(1 for op in ops if op['category'] == 'pointwise')}")
    print(f"  - Reduction: {sum(1 for op in ops if op['category'] == 'reduction')}")
    print(f"Ops with XPU impl: {xpu_count}")
    print(f"Ops with Triton coverage: {triton_count}")
    print(
        f"Ops missing both XPU and Triton: {sum(1 for op in ops if not op['has_xpu_impl']['final'] and op['triton_coverage']['status'] not in ['yes', 'likely_yes'])}"
    )


if __name__ == "__main__":
    main()
