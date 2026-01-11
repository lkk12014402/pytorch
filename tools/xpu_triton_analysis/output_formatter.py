"""
Output Formatter

Formats analysis results into JSON and CSV files.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def write_json_output(ops: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write analysis results to JSON file.

    Args:
        ops: List of op records
        output_path: Path to output JSON file
    """
    output = {"ops": ops}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=False)


def write_csv_output(ops: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write analysis results to CSV file.

    Args:
        ops: List of op records
        output_path: Path to output CSV file
    """
    # Define CSV columns
    fieldnames = [
        "op_name",
        "category",
        "has_xpu_impl",
        "triton_coverage_status",
        "dispatch_keys",
        "tags",
        "notes",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for op in ops:
            row = {
                "op_name": op["op_name"],
                "category": op["category"],
                "has_xpu_impl": op["has_xpu_impl"]["final"],
                "triton_coverage_status": op["triton_coverage"]["status"],
                "dispatch_keys": json.dumps(op["dispatch_keys"]),
                "tags": json.dumps(op["tags"]),
                "notes": op["triton_coverage"]["notes"],
            }
            writer.writerow(row)
