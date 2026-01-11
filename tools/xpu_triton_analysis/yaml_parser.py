"""
YAML Parser for native_functions.yaml

Extracts ops with pointwise or reduction tags from native_functions.yaml.
"""

import re
from pathlib import Path
from typing import Dict, List, Any
import yaml


def parse_native_functions(yaml_path: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Parse native_functions.yaml and extract ops with pointwise/reduction tags.

    Args:
        yaml_path: Path to native_functions.yaml
        verbose: Enable verbose logging

    Returns:
        List of op dictionaries with basic information
    """
    with open(yaml_path, "r") as f:
        content = yaml.safe_load(f)

    ops = []
    for entry in content:
        if not isinstance(entry, dict) or "func" not in entry:
            continue

        # Extract function signature
        func_str = entry["func"]
        tags = entry.get("tags", [])

        # Convert single tag to list
        if isinstance(tags, str):
            tags = [tags]
        elif tags is None:
            tags = []

        # Check if op has pointwise or reduction tag
        has_pointwise = "pointwise" in tags
        has_reduction = "reduction" in tags

        if not (has_pointwise or has_reduction):
            continue

        # Determine category
        if has_pointwise:
            category = "pointwise"
        elif has_reduction:
            category = "reduction"
        else:
            category = "other"

        # Extract op name from function signature
        # Format: "name.overload(args) -> return_type" or "name(args) -> return_type"
        match = re.match(r"([a-zA-Z_][a-zA-Z0-9_.]*)(?:\(|$)", func_str)
        if not match:
            if verbose:
                print(f"  Warning: Could not parse op name from: {func_str}")
            continue

        op_name = match.group(1)

        # Extract dispatch keys
        dispatch_entry = entry.get("dispatch", {})
        if isinstance(dispatch_entry, dict):
            dispatch_keys = list(dispatch_entry.keys())
        else:
            dispatch_keys = []

        # Create op record
        op_record = {
            "op_name": op_name,
            "namespace": "aten",
            "schema": func_str,
            "category": category,
            "tags": tags,
            "dispatch_keys": dispatch_keys,
            # These will be filled in later stages
            "has_xpu_impl": {
                "from_dispatch": False,
                "from_manual_impl": False,
                "final": False,
            },
            "triton_coverage": {
                "status": "unknown",
                "source": [],
                "notes": "",
            },
        }

        ops.append(op_record)

        if verbose:
            print(f"  Found: {op_name} ({category})")

    return ops
