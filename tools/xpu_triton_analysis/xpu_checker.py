"""
XPU Implementation Checker

Checks whether ops have XPU implementations by:
1. Looking for XPU in dispatch keys
2. Searching for TORCH_LIBRARY_IMPL(aten, XPU, m) registrations
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set


def find_xpu_registrations(pytorch_root: Path, verbose: bool = False) -> Set[str]:
    """
    Search for XPU op registrations in the codebase.

    Returns:
        Set of op names registered for XPU
    """
    registered_ops = set()

    # Search for TORCH_LIBRARY_IMPL blocks with XPU
    # We'll use grep to find relevant files first
    try:
        result = subprocess.run(
            [
                "grep",
                "-r",
                "-l",
                "TORCH_LIBRARY_IMPL.*XPU",
                str(pytorch_root / "aten"),
                str(pytorch_root / "torch"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        if verbose:
            print("  Warning: Could not search for XPU registrations")
        return registered_ops

    # Parse each file to extract registered op names
    for filepath in files:
        if not filepath:
            continue

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Look for TORCH_LIBRARY_IMPL(aten, XPU, m) blocks
            # and extract m.impl("op_name", ...) calls
            in_xpu_block = False
            for line in content.split("\n"):
                # Check for start of XPU library impl block
                if re.search(r"TORCH_LIBRARY_IMPL\s*\(\s*aten\s*,\s*XPU", line):
                    in_xpu_block = True
                    continue

                # Check for end of block (closing brace at start of line)
                if in_xpu_block and re.match(r"^\s*\}\s*$", line):
                    in_xpu_block = False
                    continue

                # Extract op registrations
                if in_xpu_block:
                    # Look for m.impl("aten::op_name", ...) or m.impl("op_name", ...)
                    impl_match = re.search(
                        r'm\.impl\s*\(\s*"(?:aten::)?([a-zA-Z_][a-zA-Z0-9_.]*)"',
                        line,
                    )
                    if impl_match:
                        op_name = impl_match.group(1)
                        registered_ops.add(op_name)
                        if verbose:
                            print(f"    Found XPU registration: {op_name} in {filepath}")

        except Exception as e:
            if verbose:
                print(f"  Warning: Error reading {filepath}: {e}")
            continue

    return registered_ops


def check_xpu_implementation(
    pytorch_root: Path, ops: List[Dict[str, Any]], verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Check XPU implementation status for each op.

    Args:
        pytorch_root: Path to PyTorch root directory
        ops: List of op records from Stage 1
        verbose: Enable verbose logging

    Returns:
        Updated list of op records with has_xpu_impl filled in
    """
    # Find all XPU registrations
    if verbose:
        print("  Searching for XPU registrations...")
    xpu_registered_ops = find_xpu_registrations(pytorch_root, verbose)
    if verbose:
        print(f"  Found {len(xpu_registered_ops)} XPU-registered ops")

    # Check each op
    for op in ops:
        # Check dispatch keys
        dispatch_keys = op.get("dispatch_keys", [])
        has_xpu_dispatch = any(
            "XPU" in key or "Xpu" in key for key in dispatch_keys
        )

        # Check manual registrations
        # Try matching with and without overload suffix
        op_name = op["op_name"]
        op_base = op_name.split(".")[0]  # Remove overload suffix

        has_manual_impl = (
            op_name in xpu_registered_ops or op_base in xpu_registered_ops
        )

        # Update op record
        op["has_xpu_impl"]["from_dispatch"] = has_xpu_dispatch
        op["has_xpu_impl"]["from_manual_impl"] = has_manual_impl
        op["has_xpu_impl"]["final"] = has_xpu_dispatch or has_manual_impl

        if verbose and op["has_xpu_impl"]["final"]:
            print(
                f"  {op_name}: XPU impl found "
                f"(dispatch={has_xpu_dispatch}, manual={has_manual_impl})"
            )

    return ops
