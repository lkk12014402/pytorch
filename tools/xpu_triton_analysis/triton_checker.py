"""
Triton/Inductor Coverage Checker

Determines if ops are covered by Inductor's Triton codegen paths by analyzing:
1. General pointwise/reduction handling in Inductor
2. Specific lowerings in the lowering table
3. Triton kernel implementations
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Set


def analyze_inductor_pointwise(pytorch_root: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze Inductor's pointwise handling to determine general coverage rules.

    Returns:
        Dictionary with analysis results
    """
    result = {
        "handles_generic_pointwise": False,
        "uses_triton": False,
        "notes": "",
    }

    # Check if pointwise operations are handled generically
    lowering_path = pytorch_root / "torch" / "_inductor" / "lowering.py"
    if not lowering_path.exists():
        result["notes"] = "lowering.py not found"
        return result

    try:
        with open(lowering_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for evidence of generic pointwise handling
        # Check for patterns that indicate Triton usage for pointwise ops
        triton_patterns = [
            r"import.*triton",
            r"@pointwise",
            r"def pointwise",
            r"triton.*kernel",
        ]

        pointwise_patterns = [
            r"def.*pointwise",
            r"pointwise_binary",
            r"pointwise_unary",
            r"TensorIterator",
        ]

        has_triton = any(re.search(pattern, content, re.IGNORECASE) for pattern in triton_patterns)
        has_pointwise = any(re.search(pattern, content, re.IGNORECASE) for pattern in pointwise_patterns)

        result["handles_generic_pointwise"] = has_pointwise
        result["uses_triton"] = has_triton

        if has_pointwise and has_triton:
            result["notes"] = (
                "Inductor handles pointwise ops generically and uses Triton for GPU codegen. "
                "Most pointwise ops are likely covered."
            )
        elif has_pointwise:
            result["notes"] = (
                "Inductor handles pointwise ops but Triton usage not clearly detected."
            )
        else:
            result["notes"] = "Could not determine pointwise handling pattern."

    except Exception as e:
        result["notes"] = f"Error analyzing lowering.py: {e}"

    return result


def analyze_inductor_reduction(pytorch_root: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze Inductor's reduction handling.

    Returns:
        Dictionary with analysis results
    """
    result = {
        "handles_generic_reduction": False,
        "uses_triton": False,
        "notes": "",
    }

    lowering_path = pytorch_root / "torch" / "_inductor" / "lowering.py"
    if not lowering_path.exists():
        result["notes"] = "lowering.py not found"
        return result

    try:
        with open(lowering_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for reduction handling
        reduction_patterns = [
            r"def.*reduction",
            r"@reduction",
            r"reduce_sum",
            r"reduce_mean",
            r"Reduction",
        ]

        triton_patterns = [
            r"import.*triton",
            r"triton.*kernel",
        ]

        has_reduction = any(re.search(pattern, content, re.IGNORECASE) for pattern in reduction_patterns)
        has_triton = any(re.search(pattern, content, re.IGNORECASE) for pattern in triton_patterns)

        result["handles_generic_reduction"] = has_reduction
        result["uses_triton"] = has_triton

        if has_reduction and has_triton:
            result["notes"] = (
                "Inductor handles reduction ops and uses Triton for GPU codegen. "
                "Many reduction ops are likely covered."
            )
        elif has_reduction:
            result["notes"] = (
                "Inductor handles reduction ops but Triton usage not clearly detected."
            )
        else:
            result["notes"] = "Could not determine reduction handling pattern."

    except Exception as e:
        result["notes"] = f"Error analyzing reduction handling: {e}"

    return result


def find_explicitly_lowered_ops(pytorch_root: Path, verbose: bool = False) -> Set[str]:
    """
    Find ops that have explicit lowerings in Inductor.

    Returns:
        Set of op names with explicit lowerings
    """
    lowered_ops = set()

    lowering_path = pytorch_root / "torch" / "_inductor" / "lowering.py"
    if not lowering_path.exists():
        return lowered_ops

    try:
        with open(lowering_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for @register_lowering decorators or similar patterns
        # Pattern: @register_lowering(aten.op_name) or similar
        lowering_patterns = [
            r'@register_lowering\s*\(\s*aten\.([a-zA-Z_][a-zA-Z0-9_]*)',
            r'register_lowering\s*\(\s*aten\.([a-zA-Z_][a-zA-Z0-9_]*)',
            r'lowerings\[aten\.([a-zA-Z_][a-zA-Z0-9_]*)\]',
        ]

        for pattern in lowering_patterns:
            for match in re.finditer(pattern, content):
                op_name = match.group(1)
                lowered_ops.add(op_name)
                if verbose:
                    print(f"    Found explicit lowering: {op_name}")

    except Exception as e:
        if verbose:
            print(f"  Warning: Error analyzing lowerings: {e}")

    return lowered_ops


def check_triton_coverage(
    pytorch_root: Path, ops: List[Dict[str, Any]], verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Check Triton/Inductor coverage for each op.

    Args:
        pytorch_root: Path to PyTorch root directory
        ops: List of op records from Stage 2
        verbose: Enable verbose logging

    Returns:
        Updated list of op records with triton_coverage filled in
    """
    # Analyze general Inductor patterns
    if verbose:
        print("  Analyzing Inductor pointwise handling...")
    pointwise_analysis = analyze_inductor_pointwise(pytorch_root, verbose)

    if verbose:
        print("  Analyzing Inductor reduction handling...")
    reduction_analysis = analyze_inductor_reduction(pytorch_root, verbose)

    if verbose:
        print("  Finding explicitly lowered ops...")
    explicitly_lowered = find_explicitly_lowered_ops(pytorch_root, verbose)
    if verbose:
        print(f"  Found {len(explicitly_lowered)} explicitly lowered ops")

    # Check each op
    for op in ops:
        category = op["category"]
        op_name = op["op_name"]
        op_base = op_name.split(".")[0]  # Remove overload suffix

        # Check if explicitly lowered
        if op_name in explicitly_lowered or op_base in explicitly_lowered:
            op["triton_coverage"]["status"] = "yes"
            op["triton_coverage"]["source"] = ["inductor_lowering_table"]
            op["triton_coverage"]["notes"] = (
                f"Op {op_name} has explicit lowering in Inductor's lowering table."
            )
        # Check based on category and general patterns
        elif category == "pointwise":
            if pointwise_analysis["handles_generic_pointwise"] and pointwise_analysis["uses_triton"]:
                op["triton_coverage"]["status"] = "likely_yes"
                op["triton_coverage"]["source"] = ["inductor_pointwise"]
                op["triton_coverage"]["notes"] = (
                    "Pointwise op likely handled by Inductor's generic pointwise path with Triton on CUDA. "
                    + pointwise_analysis["notes"]
                )
            else:
                op["triton_coverage"]["status"] = "unknown"
                op["triton_coverage"]["source"] = []
                op["triton_coverage"]["notes"] = (
                    "Pointwise op but could not confirm Triton coverage. "
                    + pointwise_analysis["notes"]
                )
        elif category == "reduction":
            if reduction_analysis["handles_generic_reduction"] and reduction_analysis["uses_triton"]:
                op["triton_coverage"]["status"] = "likely_yes"
                op["triton_coverage"]["source"] = ["inductor_reduction"]
                op["triton_coverage"]["notes"] = (
                    "Reduction op likely handled by Inductor's reduction path with Triton on CUDA. "
                    + reduction_analysis["notes"]
                )
            else:
                op["triton_coverage"]["status"] = "unknown"
                op["triton_coverage"]["source"] = []
                op["triton_coverage"]["notes"] = (
                    "Reduction op but could not confirm Triton coverage. "
                    + reduction_analysis["notes"]
                )
        else:
            op["triton_coverage"]["status"] = "unknown"
            op["triton_coverage"]["source"] = []
            op["triton_coverage"]["notes"] = "Op category not clearly pointwise or reduction."

        if verbose and op["triton_coverage"]["status"] in ["yes", "likely_yes"]:
            print(f"  {op_name}: Triton coverage detected ({op['triton_coverage']['status']})")

    return ops
