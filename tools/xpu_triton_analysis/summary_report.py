"""
Summary Report Generator

Generates a human-readable summary report highlighting key findings from the analysis.
"""

from pathlib import Path
from typing import Dict, List, Any


def generate_summary_report(ops: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Generate a summary report in Markdown format.

    Args:
        ops: List of op records
        output_path: Path to output markdown file
    """
    # Calculate statistics
    total_ops = len(ops)
    pointwise_ops = [op for op in ops if op["category"] == "pointwise"]
    reduction_ops = [op for op in ops if op["category"] == "reduction"]

    ops_with_xpu = [op for op in ops if op["has_xpu_impl"]["final"]]
    ops_without_xpu = [op for op in ops if not op["has_xpu_impl"]["final"]]

    ops_with_triton = [
        op for op in ops if op["triton_coverage"]["status"] in ["yes", "likely_yes"]
    ]
    ops_without_triton = [
        op
        for op in ops
        if op["triton_coverage"]["status"] not in ["yes", "likely_yes"]
    ]

    # Ops missing both XPU and Triton (high priority)
    high_priority_ops = [
        op
        for op in ops
        if not op["has_xpu_impl"]["final"]
        and op["triton_coverage"]["status"] not in ["yes", "likely_yes"]
    ]

    # Ops with Triton but no XPU (medium priority)
    medium_priority_ops = [
        op
        for op in ops
        if not op["has_xpu_impl"]["final"]
        and op["triton_coverage"]["status"] in ["yes", "likely_yes"]
    ]

    with open(output_path, "w") as f:
        f.write("# XPU/Triton Op Analysis Summary Report\n\n")

        # Overview statistics
        f.write("## Overview\n\n")
        f.write(f"- **Total ops analyzed**: {total_ops}\n")
        f.write(f"  - Pointwise: {len(pointwise_ops)}\n")
        f.write(f"  - Reduction: {len(reduction_ops)}\n\n")

        f.write(f"- **Ops with XPU implementation**: {len(ops_with_xpu)}\n")
        f.write(f"- **Ops without XPU implementation**: {len(ops_without_xpu)}\n\n")

        f.write(f"- **Ops with Triton coverage**: {len(ops_with_triton)}\n")
        f.write(
            f"- **Ops without Triton coverage**: {len(ops_without_triton)}\n\n"
        )

        # Priority breakdown
        f.write("## Priority Breakdown\n\n")
        f.write(
            f"- **High Priority** (Missing both XPU and Triton): {len(high_priority_ops)}\n"
        )
        f.write(
            f"- **Medium Priority** (Has Triton but missing XPU): {len(medium_priority_ops)}\n"
        )
        f.write(
            f"- **Low Priority** (Has XPU implementation): {len(ops_with_xpu)}\n\n"
        )

        # High priority ops (missing both)
        if high_priority_ops:
            f.write("## High Priority: Ops Missing Both XPU and Triton\n\n")
            f.write(
                "These ops lack both XPU implementation and clear Triton coverage. "
                "Consider implementing XPU kernels or verifying Triton support.\n\n"
            )

            # Group by category
            hp_pointwise = [op for op in high_priority_ops if op["category"] == "pointwise"]
            hp_reduction = [op for op in high_priority_ops if op["category"] == "reduction"]

            if hp_pointwise:
                f.write(f"### Pointwise Ops ({len(hp_pointwise)})\n\n")
                for op in hp_pointwise[:20]:  # Limit to first 20
                    f.write(f"- `{op['op_name']}`\n")
                    f.write(f"  - Schema: `{op['schema']}`\n")
                    f.write(f"  - Dispatch keys: {', '.join(op['dispatch_keys']) if op['dispatch_keys'] else 'None'}\n")
                if len(hp_pointwise) > 20:
                    f.write(f"\n... and {len(hp_pointwise) - 20} more\n")
                f.write("\n")

            if hp_reduction:
                f.write(f"### Reduction Ops ({len(hp_reduction)})\n\n")
                for op in hp_reduction[:20]:  # Limit to first 20
                    f.write(f"- `{op['op_name']}`\n")
                    f.write(f"  - Schema: `{op['schema']}`\n")
                    f.write(f"  - Dispatch keys: {', '.join(op['dispatch_keys']) if op['dispatch_keys'] else 'None'}\n")
                if len(hp_reduction) > 20:
                    f.write(f"\n... and {len(hp_reduction) - 20} more\n")
                f.write("\n")

        # Medium priority ops (has Triton but no XPU)
        if medium_priority_ops:
            f.write("## Medium Priority: Ops with Triton Coverage but Missing XPU\n\n")
            f.write(
                "These ops have Triton/Inductor coverage but lack XPU implementations. "
                "XPU could potentially leverage the Triton implementations.\n\n"
            )

            # Group by category
            mp_pointwise = [op for op in medium_priority_ops if op["category"] == "pointwise"]
            mp_reduction = [op for op in medium_priority_ops if op["category"] == "reduction"]

            f.write(f"- Pointwise ops: {len(mp_pointwise)}\n")
            f.write(f"- Reduction ops: {len(mp_reduction)}\n\n")

            # Show a sample
            f.write("### Sample Ops\n\n")
            for op in medium_priority_ops[:10]:
                f.write(f"- `{op['op_name']}` ({op['category']})\n")
            if len(medium_priority_ops) > 10:
                f.write(f"\n... and {len(medium_priority_ops) - 10} more\n")
            f.write("\n")

        # Ops with XPU
        if ops_with_xpu:
            f.write("## Low Priority: Ops with XPU Implementation\n\n")
            f.write(
                f"These {len(ops_with_xpu)} ops already have XPU implementations. "
                "No immediate action needed.\n\n"
            )

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("1. **Focus on high-priority ops first**: These lack both XPU and Triton support.\n")
        f.write("2. **For medium-priority ops**: Consider leveraging existing Triton implementations for XPU.\n")
        f.write("3. **Verify Triton coverage**: The 'likely_yes' status indicates probable coverage based on patterns, but manual verification is recommended.\n")
        f.write("4. **Cross-reference with torch-xpu-ops**: Compare with XPU's native_functions.yaml to identify additional gaps.\n\n")

        # Data files reference
        f.write("## Output Files\n\n")
        f.write("- **ops_analysis.json**: Complete analysis data in JSON format\n")
        f.write("- **ops_analysis.csv**: Simplified CSV format for spreadsheet analysis\n")
        f.write("- **summary_report.md**: This summary report\n\n")

        f.write("---\n\n")
        f.write("*Generated by XPU/Triton Op Analysis Tool*\n")
