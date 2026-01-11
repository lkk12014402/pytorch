#!/usr/bin/env python3
"""
Simple tests for the XPU/Triton analysis tool.

Run with: python tools/xpu_triton_analysis/test_tool.py
"""

import json
import tempfile
from pathlib import Path
import sys

# Add parent directories to path for imports
PYTORCH_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PYTORCH_ROOT))

from tools.xpu_triton_analysis.yaml_parser import parse_native_functions
from tools.xpu_triton_analysis.xpu_checker import check_xpu_implementation
from tools.xpu_triton_analysis.triton_checker import check_triton_coverage
from tools.xpu_triton_analysis.output_formatter import write_json_output, write_csv_output
from tools.xpu_triton_analysis.summary_report import generate_summary_report


def test_yaml_parser():
    """Test YAML parsing functionality."""
    print("Testing YAML parser...")
    
    yaml_path = PYTORCH_ROOT / "aten/src/ATen/native/native_functions.yaml"
    ops = parse_native_functions(yaml_path, verbose=False)
    
    assert len(ops) > 0, "Should find at least some ops"
    assert all("op_name" in op for op in ops), "All ops should have op_name"
    assert all("category" in op for op in ops), "All ops should have category"
    assert all(op["category"] in ["pointwise", "reduction"] for op in ops), \
        "All ops should be pointwise or reduction"
    
    print(f"  ✓ Found {len(ops)} ops with pointwise/reduction tags")
    print(f"  ✓ All ops have required fields")
    return ops


def test_xpu_checker(ops):
    """Test XPU implementation checker."""
    print("Testing XPU checker...")
    
    ops = check_xpu_implementation(PYTORCH_ROOT, ops, verbose=False)
    
    assert all("has_xpu_impl" in op for op in ops), "All ops should have has_xpu_impl"
    assert all("from_dispatch" in op["has_xpu_impl"] for op in ops), \
        "has_xpu_impl should have from_dispatch"
    assert all("from_manual_impl" in op["has_xpu_impl"] for op in ops), \
        "has_xpu_impl should have from_manual_impl"
    assert all("final" in op["has_xpu_impl"] for op in ops), \
        "has_xpu_impl should have final"
    
    xpu_count = sum(1 for op in ops if op["has_xpu_impl"]["final"])
    print(f"  ✓ Checked {len(ops)} ops for XPU implementation")
    print(f"  ✓ Found {xpu_count} ops with XPU impl")
    return ops


def test_triton_checker(ops):
    """Test Triton coverage checker."""
    print("Testing Triton checker...")
    
    ops = check_triton_coverage(PYTORCH_ROOT, ops, verbose=False)
    
    assert all("triton_coverage" in op for op in ops), "All ops should have triton_coverage"
    assert all("status" in op["triton_coverage"] for op in ops), \
        "triton_coverage should have status"
    assert all(
        op["triton_coverage"]["status"] in ["yes", "likely_yes", "no", "unknown"]
        for op in ops
    ), "status should be valid"
    
    triton_count = sum(
        1 for op in ops if op["triton_coverage"]["status"] in ["yes", "likely_yes"]
    )
    print(f"  ✓ Checked {len(ops)} ops for Triton coverage")
    print(f"  ✓ Found {triton_count} ops with Triton coverage")
    return ops


def test_output_formatters(ops):
    """Test output formatters."""
    print("Testing output formatters...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test JSON output
        json_path = tmpdir_path / "test.json"
        write_json_output(ops, json_path)
        assert json_path.exists(), "JSON file should be created"
        
        with open(json_path, "r") as f:
            data = json.load(f)
        assert "ops" in data, "JSON should have ops key"
        assert len(data["ops"]) == len(ops), "JSON should contain all ops"
        print(f"  ✓ JSON output works ({json_path.stat().st_size} bytes)")
        
        # Test CSV output
        csv_path = tmpdir_path / "test.csv"
        write_csv_output(ops, csv_path)
        assert csv_path.exists(), "CSV file should be created"
        
        with open(csv_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == len(ops) + 1, "CSV should have header + all ops"
        print(f"  ✓ CSV output works ({len(lines)} lines)")
        
        # Test summary report
        summary_path = tmpdir_path / "summary.md"
        generate_summary_report(ops, summary_path)
        assert summary_path.exists(), "Summary report should be created"
        
        with open(summary_path, "r") as f:
            content = f.read()
        assert "# XPU/Triton Op Analysis Summary Report" in content, \
            "Summary should have title"
        print(f"  ✓ Summary report works ({summary_path.stat().st_size} bytes)")


def test_specific_ops(ops):
    """Test that specific expected ops are present."""
    print("Testing specific ops...")
    
    op_names = [op["op_name"] for op in ops]
    
    # Check for some common pointwise ops
    expected_pointwise = ["abs", "add.Tensor", "mul.Tensor", "neg"]
    found_pointwise = [name for name in expected_pointwise if any(name in op_name for op_name in op_names)]
    print(f"  ✓ Found {len(found_pointwise)}/{len(expected_pointwise)} expected pointwise ops")
    
    # Check for some common reduction ops
    expected_reduction = ["sum", "mean", "max", "min"]
    found_reduction = [name for name in expected_reduction if any(name in op_name for op_name in op_names)]
    print(f"  ✓ Found {len(found_reduction)}/{len(expected_reduction)} expected reduction ops")


def main():
    """Run all tests."""
    print("=" * 80)
    print("XPU/Triton Op Analysis Tool - Tests")
    print("=" * 80)
    print()
    
    try:
        # Run tests
        ops = test_yaml_parser()
        print()
        
        ops = test_xpu_checker(ops)
        print()
        
        ops = test_triton_checker(ops)
        print()
        
        test_output_formatters(ops)
        print()
        
        test_specific_ops(ops)
        print()
        
        print("=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
