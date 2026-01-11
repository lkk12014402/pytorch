# Usage Examples

This document provides practical examples of using the XPU/Triton Op Analysis Tool.

## Basic Usage

### Run full analysis with default settings

```bash
cd /path/to/pytorch
python tools/xpu_triton_analysis/analyze_ops.py
```

This will:
- Parse `aten/src/ATen/native/native_functions.yaml`
- Check for XPU implementations
- Check for Triton/Inductor coverage
- Generate JSON, CSV, and summary report in `tools/xpu_triton_analysis/output/`

## Command-Line Options

### Generate only JSON output

```bash
python tools/xpu_triton_analysis/analyze_ops.py --format json
```

### Generate only CSV output

```bash
python tools/xpu_triton_analysis/analyze_ops.py --format csv
```

### Use custom output directory

```bash
python tools/xpu_triton_analysis/analyze_ops.py --output-dir /tmp/xpu_analysis
```

### Enable verbose logging

```bash
python tools/xpu_triton_analysis/analyze_ops.py --verbose
```

This will show detailed progress including:
- Each op found during YAML parsing
- XPU registrations discovered
- Triton coverage determinations

### Use custom native_functions.yaml path

```bash
python tools/xpu_triton_analysis/analyze_ops.py \
  --native-functions /path/to/custom/native_functions.yaml
```

## Understanding the Output

### JSON Output Structure

The JSON output (`ops_analysis.json`) contains a list of ops with detailed information:

```json
{
  "ops": [
    {
      "op_name": "add.Tensor",
      "namespace": "aten",
      "schema": "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      "category": "pointwise",
      "tags": ["core", "pointwise"],
      "dispatch_keys": ["SparseCPU", "SparseCUDA", "MPS"],
      "has_xpu_impl": {
        "from_dispatch": false,
        "from_manual_impl": false,
        "final": false
      },
      "triton_coverage": {
        "status": "likely_yes",
        "source": ["inductor_pointwise"],
        "notes": "Pointwise op likely handled by..."
      }
    }
  ]
}
```

### CSV Output Structure

The CSV output (`ops_analysis.csv`) is simplified for spreadsheet analysis:

```csv
op_name,category,has_xpu_impl,triton_coverage_status,dispatch_keys,tags,notes
add.Tensor,pointwise,False,likely_yes,"[""SparseCPU"", ""CUDA""]","[""core"", ""pointwise""]","Pointwise op likely..."
```

### Summary Report

The summary report (`summary_report.md`) provides:
- Overview statistics
- Priority breakdown (high/medium/low)
- Detailed lists of high-priority ops
- Recommendations for next steps

## Common Workflows

### 1. Identify ops missing XPU implementation

```bash
# Run the analysis
python tools/xpu_triton_analysis/analyze_ops.py

# Check the summary report
cat tools/xpu_triton_analysis/output/summary_report.md

# Or filter CSV for ops without XPU
grep ",False," tools/xpu_triton_analysis/output/ops_analysis.csv
```

### 2. Find pointwise ops without XPU

```bash
# Using jq to query JSON
cat tools/xpu_triton_analysis/output/ops_analysis.json | \
  jq '.ops[] | select(.category == "pointwise" and .has_xpu_impl.final == false) | .op_name'
```

### 3. Find reduction ops without XPU

```bash
cat tools/xpu_triton_analysis/output/ops_analysis.json | \
  jq '.ops[] | select(.category == "reduction" and .has_xpu_impl.final == false) | .op_name'
```

### 4. Identify ops with explicit Triton lowerings

```bash
cat tools/xpu_triton_analysis/output/ops_analysis.json | \
  jq '.ops[] | select(.triton_coverage.status == "yes") | {op_name, source: .triton_coverage.source}'
```

### 5. Export specific fields to custom CSV

```bash
cat tools/xpu_triton_analysis/output/ops_analysis.json | \
  jq -r '.ops[] | [.op_name, .category, .has_xpu_impl.final, .triton_coverage.status] | @csv' > custom_report.csv
```

## Integration with Development Workflow

### Before adding a new XPU kernel

1. Run the analysis to check current status
2. Identify if the op already has Triton coverage
3. If yes, consider reusing the Triton implementation for XPU
4. If no, implement both XPU and Triton versions

### Tracking progress

```bash
# Run analysis before work
python tools/xpu_triton_analysis/analyze_ops.py --output-dir before/

# Implement XPU kernels...

# Run analysis after work
python tools/xpu_triton_analysis/analyze_ops.py --output-dir after/

# Compare results
diff before/ops_analysis.csv after/ops_analysis.csv
```

## Filtering and Sorting Results

### Sort CSV by category then op name

```bash
(head -n 1 tools/xpu_triton_analysis/output/ops_analysis.csv && \
 tail -n +2 tools/xpu_triton_analysis/output/ops_analysis.csv | sort -t, -k2,2 -k1,1) \
 > sorted_ops.csv
```

### Count ops by category

```bash
# Using awk on CSV
tail -n +2 tools/xpu_triton_analysis/output/ops_analysis.csv | \
  awk -F, '{print $2}' | sort | uniq -c
```

### Find ops with specific dispatch keys

```bash
# Find ops dispatched to CUDA
grep "CUDA" tools/xpu_triton_analysis/output/ops_analysis.csv
```

## Troubleshooting

### Issue: No ops found

**Problem**: Analysis shows "Found 0 ops"

**Solution**: Check that:
1. You're running from PyTorch root directory
2. `native_functions.yaml` path is correct
3. The YAML file contains ops with `pointwise` or `reduction` tags

### Issue: Cannot find XPU registrations

**Problem**: All ops show `has_xpu_impl: false`

**Solution**: This may be expected if:
1. XPU support hasn't been added to this PyTorch branch
2. XPU implementations are in a separate repository (torch-xpu-ops)
3. The grep search for TORCH_LIBRARY_IMPL needs adjustment

### Issue: Permission denied when writing output

**Problem**: Cannot write to output directory

**Solution**: 
```bash
# Use a different output directory
python tools/xpu_triton_analysis/analyze_ops.py --output-dir /tmp/xpu_output
```

## Advanced Usage

### Programmatic usage in Python

```python
from pathlib import Path
from tools.xpu_triton_analysis.yaml_parser import parse_native_functions
from tools.xpu_triton_analysis.xpu_checker import check_xpu_implementation
from tools.xpu_triton_analysis.triton_checker import check_triton_coverage

pytorch_root = Path("/path/to/pytorch")
yaml_path = pytorch_root / "aten/src/ATen/native/native_functions.yaml"

# Stage 1: Parse YAML
ops = parse_native_functions(yaml_path, verbose=True)

# Stage 2: Check XPU
ops = check_xpu_implementation(pytorch_root, ops, verbose=True)

# Stage 3: Check Triton
ops = check_triton_coverage(pytorch_root, ops, verbose=True)

# Custom analysis
pointwise_without_xpu = [
    op for op in ops 
    if op["category"] == "pointwise" and not op["has_xpu_impl"]["final"]
]
print(f"Found {len(pointwise_without_xpu)} pointwise ops without XPU")
```

## See Also

- [README.md](README.md) - Tool overview and documentation
- [../../aten/src/ATen/native/README.md](../../aten/src/ATen/native/README.md) - ATen operator implementation guide
- PyTorch Inductor documentation
