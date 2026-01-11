# XPU/Triton Op Analysis Tool - Quick Start Guide

## Overview

This tool implements the three-stage workflow for analyzing PyTorch operators to identify XPU/Triton development priorities. It analyzes 640 ops with `pointwise` or `reduction` tags from `native_functions.yaml`.

## Quick Start

```bash
# Navigate to PyTorch root
cd /path/to/pytorch

# Run the analysis
python tools/xpu_triton_analysis/analyze_ops.py

# View results
ls -lh tools/xpu_triton_analysis/output/
# ops_analysis.json (~500KB) - Complete structured data
# ops_analysis.csv  (~200KB) - Spreadsheet format
# summary_report.md (~2KB)   - Human-readable summary
```

## Example Output

### Summary Statistics

From a recent run:
```
Total ops analyzed: 640
  - Pointwise: 538
  - Reduction: 102

Ops with XPU implementation: 0
Ops with Triton coverage: 640

Priority Breakdown:
  - High Priority (Missing both): 0
  - Medium Priority (Has Triton, missing XPU): 640
  - Low Priority (Has XPU): 0
```

### Sample Data

**Pointwise Ops:**
```
op_name                   category    has_xpu_impl   triton_coverage
abs                       pointwise   False          likely_yes
add.Tensor                pointwise   False          likely_yes
mul.Tensor                pointwise   False          likely_yes
```

**Reduction Ops:**
```
op_name                   category    has_xpu_impl   triton_coverage
sum.dim_IntList           reduction   False          likely_yes
mean.dim                  reduction   False          likely_yes
max.dim                   reduction   False          likely_yes
```

## Key Features

### 1. Comprehensive Op Discovery
- Parses `native_functions.yaml` using PyYAML
- Filters for `pointwise` and `reduction` tags
- Extracts complete schema and metadata

### 2. XPU Implementation Detection
- Checks `dispatch:` keys for XPU entries
- Searches for `TORCH_LIBRARY_IMPL(aten, XPU, m)` registrations
- Combines both sources for final status

### 3. Triton Coverage Analysis
- Analyzes Inductor's generic pointwise/reduction handling
- Checks for explicit lowerings in lowering table
- Determines coverage based on code patterns

### 4. Multiple Output Formats
- **JSON**: Complete data for programmatic use
- **CSV**: Easy filtering and sorting in spreadsheets
- **Markdown**: Human-readable summary report

## Common Queries

### Find all ops without XPU
```bash
grep ",False," tools/xpu_triton_analysis/output/ops_analysis.csv | wc -l
```

### Find pointwise ops without XPU
```bash
grep ",pointwise,False," tools/xpu_triton_analysis/output/ops_analysis.csv | head -10
```

### Find reduction ops without XPU
```bash
grep ",reduction,False," tools/xpu_triton_analysis/output/ops_analysis.csv | head -10
```

### Get specific op info
```bash
cat tools/xpu_triton_analysis/output/ops_analysis.json | \
  jq '.ops[] | select(.op_name == "add.Tensor")'
```

## Advanced Usage

### Enable verbose logging
```bash
python tools/xpu_triton_analysis/analyze_ops.py --verbose
```

### Custom output directory
```bash
python tools/xpu_triton_analysis/analyze_ops.py --output-dir /tmp/analysis
```

### Generate only JSON
```bash
python tools/xpu_triton_analysis/analyze_ops.py --format json
```

## Workflow Integration

### Typical Developer Workflow

1. **Run analysis to get baseline**
   ```bash
   python tools/xpu_triton_analysis/analyze_ops.py
   ```

2. **Identify high-priority ops**
   - Check `summary_report.md` for overview
   - Filter CSV for ops without XPU implementation
   - Prioritize based on usage frequency

3. **Implement XPU kernels**
   - Add XPU dispatch entries in `native_functions.yaml`
   - Implement kernels in `aten/src/ATen/native/xpu/`
   - Or add manual registrations with `TORCH_LIBRARY_IMPL`

4. **Re-run analysis to verify**
   ```bash
   python tools/xpu_triton_analysis/analyze_ops.py --output-dir after/
   ```

5. **Compare results**
   ```bash
   diff tools/xpu_triton_analysis/output/ops_analysis.csv after/ops_analysis.csv
   ```

### Cross-Reference with torch-xpu-ops

The tool currently checks the main PyTorch repository. To compare with torch-xpu-ops:

1. Run this tool on PyTorch main
2. Run similar analysis on torch-xpu-ops (if available)
3. Diff the results to find gaps
4. Prioritize ops that:
   - Are in PyTorch main
   - Are NOT in torch-xpu-ops
   - Have high usage in real workloads

## Understanding Results

### XPU Implementation Status

- **from_dispatch**: Found in `dispatch:` keys (e.g., `XPU: my_kernel`)
- **from_manual_impl**: Found in `TORCH_LIBRARY_IMPL(aten, XPU, m)`
- **final**: Combined result (has XPU if either is true)

### Triton Coverage Status

- **yes**: Explicit lowering found in Inductor's lowering table
- **likely_yes**: Covered by generic pointwise/reduction Triton codegen
- **no**: Not covered by Triton (uses ATen fallback)
- **unknown**: Could not determine coverage

## Testing

Run automated tests:
```bash
python tools/xpu_triton_analysis/test_tool.py
```

Expected output:
```
✓ Found 640 ops with pointwise/reduction tags
✓ Checked 640 ops for XPU implementation
✓ Checked 640 ops for Triton coverage
✓ JSON output works
✓ CSV output works
✓ Summary report works
✓ All tests passed!
```

## Documentation

- **README.md**: Tool overview and architecture
- **EXAMPLES.md**: Detailed usage examples and workflows
- **QUICKSTART.md**: This quick start guide
- **output/README.md**: Information about generated files

## Troubleshooting

### Issue: Module not found
**Solution**: Make sure you're running from PyTorch root directory

### Issue: YAML parsing error
**Solution**: Ensure `pyyaml` is installed: `pip install pyyaml`

### Issue: No ops found
**Solution**: Verify `native_functions.yaml` path is correct

### Issue: Grep errors in XPU checker
**Solution**: This may happen on non-Unix systems. The tool will continue with limited XPU detection.

## Next Steps

After running the analysis:

1. Review `summary_report.md` for high-level insights
2. Filter `ops_analysis.csv` for ops relevant to your work
3. Use `ops_analysis.json` for programmatic analysis
4. Prioritize based on:
   - Usage frequency in real models
   - Performance impact
   - Complexity of implementation

## Support

For issues or questions about the tool:
- Check the documentation in this directory
- Review the source code (well-commented)
- Run tests to verify your environment

The tool is designed to be self-contained and requires no external dependencies beyond standard PyTorch development tools.
