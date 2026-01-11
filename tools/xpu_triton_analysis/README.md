# XPU/Triton Op Analysis Tool

This tool analyzes PyTorch ATen operators to identify which ones need XPU and/or Triton implementations.

## Purpose

The tool helps prioritize Triton kernel development for XPU by:
1. Identifying ops with `pointwise` or `reduction` tags in `native_functions.yaml`
2. Checking whether each op has XPU implementation (from dispatch keys or manual registration)
3. Determining if the op is covered by Inductor's Triton codegen paths

This tool implements the workflow described for identifying XPU Triton high-priority operations by:
- Parsing PyTorch's `native_functions.yaml` to find ops with specific tags
- Cross-referencing with XPU implementations
- Checking Triton/Inductor coverage
- Generating reports in JSON, CSV, and Markdown formats

## Usage

```bash
# Run the analysis
cd /path/to/pytorch
python tools/xpu_triton_analysis/analyze_ops.py

# Output files will be generated in tools/xpu_triton_analysis/output/
# - ops_analysis.json (detailed JSON format)
# - ops_analysis.csv (simplified CSV format)
```

## Output Schema

### JSON Format

Each op entry contains:
- `op_name`: Function name with overload (e.g., "add.Tensor")
- `namespace`: Typically "aten"
- `schema`: Complete function signature
- `category`: "pointwise" or "reduction"
- `tags`: Array of tags from native_functions.yaml
- `dispatch_keys`: List of dispatch keys (CPU, CUDA, MPS, XPU, etc.)
- `has_xpu_impl`: Object with:
  - `from_dispatch`: Boolean, if XPU key in dispatch
  - `from_manual_impl`: Boolean, if found in TORCH_LIBRARY_IMPL
  - `final`: Boolean, combined result
- `triton_coverage`: Object with:
  - `status`: "yes", "likely_yes", "no", or "unknown"
  - `source`: Array of sources (e.g., ["inductor_pointwise"])
  - `notes`: Human-readable explanation

### CSV Format

Simplified columns:
- op_name
- category
- has_xpu_impl
- triton_coverage_status
- dispatch_keys
- tags
- notes

## Implementation Stages

1. **Stage 1**: Parse native_functions.yaml for pointwise/reduction ops
2. **Stage 2**: Determine XPU implementation status
3. **Stage 3**: Determine Triton/Inductor coverage

## High-Priority Ops for XPU Triton Development

Based on the analysis, ops missing from XPU but present in PyTorch main include:
- dot / vdot / grouped_mm / mixed_dtypes_linear
- scaled_dot_product_attention / flash_attention / efficient_attention
- fused RMSNorm / fused Adagrad
- semi-structured sparse ops (sspaddmm, etc.)

Note: The current tool focuses on ops with `pointwise` or `reduction` tags. Some high-priority ops like attention mechanisms may not have these tags and would need separate analysis.

## Implementation Details

### Three-Stage Pipeline

1. **Stage 1: Parse YAML** (`yaml_parser.py`)
   - Reads `native_functions.yaml`
   - Filters ops with `pointwise` or `reduction` tags
   - Extracts basic metadata (name, schema, dispatch keys)

2. **Stage 2: Check XPU Implementation** (`xpu_checker.py`)
   - Checks dispatch keys for XPU entries
   - Searches codebase for `TORCH_LIBRARY_IMPL(aten, XPU, m)` registrations
   - Combines results to determine final XPU support status

3. **Stage 3: Check Triton Coverage** (`triton_checker.py`)
   - Analyzes Inductor's pointwise/reduction handling
   - Checks for explicit lowerings in lowering table
   - Determines if ops are covered by generic Triton codegen

### Output Formats

- **JSON** (`ops_analysis.json`): Complete structured data
- **CSV** (`ops_analysis.csv`): Spreadsheet-friendly format
- **Markdown** (`summary_report.md`): Human-readable summary with priorities

## Files

- `analyze_ops.py` - Main entry point script
- `yaml_parser.py` - Stage 1: Parse native_functions.yaml
- `xpu_checker.py` - Stage 2: Check XPU implementations
- `triton_checker.py` - Stage 3: Check Triton/Inductor coverage
- `output_formatter.py` - Generate JSON/CSV outputs
- `summary_report.py` - Generate summary report
- `test_tool.py` - Automated tests
- `EXAMPLES.md` - Detailed usage examples
- `output/` - Generated output files directory

## See Also

- [EXAMPLES.md](EXAMPLES.md) - Detailed usage examples and workflows
- [output/README.md](output/README.md) - Information about generated output files
