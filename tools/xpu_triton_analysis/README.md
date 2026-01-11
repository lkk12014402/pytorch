# XPU/Triton Op Analysis Tool

This tool analyzes PyTorch ATen operators to identify which ones need XPU and/or Triton implementations.

## Purpose

The tool helps prioritize Triton kernel development for XPU by:
1. Identifying ops with `pointwise` or `reduction` tags in `native_functions.yaml`
2. Checking whether each op has XPU implementation (from dispatch keys or manual registration)
3. Determining if the op is covered by Inductor's Triton codegen paths

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
