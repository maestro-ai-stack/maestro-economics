---
name: maestro-economics-computation
description: |
  Run Python estimation code on cloud GPU via RA Compute (Modal). Handles onboarding, code generation, job submission, monitoring.
  Triggers: GPU compute, run on GPU, structural estimation, BLP, DDC, bootstrap, Monte Carlo, JAX, GPU acceleration, submit job, mecon, RA Compute, 结构估计, GPU加速, 远程计算, 高性能计算, run my script on GPU, estimate on cloud.
  Do NOT use for: simple pandas workflows → data-analyst, local DuckDB acceleration → data-analyst.
---

# RA Compute — GPU Research Computing

Submit Python estimation code to run on cloud GPU via RA Compute.
Users write a `run(ctx)` function. The platform handles CLI setup, API key,
GPU allocation, progress monitoring, and result collection.

## Section 0: Onboarding (run on first use)

1. Check `mecon` installed: `which mecon` — if missing, install: `pip install maestro-economics`
2. Check API key configured: `mecon whoami` — if not, guide user to https://ra.maestro.onl/settings/compute to create key, then `mecon setup`
3. Check balance: `mecon balance` — if zero, guide to purchase credits

## Scripts (ONLY interface)

All scripts in skills/computation/scripts/. NEVER call API directly.

### submit.sh — End-to-end job submission
Usage: ./submit.sh --code script.py --data data.csv [--gpu 4090] [--timeout 3600]

### status.sh — Check job status
Usage: ./status.sh <job_id>

### download.sh — Download results
Usage: ./download.sh <job_id> [--output ./results]

## The run(ctx) Convention

Users write a Python file with a `run(ctx)` function:

```python
def run(ctx):
    import jax.numpy as jnp

    # Load data
    panel = ctx.data.load("data.csv", entity="firm_id", time="year")

    # Transform (Ontology Category 3)
    panel = ctx.transform.winsorize(panel, "revenue", pct=0.01)
    panel = ctx.transform.lag(panel, "investment", periods=1)

    ctx.progress(0.2, "Data prepared")

    # Computation
    X = jnp.array(panel[["x1", "x2"]].values)
    # ... estimation logic ...

    ctx.progress(0.9, "Computing SE")

    return {
        "estimates": {"beta": [1.2, -0.5]},
        "diagnostics": {"converged": True}
    }
```

## ctx API Reference

| Namespace | Method | Description |
|-----------|--------|-------------|
| ctx.data | .load(file, entity=, time=) | Load CSV/Parquet, tag panel metadata |
| ctx | .data_dir | Path to uploaded data files |
| ctx | .output_dir | Path for output (auto-uploaded) |
| ctx.transform | .winsorize(df, col, pct) | Clip at quantile |
| ctx.transform | .lag(df, col, periods, entity, time) | Panel lag |
| ctx.transform | .lead(df, col, periods, entity, time) | Panel lead |
| ctx.transform | .diff(df, col, periods, entity, time) | First difference |
| ctx.transform | .merge(left, right, on, how) | Merge with diagnostics |
| ctx.transform | .balance_panel(df, entity, time) | Keep complete entities |
| ctx.transform | .dummy(df, col) | Dummy variables |
| ctx.transform | .standardize(df, cols) | Zero-mean unit-var |
| ctx.transform | .recode(df, col, mapping) | Recode values |
| ctx.progress | (pct, msg) | Report progress 0.0-1.0 |
| ctx.config | dict | User config from submission |
| ctx.gpu | .type, .memory_gb | GPU hardware info |

## Writing JAX Code for GPU

- Import JAX inside run(ctx), not at module level
- Use jnp.array() to move data to GPU
- Use jax.jit for hot loops
- Use jax.vmap for vectorization
- Use jaxopt for optimization (L-BFGS, etc.)
- FP32 is fine for search; validate at FP64 if needed

## Translating from MATLAB/R

Common patterns:
- MATLAB `fminunc` → `jaxopt.LBFGS` or `jaxopt.ScipyMinimize`
- R `optim(method="L-BFGS-B")` → `jaxopt.LBFGS` with bounds
- MATLAB matrix ops → `jnp.` equivalents
- R data.table → pandas DataFrame with ctx.transform helpers

## MUST NOT

- Call API endpoints directly (use scripts only)
- Skip progress reporting (always call ctx.progress)
- Return non-dict results (must return a dict)
- Use scipy in hot loops (use jax/jaxopt instead)

## Validation Checklist

After job completes, verify:
- [ ] Return dict has 'estimates' and 'diagnostics' keys
- [ ] Optimization converged
- [ ] Results are finite (no NaN/Inf)
- [ ] Standard errors are positive
- [ ] Parameter magnitudes are plausible for the domain
