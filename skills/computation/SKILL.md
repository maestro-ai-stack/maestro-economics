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

## Section 1: Project Structure (MANDATORY)

Before writing any code, set up this structure:

```
my_project/
├── run.py              # REQUIRED: def run(ctx) -> dict
├── *.py                # additional modules (imported by run.py)
├── data/               # input data — ONLY .csv and .parquet allowed
│   ├── panel.parquet   # ✓ extracted, minimal columns
│   └── config.json     # ✓ small metadata
├── output/             # worker writes results here (auto-uploaded as zip)
└── .meconignore        # exclude patterns (like .gitignore)
```

### Data Rules (ENFORCED by CLI)

- **Max upload: 50 MB** (code + data combined zip)
- **Allowed formats**: .py .csv .parquet .json .txt .yaml .toml .R .jl
- **Auto-excluded**: .mat .h5 .hdf5 .npz .pkl .png .jpg .log .pyc
- **Upload/download time burns GPU credits** — keep data minimal

### Data Preparation (MANDATORY before submit)

If user has .mat/.h5/.dta files, convert FIRST:

```python
# .mat → parquet
import scipy.io, pandas as pd
mat = scipy.io.loadmat("data.mat")
df = pd.DataFrame(mat["variable_name"])
df.to_parquet("data/variable.parquet")

# .dta → parquet
df = pd.read_stata("data.dta")
df.to_parquet("data/panel.parquet")
```

**Extract only needed columns/rows.** A 96MB .mat with 80 variables should become
a 2MB .parquet with the 5 columns actually used in estimation.

### .meconignore Example

```
# Large raw data (already converted to parquet)
*.mat
*.h5
*.dta
# Results from previous runs
results/
*.npz
# Plots and logs
*.png
*.log
convergence_*.json
```

## Section 2: The run(ctx) Contract

```python
def run(ctx):
    import jax.numpy as jnp
    import pandas as pd

    ctx.progress(0.1, "Loading data")
    df = pd.read_parquet(f"{ctx.data_dir}/panel.parquet")

    ctx.progress(0.3, "Starting estimation")
    X = jnp.array(df[["x1", "x2"]].values, dtype=jnp.float32)
    # ... estimation with JAX ...

    ctx.progress(0.8, "Computing standard errors")
    # ... write detailed output ...
    df_results = pd.DataFrame({"param": names, "estimate": betas, "se": ses})
    df_results.to_csv(f"{ctx.output_dir}/estimates.csv", index=False)

    ctx.progress(1.0, "Done")
    return {
        "estimates": {"beta": betas.tolist()},
        "diagnostics": {"converged": True, "iterations": n_iter}
    }
```

### Return Dict (REQUIRED)

```python
{
    "estimates": { ... },     # parameter estimates, coefficients
    "diagnostics": {          # convergence info
        "converged": bool,    # REQUIRED
        ...                   # gradient_norm, iterations, loss, etc.
    }
}
```

## Section 3: CLI Reference

```bash
# Setup
mecon setup                          # configure API key (interactive)
mecon setup --profile local          # configure for localhost testing
mecon whoami                         # verify auth
mecon balance                        # check credits
mecon profiles                       # list all profiles

# Submit
mecon submit run.py                  # single file
mecon submit ./my_project/           # directory (auto-zips, respects .meconignore)
mecon submit ./project/ --gpu a100   # specify GPU
mecon submit ./project/ --timeout 7200 --no-watch

# Monitor
mecon status <job_id>
mecon watch <job_id>                 # live progress polling
mecon list                           # recent jobs
mecon logs <job_id>                  # stdout/stderr

# Results
mecon download <job_id>              # download + extract results to ./results/
mecon download <job_id> -o ./out     # custom output dir

# Cancel
mecon cancel <job_id>

# Environment switching
mecon --profile local submit ...     # use localhost
mecon --endpoint http://localhost:3000 balance
```

## Section 4: Writing JAX Code for GPU

- Import JAX inside `run(ctx)`, not at module level
- Use `jnp.array(data, dtype=jnp.float32)` — FP32 for search, FP64 for final validation
- Use `jax.jit` for hot loops, `jax.vmap` for vectorization
- Use `jaxopt.LBFGS` for optimization (not scipy in hot path)
- Use `jax.lax.while_loop` / `fori_loop` — never Python while/if on traced values

### FP32 vs FP64 Decision

| GPU | FP64:FP32 ratio | Recommendation |
|-----|-----------------|----------------|
| T4/L4 | 1/64 | FP32 only. FP64 = 64x slower |
| A100 | 1/2 | FP64 viable for final pass |
| H100 | 1/2 | FP64 viable |

**Pattern**: Search in FP32 on L4 (cheap), validate final result in FP64 on A100.

### Translating from MATLAB/R

- MATLAB `fminunc` → `jaxopt.LBFGS` or `jaxopt.ScipyMinimize`
- R `optim(method="L-BFGS-B")` → `jaxopt.LBFGS` with bounds
- MATLAB `normcdf` → `jax.scipy.stats.norm.cdf`
- MATLAB matrix ops → `jnp.` equivalents

## Section 5: Validation Stages

### Pre-Submit (agent checks locally)

1. `run.py` exists and has `def run(ctx)`
2. No `.mat/.h5/.npz` in data/ (must be parquet/csv)
3. Total project size < 50MB
4. JAX imports are inside `run()`, not at module level
5. `ctx.progress()` calls present
6. Return dict has `estimates` and `diagnostics` keys

### Post-Complete (agent verifies results)

1. Job status == `completed` (not failed/cancelled)
2. `diagnostics.converged == True`
3. All estimates are finite (no NaN/Inf)
4. Standard errors are positive
5. Parameter magnitudes are plausible for the domain
6. If FP32 search: recommend FP64 validation run on A100

### FP Validation (for structural estimation)

```python
# After FP32 search converges, validate:
# 1. Re-evaluate at optimum in FP64
# 2. Check gradient norm < 1e-6
# 3. Verify Hessian is positive definite
# 4. Compare FP32 vs FP64 estimates (should be within 1e-4 relative)
```

## Section 6: GPU Rate Table

| GPU | Credits/hr | Best for |
|-----|-----------|----------|
| T4 | 60 | Testing, small Monte Carlo |
| L4 | 120 | FP32 search, medium estimation |
| A10G | 180 | Larger models |
| L40S | 300 | Large-scale search |
| A100 | 480 | FP64 validation, BLP/DDC production |
| H100 | 720 | Largest models |

## MUST NOT

- Upload raw .mat/.h5/.dta files (convert to parquet first)
- Skip `ctx.progress()` calls (no progress = no monitoring)
- Return non-dict results
- Use scipy.optimize in JIT-compiled hot loops
- Upload unnecessary files (plots, logs, previous results)
- Run FP64 on T4/L4 (1/64 ratio = wasted credits)

## ctx API Reference

| Namespace | Method | Description |
|-----------|--------|-------------|
| ctx.data_dir | str | Path to uploaded data files |
| ctx.output_dir | str | Path for output (auto-zipped and uploaded) |
| ctx.progress | (pct, msg) | Report progress 0.0-1.0 |
| ctx.config | dict | User config from submission |
| ctx.gpu.type | str | GPU type (T4, L4, A100, etc.) |
| ctx.gpu.memory_gb | float | GPU VRAM in GB |
