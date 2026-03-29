---
name: maestro-economics-computation
description: |
  Run Python estimation code on cloud GPU via RA Compute (Modal). Handles onboarding, workspace setup, job submission, monitoring, result sync.
  Triggers: GPU compute, run on GPU, structural estimation, BLP, DDC, bootstrap, Monte Carlo, JAX, GPU acceleration, submit job, mecon, RA Compute, workspace, sync results, cloud compute, run my script on GPU, estimate on cloud.
---

# RA Compute — GPU Research Computing

Submit Python estimation code to run on cloud GPU via RA Compute.
Users write a `run(ctx)` function. The platform handles workspace sync,
GPU allocation, progress monitoring, and result collection.

## Section 0: Onboarding (run on first use)

1. Check `mecon` installed: `which mecon` — if missing, install: `pip install maestro-economics`
2. Check API key configured: `mecon whoami` — if not, guide user to https://ra.maestro.onl/settings/compute to create key, then `mecon setup`
3. Check balance: `mecon balance` — if zero, guide to purchase credits

## Section 1: Workspace Setup (MANDATORY)

### Initialize workspace

```bash
cd ~/my-estimation-project
mecon workspace init                    # auto-names from directory path
```

### Track files (whitelist — only tracked files sync to cloud)

```bash
mecon add run.py helpers.py             # code files
mecon add data/*.parquet data/*.csv     # data files
mecon workspace status                  # verify tracked files
```

### Project structure

```
my_project/
  run.py              # REQUIRED: def run(ctx) -> dict
  *.py                # additional modules
  data/               # input data — ONLY .csv and .parquet
    panel.parquet
    config.json
  output/             # results synced back from cloud
  .mecon/             # workspace manifest (auto-created)
    manifest.json
  .meconignore        # exclude patterns (like .gitignore)
```

### Data Rules (ENFORCED by CLI)

- **Allowed formats**: .py .csv .parquet .json .txt .yaml .toml .R .jl
- **Banned (must convert)**: .mat .h5 .hdf5 .dta .npz .pkl .sav
- **Keep data minimal** — only columns/rows needed for estimation
- Upload/download time burns GPU credits on legacy mode

### Data Preparation (MANDATORY before sync)

If user has .mat/.h5/.dta files, convert FIRST:

```python
# .mat -> parquet
import scipy.io, pandas as pd
mat = scipy.io.loadmat("data.mat")
df = pd.DataFrame(mat["variable_name"])
df.to_parquet("data/variable.parquet")

# .dta -> parquet
df = pd.read_stata("data.dta")
df.to_parquet("data/panel.parquet")
```

## Section 2: The run(ctx) Contract

```python
def run(ctx):
    import jax.numpy as jnp
    import pandas as pd

    ctx.log("Loading data")
    ctx.progress(0.1, "Loading data")
    df = pd.read_parquet(f"{ctx.data_dir}/panel.parquet")

    # Resume from checkpoint if available
    state = ctx.load_checkpoint()
    if state:
        start = int(state["iteration"])
        best_x = state["best_x"]
        ctx.log(f"Resumed from checkpoint at iter {start}")
    else:
        start = 0
        best_x = jnp.zeros(10)

    ctx.progress(0.3, "Starting estimation")
    X = jnp.array(df[["x1", "x2"]].values, dtype=jnp.float32)

    for i in range(start, ctx.config.get("max_evals", 100)):
        # ... estimation with JAX ...
        if i % 20 == 0:
            ctx.save_checkpoint(iteration=i, best_x=best_x)
            ctx.log(f"Iter {i}: f={best_f:.4f}")

    ctx.progress(0.8, "Computing standard errors")
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
    "diagnostics": {
        "converged": bool,    # REQUIRED
        ...                   # gradient_norm, iterations, loss, etc.
    }
}
```

## Section 3: Submit & Monitor

### Sync and submit

```bash
mecon sync                              # push code + pull results (incremental)
mecon submit run.py --gpu l4            # submit specific entry point
mecon submit run.py --config '{"simul_times": 5, "max_evals": 100}'
mecon submit run_blp.py --gpu a100      # different script, different GPU
```

### Monitor

```bash
mecon logs -f <job_id>                  # stream logs (real-time)
mecon watch <job_id>                    # poll progress until done
mecon status <job_id>                   # one-shot status check
mecon list                              # recent jobs
```

### Get results

```bash
mecon sync                              # pull output/ files back to local
ls output/                              # results.json, job.log, etc.
```

### Cancel

```bash
mecon cancel <job_id>
```

## Section 4: CLI Reference

```bash
# Setup
mecon setup                              # configure API key (interactive)
mecon setup --profile local              # configure for localhost testing
mecon whoami                             # verify auth
mecon balance                            # check credits
mecon profiles                           # list all profiles

# Workspace
mecon workspace init                     # initialize in current directory
mecon workspace status                   # show tracked files + sync state
mecon add run.py data/*.parquet          # track files
mecon rm old_file.py                     # untrack file
mecon sync                               # push changes + pull results

# Submit
mecon submit run.py                      # workspace mode (no zip, no upload)
mecon submit run.py --gpu a100           # specify GPU
mecon submit run.py --config '{"k": 5}'  # pass config to ctx.config
mecon submit run.py --timeout 7200 --no-watch

# Legacy (no workspace)
mecon submit ./my_project/               # zip + upload directory
mecon submit script.py --data input.csv  # single file + data

# Monitor
mecon status <job_id>
mecon watch <job_id>
mecon logs <job_id>
mecon logs -f <job_id>                   # follow mode (poll)
mecon list

# Results
mecon sync                               # pull results to local output/

# Cancel
mecon cancel <job_id>

# Environment switching
mecon --profile local submit ...
mecon --endpoint http://localhost:3000 balance
```

## Section 5: Writing JAX Code for GPU

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

### Pre-installed packages (no pip install needed)

JAX, jaxlib, jaxopt, numpy, pandas, scipy, pyarrow, cma, statsmodels, scikit-learn, duckdb, matplotlib, httpx

## Section 6: Validation

### Pre-Submit (agent checks locally)

1. `run.py` exists and has `def run(ctx)`
2. No `.mat/.h5/.npz` in tracked files (must be parquet/csv)
3. `mecon workspace status` shows all files synced
4. JAX imports are inside `run()`, not at module level
5. `ctx.progress()` calls present
6. Return dict has `estimates` and `diagnostics` keys

### Post-Complete (agent verifies results)

1. `mecon sync` pulls output files
2. `output/results.json` exists
3. `diagnostics.converged == True`
4. All estimates are finite (no NaN/Inf)
5. Standard errors are positive
6. Parameter magnitudes are plausible for the domain

## Section 7: GPU Rate Table

| GPU | Credits/hr | Best for |
|-----|-----------|----------|
| T4 | 60 | Testing, small Monte Carlo |
| L4 | 120 | FP32 search, medium estimation |
| A10G | 180 | Larger models |
| L40S | 300 | Large-scale search |
| A100 | 480 | FP64 validation, BLP/DDC production |
| H100 | 720 | Largest models |

## ctx API Reference

| Namespace | Method | Description |
|-----------|--------|-------------|
| ctx.data_dir | str | Path to data files (R2 mount, lazy read) |
| ctx.output_dir | str | Path for output (synced back via `mecon sync`) |
| ctx.config | dict | User config from `--config` JSON |
| ctx.gpu.type | str | GPU type (T4, L4, A100, etc.) |
| ctx.gpu.memory_gb | float | GPU VRAM in GB |
| ctx.progress | (pct, msg) | Report progress 0.0-1.0 |
| ctx.log | (msg) | Write log chunk (streamable via `mecon logs -f`) |
| ctx.save_checkpoint | (**arrays) | Save numpy arrays to R2 (survives crash) |
| ctx.load_checkpoint | () -> dict\|None | Load last checkpoint (None if first run) |
| ctx.has_checkpoint | bool | True if checkpoint exists |

## MUST NOT

- Upload raw .mat/.h5/.dta files (convert to parquet first)
- Skip `ctx.progress()` calls (no progress = no monitoring)
- Return non-dict results
- Use scipy.optimize in JIT-compiled hot loops
- Upload unnecessary files (plots, logs, previous results)
- Run FP64 on T4/L4 (1/64 ratio = wasted credits)
