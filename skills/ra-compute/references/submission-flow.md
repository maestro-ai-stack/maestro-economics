# Submission Flow

Use this guide for new or revised compute jobs.

## Preflight

```bash
mecon --version
mecon submit --help
mecon resources
mecon doctor
mecon sync
```

If `--resource` is missing:

```bash
python3 -m pip install --upgrade maestro-economics
```

Use `mecon resources` as the current catalog. Do not use a profile that the
server does not return.

## CPU Smoke

```bash
mecon submit . --resource cpu-4c-16gb --timeout 3600 --config '{"smoke": true}'
mecon watch <job_id>
mecon download <job_id>
```

If smoke fails, fix code or workspace before scaling.

## GPU Smoke

For JAX/PyTorch/CUDA or structural estimation:

```bash
mecon precompile . --timeout 600
mecon sync
mecon submit . --resource l4 --timeout 3600 --config '{"max_iter": 5}'
```

Move to a larger server-returned profile only with logs/profile evidence.

## `run(ctx)` Contract

Return final data and write durable files when useful:

```python
def run(ctx):
    result = do_work(ctx.config)
    return {
        "estimates": result.estimates,
        "diagnostics": result.diagnostics,
    }
```

Useful helpers:

- `ctx.progress(pct, message)`: human-visible progress.
- `ctx.update_result(**fields)`: current best partial result.
- `ctx.save_checkpoint(**arrays)`: simple restart state.
- `ctx.load_checkpoint()`: load previous checkpoint.
- `ctx.output_dir`: durable output files.
- `ctx.log(message)`: diagnostic log line.

Checkpointing is user-code owned. A rerun resumes only if `run(ctx)` loads state
and continues from it.

## Long Loop Pattern

Use coarse progress and checkpoint cadence:

```python
def run(ctx):
    max_iter = int(ctx.config.get("max_iter", 100))
    if ctx.has_checkpoint:
        state = ctx.load_checkpoint()
        start_iter = int(state["iter"])
        best = state["best"]
    else:
        start_iter = 0
        best = None

    for i in range(start_iter, max_iter):
        best = step(i, best)
        if (i + 1) % 5 == 0:
            ctx.progress((i + 1) / max_iter, f"iteration {i + 1}/{max_iter}")
            ctx.update_result(iter=i + 1, best=best)
            ctx.save_checkpoint(iter=i + 1, best=best)

    return {"best": best, "iterations": max_iter}
```

Do not rely on logs as the business result. Put actual outputs in return values
or files under `ctx.output_dir`.
