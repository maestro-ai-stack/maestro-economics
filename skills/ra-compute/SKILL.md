---
name: ra-compute
description: |
  SOP for RA Compute GPU and long-running research jobs. Use when the user asks to prepare, submit, monitor, diagnose, resume, optimize, or download a compute job; asks about run(ctx), ctx.progress, ctx.update_result, checkpoints, JAX/XLA/JIT, CMA-ES, structural estimation, GPU tier choice, OOM, timed_out, failed, stalled, low GPU utilization, or mecon compute commands.
---

# RA Compute

Use this skill for long-running research compute jobs. The goal is to help the
user finish work with durable outputs, not to expose platform internals.

## Job SOP

For a new or revised job:

1. Validate the code locally on CPU with a tiny config that reaches the
   expensive path.
2. Make `run(ctx)` return a final result and write durable outputs.
3. For long loops, add coarse `ctx.progress(...)`, `ctx.update_result(...)`, and
   checkpointing only where it does not slow the workload.
4. Run `mecon doctor`.
5. Run `mecon sync`.
6. Submit a bounded smoke run before the full run.
7. Watch the run and download artifacts after terminal status.
8. Profile before recommending a larger GPU.

Use `mecon >= 0.6.9` for long GPU jobs.

## Core Commands

- `mecon doctor`: check local readiness before long work.
- `mecon sync`: upload the current workspace snapshot.
- `mecon submit . --gpu l4 --timeout 3600 --config '{"max_iter": 5}'`: submit a
  smoke run.
- `mecon watch <job_id>`: monitor live status.
- `mecon status <job_id>`: inspect status and server advice.
- `mecon logs <job_id>`: inspect job logs.
- `mecon profile <job_id>`: inspect GPU and memory profile.
- `mecon download <job_id>`: retrieve outputs.
- `mecon precompile . --timeout 600`: test JAX/XLA compile before a GPU retry.

Do not ask user code to mark a job completed or failed. The platform owns
terminal status, callbacks, retries, artifacts, and credit settlement.

## `run(ctx)` Contract

The entrypoint should return a final dict:

```python
def run(ctx):
    ...
    return {
        "estimates": {...},
        "diagnostics": {...},
    }
```

Use context helpers for their narrow purpose:

- `ctx.progress(pct, message)`: human-visible progress, not a keepalive
  requirement.
- `ctx.update_result(**fields)`: persist the current best partial result.
- `ctx.save_checkpoint(**arrays)`: save simple NumPy-compatible restart state.
- `ctx.load_checkpoint()`: load that state at the start of a later run.
- `ctx.has_checkpoint`: check whether the managed checkpoint exists.
- `ctx.checkpoint_dir`: write framework-native checkpoints or custom state
  files.
- `ctx.output_dir`: write durable result files.
- `ctx.log(message)`: explicit diagnostic logs.

Do not rely on log text as the business result.

## Long-Running Loop Pattern

Use this pattern for search, optimization, simulation, or estimation loops:

```python
def run(ctx):
    max_iter = int(ctx.config.get("max_iter", 100))

    if ctx.has_checkpoint:
        state = ctx.load_checkpoint()
        start_iter = int(state["iter"])
        best_x = state["best_x"]
        best_dist = float(state["best_dist"])
    else:
        start_iter = 0
        best_x = None
        best_dist = float("inf")

    for i in range(start_iter, max_iter):
        candidate_x, candidate_dist = step(i, best_x)

        if candidate_dist < best_dist:
            best_x = candidate_x
            best_dist = candidate_dist
            ctx.update_result(iter=i + 1, best_dist=best_dist)

        if (i + 1) % 5 == 0:
            ctx.progress((i + 1) / max_iter, f"iteration {i + 1}/{max_iter}")
            ctx.save_checkpoint(iter=i + 1, best_x=best_x, best_dist=best_dist)

    return {
        "estimates": {"best_x": best_x, "best_dist": best_dist},
        "diagnostics": {"iterations": max_iter},
    }
```

Keep callback/checkpoint cadence coarse. For GPU workloads, batching and static
shapes usually matter more than frequent progress messages.

## Checkpoint Boundary

Checkpointing is user-code owned. RA Compute stores files; it does not know the
meaning of algorithm state.

- Resubmitting only resumes if `run(ctx)` explicitly loads prior state and
  continues from it.
- `ctx.save_checkpoint(**arrays)` is for simple NumPy-compatible state, not a
  generic pickle, TensorFlow, or PyTorch format.
- For framework-native state, write files under `ctx.checkpoint_dir` and load
  them explicitly.
- Say "checkpoint files are available" unless the code actually resumes from
  them.

Framework-native example:

```python
import os


def run(ctx):
    ckpt_path = f"{ctx.checkpoint_dir}/model.keras"
    if os.path.exists(ckpt_path):
        model = keras.models.load_model(ckpt_path)
    else:
        model = build_model()

    ...
    model.save(ckpt_path)
```

## Status Semantics

- `completed`: download outputs.
- `timed_out`: the configured time budget was reached. This is not a crash.
  Partial results, outputs, logs, or checkpoints may exist.
- `failed`: actual failure signal such as worker death, callback failure,
  unhandled runtime error, or OOM.
- `cancelled`: cancellation path. Check whether outputs were flushed.
- Stale progress is not enough to call a hang. Check status, logs, activity, and
  profile before deciding.

If a terminal job has artifacts, download them before retrying.

## Diagnosis SOP

For a failed, stalled, or suspicious job:

1. Run `mecon status <job_id>` and read advice, status, progress, and artifacts.
2. Run `mecon logs <job_id>` if status does not explain the issue.
3. Run `mecon profile <job_id>` before GPU-tier advice.
4. Run `mecon download <job_id>` if outputs or checkpoints exist.
5. Choose the smallest correct next action: no rerun, download result,
   resume-capable rerun, precompile, optimize workload, smaller config, or
   larger GPU.

## GPU Advice

Recommend a larger GPU only with evidence:

- OOM evidence: `RESOURCE_EXHAUSTED`, CUDA allocation failure, cgroup OOM, or
  memory profile near the limit with allocation failure.
- Compile/JAX evidence: long silence during first JIT/warmup, static-shape
  issue, or precompile failure.
- Low utilization usually means host-bound Python orchestration, serial search,
  many small kernels, or too many callbacks. Advise batching/vectorization and
  static shapes before buying a bigger GPU.

Use plain customer language: "The output files are available", "The run reached
its time budget", "This code needs to load saved state to continue from prior
work."
