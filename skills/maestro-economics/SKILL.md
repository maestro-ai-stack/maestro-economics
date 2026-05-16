---
name: maestro-economics
description: |
  Public SOP skill for Maestro Economics / RA Compute research-compute jobs. Use when the user asks about Maestro Economics, RA Compute, mecon CLI, GPU compute jobs, structural estimation, JAX/XLA/JIT warmup, CMA-ES, checkpointing/resume, timed_out jobs, job diagnosis, low GPU utilization, OOM, IO estimation, discrete choice estimation, random coefficients logit, BLP, dynamic discrete choice, IV, 2SLS, DiD, RDD, event study, or panel regression.
  Do NOT use for: internal worker implementation, private CLI source, Modal worker code, deployment scripts, credentials, production runbooks, billing internals, or private infrastructure details.
---

# Maestro Economics

This skill is the public operating procedure for helping a user or agent run
research compute with `mecon`. Keep guidance at the CLI and `run(ctx)` contract
level. Do not expose private worker source, endpoints, credentials, deployment
commands, or infrastructure runbooks.

## First Decision

Classify the request before acting:

- **Prepare a new job**: help write or review `run(ctx)`, sync inputs, smoke test,
  then submit.
- **Monitor a live job**: use `mecon watch <job_id>` and status output.
- **Diagnose a terminal job**: use status, logs, profile, and artifacts before
  recommending retry, resume, larger GPU, or code changes.
- **Optimize workload**: inspect GPU profile and code shape before suggesting a
  bigger tier.

Never guess OOM, XLA hang, deployment drift, or customer-code failure from a
stale progress line alone.

## Job SOP

Use this order for ordinary long-running compute work:

1. Validate locally on CPU with a tiny config that reaches the expensive path.
2. Make `run(ctx)` produce progress, partial results, and restartable state when
   the algorithm supports it.
3. Run `mecon doctor` before long GPU work.
4. Sync the workspace: `mecon sync`.
5. Submit the smallest meaningful smoke run first, with a bounded timeout.
6. Watch the job: `mecon watch <job_id>`.
7. Download artifacts after terminal status: `mecon download <job_id>`.
8. Profile before scaling or changing GPU tier: `mecon profile <job_id>`.

Use `mecon >= 0.6.9` for long GPU jobs. Recent clients reject stale workspace
snapshots before submit and surface server advice in status/watch output.

## CLI Source Of Truth

For live jobs, prefer `mecon` output over inference:

- `mecon status <job_id>`: first command for any job question. Surface
  `server_advice` if present.
- `mecon watch <job_id>`: monitor live progress, heartbeat/activity warnings,
  timeout, terminal status, and next action.
- `mecon logs <job_id>`: inspect user logs when status is not enough.
- `mecon debug <job_id>`: collect deeper public diagnostics.
- `mecon profile <job_id>`: inspect GPU utilization and memory before advising
  tier changes.
- `mecon download <job_id>`: retrieve result files and artifacts.
- `mecon precompile . --timeout 600`: use before GPU retry when JAX/XLA compile
  is suspected.

Do not tell user code or an agent to mark jobs completed/failed. The platform
owns terminal status, callbacks, retries, attempts, artifact persistence, and
credit settlement.

## `run(ctx)` Contract

The user's entrypoint should be:

```python
def run(ctx):
    ...
    return {
        "estimates": {...},
        "diagnostics": {...},
    }
```

Use the context helpers this way:

- `ctx.progress(pct, message)`: human-visible progress. It is not a keepalive
  requirement and not a terminal callback.
- `ctx.update_result(**fields)`: persist the current best partial result. Use it
  whenever the incumbent improves or a meaningful milestone completes.
- `ctx.save_checkpoint(**arrays)`: save simple NumPy-compatible restart state to
  the managed checkpoint file.
- `ctx.load_checkpoint()`: load that state at the start of a later run.
- `ctx.has_checkpoint`: check whether the managed checkpoint file exists.
- `ctx.checkpoint_dir`: directory for framework-native checkpoints such as
  TensorFlow, PyTorch, JAX/orbax, or custom files.
- `ctx.output_dir`: directory for final files and durable outputs.
- `ctx.log(message)`: explicit log messages for diagnostics.

Return a final dict for the completed result. Do not rely on log text as the
business result.

## Long-Running Search Pattern

For search, optimization, simulation, or estimation loops, use this shape:

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

Keep progress/checkpoint cadence coarse enough to avoid slowing the workload.
For GPU-heavy code, prefer batching/vectorization over frequent Python callbacks.

## Checkpoint And Resume Boundary

Checkpointing is user-code owned. Maestro stores files; it does not understand
the algorithm state.

- A resubmitted job only resumes if the user's `run(ctx)` loads prior state and
  continues from it.
- `ctx.save_checkpoint(**arrays)` is for simple NumPy-compatible state. It is
  not a generic pickle, TensorFlow, or PyTorch checkpoint format.
- For framework-native state, write files under `ctx.checkpoint_dir` and load
  them explicitly in `run(ctx)`.
- Do not promise automatic resume. Say "checkpoint files are available" unless
  the code actually loads them.

Framework-style checkpoint example:

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

- `completed`: final callback/result was recorded. Download artifacts.
- `timed_out`: configured time budget was reached. This is not a crash. Partial
  results, logs, outputs, and checkpoints may be available.
- `failed`: actual failure signal such as worker/container death, callback
  failure, unhandled runtime error, or OOM.
- `cancelled`: user/platform cancellation path. Check whether artifacts were
  flushed before advising retry.
- Stale progress is not enough to call a hang. Check heartbeat/activity,
  logs, and profile.

If the job is terminal and has artifacts, download before retrying.

## Diagnosis SOP

For a failed, stalled, or suspicious job:

1. `mecon status <job_id>` and read status, advice, progress, and artifact
   availability.
2. `mecon logs <job_id>` if the failure is not already explained.
3. `mecon profile <job_id>` before GPU-tier advice.
4. `mecon download <job_id>` if outputs/checkpoints exist.
5. Only then choose the next action: no rerun, download result, resume-capable
   rerun, precompile, optimize workload, smaller config, or larger GPU.

Do not make nontechnical users run exploratory diagnostics when the platform or
agent can inspect status and artifacts directly.

## GPU Advice

Recommend a larger GPU only when evidence supports it:

- OOM evidence: `RESOURCE_EXHAUSTED`, CUDA allocation failure, cgroup OOM, or
  memory profile near the limit with an allocation failure.
- Compile/JAX suspicion: long silence during first JIT/warmup, static-shape
  issues, or precompile failure. Run `mecon precompile . --timeout 600` before
  another GPU retry.
- Low utilization: low average GPU use, near-zero p50, and modest p95 usually
  means host-bound orchestration, serial optimizer evaluation, many small
  kernels, or excess Python callbacks. Advise batching/vectorization/static
  shapes before buying a bigger GPU.

Do not treat low GPU utilization as proof that a larger tier will help.

## Customer-Facing Language

Use plain language:

- "The run reached its time budget" instead of "timed_out callback".
- "The output files are available" instead of storage internals.
- "This needs a code change to continue from saved work" instead of "resume
  semantics are user-code owned".
- "You do not need to rerun this job" when artifacts already contain the final
  result.

Avoid mentioning private infrastructure names, credentials, worker internals,
or deployment details.
