---
name: maestro-economics
description: |
  Public routing skill for Maestro Economics research-compute tasks. Use when the user asks about Maestro Economics, RA Compute, structural-estimation compute, or economics workflow routing.
  Triggers: maestro-economics, RA Compute, mecon, compute job, structural estimation, JAX, XLA, JIT warmup, CMA-ES, GPU optimization, low GPU utilization, checkpointing, timed_out, IO estimation, discrete choice estimation, random coefficients logit, BLP, dynamic discrete choice, IV, 2SLS, DiD, RDD, event study, panel regression.
  Do NOT use for: internal worker implementation, private CLI code, deployment scripts, credentials, operational runbooks, or private infrastructure details.
---

# Maestro Economics

This public skill is only a routing layer.

When a task requires live RA Compute operation, private runtime code, CLI behavior, worker debugging, deployment, or production incident handling, use the private RA Suite repository and its internal docs. Do not infer worker internals from this public plugin.

For user-facing research work, keep guidance at the workflow level:

1. Validate the local research code on CPU first.
2. Make the job resumable with checkpoints and explicit progress messages.
3. Start with the smallest GPU tier and shortest smoke run that exercises the expensive path.
4. Inspect terminal status, logs, profile metrics, and produced artifacts before scaling the run.
5. Treat repeated stalls as a systems issue first, not as a reason to burn more credits.

## RA Compute CLI-first diagnosis

For live RA Compute jobs, use the `mecon` CLI as the source of truth. Prefer
`mecon >= 0.6.8`; the API advertises the latest CLI through response headers,
and old clients should be upgraded before long-running GPU work.

When diagnosing a job:

1. Run `mecon status <job_id>` first. Surface any `server_advice` verbatim.
2. For live monitoring, use `mecon watch <job_id>`. It escalates stalled
   activity, heartbeat loss, `timed_out`, and terminal advice.
3. For terminal jobs, run `mecon profile <job_id>` before recommending a
   different GPU tier.
4. If logs are needed, run `mecon logs <job_id>` or `mecon debug <job_id>`.
5. If a terminal job has artifacts, run `mecon download <job_id>`; recent
   clients can recover inline result JSON when the server has a DB result but
   no legacy result file.

Status semantics:

- `timed_out` means the job reached its configured time budget. It is not a
  crash, Modal platform kill, or proof of XLA hang. Partial logs, checkpoints,
  outputs, and runtime-recovered partial result JSON may be available.
- `failed` is reserved for actual failure signals such as worker/container
  death, callback failure, unhandled runtime error, or OOM.
- A stale progress label is not enough to call a hang. Check heartbeat and
  latest worker activity.
- `ctx.progress()` is a user-code observability signal, not the terminal
  network callback. Maestro's private worker/runtime owns terminal callbacks,
  timeout handling, artifact persistence, and credit settlement.
- Do not tell user scripts or agents to mark jobs completed/failed themselves.
  They should improve `ctx.progress(...)` and `ctx.update_result(...)`; server
  status, retries, attempts, and billing are platform-owned.
- For long-running searches, tell user code to call
  `ctx.update_result(best_dist=..., nfev=..., ...)` whenever the incumbent
  improves. The runtime persists this generic partial result; agents must not
  rely on workload-specific log parsing for business results.

GPU advice:

- OOM signals such as `RESOURCE_EXHAUSTED`, CUDA allocation failure, or cgroup
  OOM can justify a larger tier plus a smaller static JAX graph.
- Low GPU utilization is not automatically a reason to upgrade. If
  `mecon profile` reports low average utilization, near-zero p50, and only
  moderate p95, treat it as `optimize_workload`: likely host-bound Python
  orchestration, many small kernels, or serial optimizer evaluation.
- For `optimize_workload`, advise batching/vectorizing candidate evaluations,
  keeping JAX shapes static, reducing Python callbacks/checkpoint/log cadence,
  and shrinking serial search before buying a bigger GPU.

Never place Python source, CLI code, Modal worker code, deployment commands, credentials, endpoints, or internal runbooks in this repository.
