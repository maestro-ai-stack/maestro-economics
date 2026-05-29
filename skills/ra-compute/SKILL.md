---
name: ra-compute
description: |
  Use for RA Compute CPU/GPU research execution: submitting, monitoring, diagnosing, cancelling, resuming, optimizing, or downloading compute jobs; mecon CLI setup; CPU resource profiles; GPU tier choice; long empirical scripts; structural/JAX estimation; bootstrap sweeps; large Parquet first-pass analysis; failed, stalled, timed_out, or cancelling runs.
---

# RA Compute

Goal: help an economist finish a research run with durable outputs. Keep the
conversation about scripts, runs, logs, artifacts, cost, and next steps.

This public skill is a thin router. Product logic belongs in `mecon` and the RA
Compute server, not in this markdown. Do not expose private platform internals.

## Triggers

Use this skill when the user mentions:

- RA Compute, `mecon submit`, `mecon resources`, `mecon status`, `mecon logs`,
  `mecon watch`, `mecon download`, `mecon precompile`;
- CPU jobs, GPU jobs, resource profile, `cpu-4c-16gb`, L4, A100, H100;
- long empirical scripts, structural estimation, JAX/XLA/JIT, BLP, DDC, SMM,
  bootstrap, placebo, parameter sweep, Monte Carlo;
- large Parquet, DuckDB, too large for laptop, first-pass summaries;
- failed/stalled/timed_out/cancelling compute runs, missing outputs, stale
  progress, OOM, low GPU utilization, workspace stale, missing API key.

Do not use this skill for dataset discovery/download unless the user is asking
to execute a compute job. Use `ra-data` for catalog/query/download-only work.
Do not use it for experiment lifecycle work. Use `ra-experiments` there.

## First Actions

Run the helper first when possible:

```bash
skills/ra-compute/scripts/compute-readiness.sh
```

The helper only calls supported `mecon` commands. It does not call raw API
endpoints and does not submit paid work.

If the helper is unavailable, run these checks manually:

```bash
mecon --version
mecon submit --help
mecon resources
mecon doctor
```

`mecon --version` must be `0.7.1` or newer for CPU/GPU `--resource` support.

## Server-Owned Flow

Use server-backed commands as the source of truth:

- capability and pricing: `mecon resources`
- account/workspace readiness: `mecon doctor`
- submission: `mecon submit . --resource <profile>`
- live state: `mecon status <job_id>`, `mecon watch <job_id>`
- logs/artifacts: `mecon logs <job_id>`, `mecon download <job_id>`
- post-run advice: `mecon profile <job_id>`

Do not maintain pricing tables, raw API payloads, provider details, or hidden
scheduling heuristics in the skill.

## Load The Right Guide

Read only the guide needed for the task:

- Server/product boundary: `references/server-owned-workflows.md`
- New job or resource choice: `references/resource-selection.md`
- Workspace setup and submission: `references/submission-flow.md`
- Large Parquet / first-pass data work: `references/large-parquet.md`
- Failed, stalled, timed_out, cancelling, OOM, low utilization: `references/diagnosis.md`
- Client-facing explanation: `references/customer-language.md`

## Safe Commands

Supported customer-facing path:

```bash
mecon doctor
mecon resources
mecon sync
mecon submit . --resource cpu-4c-16gb --timeout 3600
mecon submit . --resource l4 --timeout 3600
mecon watch <job_id>
mecon status <job_id>
mecon logs <job_id>
mecon profile <job_id>
mecon download <job_id>
```

Use `mecon open billing | jobs | keys | dashboard | job <id>` when the human
must act in the browser.

## Hard Rules

- Do not submit CPU work with raw API JSON or helper scripts. Use
  `mecon submit . --resource <profile>`.
- Do not copy server-returned pricing or resource policy into the skill as a
  static table.
- Do not recommend a bigger GPU without logs/profile evidence.
- Do not call a job hung from stale progress text alone. Check status, logs,
  activity, and artifacts.
- Do not rerun before downloading existing terminal outputs.
- Do not expose Modal/R2/callback internals in client-facing text unless the user
  explicitly asks for platform details.
- A task is not done until there is a terminal status, a verified blocker, or
  the user explicitly asked to stop after submission/status.

## Completion Output

Report only what matters:

- job id;
- resource profile;
- current or terminal status;
- artifact/download state;
- next concrete command or human action.
