# Diagnosis

Use this guide for failed, stalled, timed_out, cancelling, OOM, low utilization,
or missing-output jobs.

## First Commands

```bash
mecon status <job_id>
mecon logs <job_id>
mecon profile <job_id>
mecon download <job_id>
```

Run `download` if the job is terminal or artifacts are visible. Do this before
retrying.

## Status Meaning

- `queued`: accepted, waiting to start.
- `running`: worker is active or recently active.
- `cancelling`: cancel requested; wait for terminal state.
- `completed`: download outputs.
- `timed_out`: configured time budget reached; partial outputs may exist.
- `failed`: runtime error, worker death, callback failure, or OOM.
- `cancelled`: cancellation reached terminal state.

Stale progress text alone is not a hang. Check logs, activity freshness, profile,
and artifacts.

If `mecon status` returns a newer status label than this guide knows, trust the
server output and report the exact label.

## Common Cases

Workspace stale:

```bash
mecon sync
mecon submit . --resource <profile>
```

Missing API key:

```bash
mecon login
```

CPU option missing:

```bash
python3 -m pip install --upgrade maestro-economics
mecon submit --help
```

JAX compile hang:

```bash
mecon precompile . --timeout 600
```

Then make non-varying args static and avoid dynamic shapes.

OOM:

- confirm from logs/profile;
- reduce data/config first if possible;
- then choose a larger profile.

Low GPU utilization:

- run `mecon profile <job_id>`;
- follow the server recommendation first;
- do not buy a bigger GPU first if the server says the workload is host-bound.

Timeout:

- download partial outputs;
- check whether `run(ctx)` can resume from checkpoint;
- increase timeout only if the code is making useful progress.

Cancelling:

- keep polling until `cancelled` or another terminal state;
- do not submit duplicate paid work unless the user explicitly accepts the risk.
