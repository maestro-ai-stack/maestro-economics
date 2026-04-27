---
name: maestro-economics
description: |
  Public routing skill for Maestro Economics research-compute tasks. Use when the user asks about Maestro Economics, RA Compute, structural-estimation compute, or economics workflow routing.
  Triggers: maestro-economics, RA Compute, compute job, structural estimation, JAX, CMA-ES, GPU optimization, checkpointing, IO estimation, discrete choice estimation, random coefficients logit, BLP, dynamic discrete choice, IV, 2SLS, DiD, RDD, event study, panel regression.
  Do NOT use for: internal worker implementation, private CLI code, deployment scripts, credentials, operational runbooks, or private infrastructure details.
---

# Maestro Economics

This public skill is only a routing layer.

When a task requires live RA Compute operation, private runtime code, CLI behavior, worker debugging, deployment, or production incident handling, use the private RA Suite repository and its internal docs. Do not infer worker internals from this public plugin.

For user-facing research work, keep guidance at the workflow level:

1. Validate the local research code on CPU first.
2. Make the job resumable with checkpoints and explicit progress messages.
3. Start with the smallest GPU tier and shortest smoke run that exercises the expensive path.
4. Inspect terminal status, logs, and produced artifacts before scaling the run.
5. Treat repeated stalls as a systems issue first, not as a reason to burn more credits.

Never place Python source, CLI code, Modal worker code, deployment commands, credentials, endpoints, or internal runbooks in this repository.
