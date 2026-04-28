# maestro-economics

Public plugin shell for Maestro Economics.

This repository intentionally contains only host-plugin metadata and the public agent skill. It does not contain the RA Compute CLI, Python runtime, Modal worker, deployment workflow, tests, or implementation references.

Current public plugin version: `0.6.9`. Live RA Compute diagnosis is
CLI-first: agents should use `mecon status`, `mecon watch`, `mecon profile`,
`mecon logs`, `mecon download`, and `mecon debug` rather than guessing from
stale progress text.
Timeouts are platform-owned: user code may emit `ctx.progress()` or return a
dict, but terminal status, timeout handling, result persistence, and billing
belong to the Maestro worker/runtime.
Agents should not ask user code to mark jobs completed or failed; server-side
status, attempts, retries, and billing are owned by the platform.
Workspace snapshots are submit-time checked in `mecon >= 0.6.9`; if tracked
files changed after the last `mecon sync`, submit refuses instead of running
stale code from R2.
For long-running searches, user code should publish structured incumbents with
`ctx.update_result(...)`; the runtime persists those as partial results if the
configured time budget is reached.

## Boundary

- Public: plugin manifests, marketplace metadata, and high-level skill routing.
- Private: CLI, Python package, worker code, deployment scripts, runtime tests, and operational runbooks.

The private implementation lives with RA Suite. Do not add runtime or worker code back to this repository.

## Contents

- `.claude-plugin/`
- `.codex-plugin/`
- `.github/workflows/public-shell-boundary.yml`
- `skills/maestro-economics/SKILL.md`
