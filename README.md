# maestro-economics

Public plugin shell for Maestro research workflows.

This repository intentionally contains only host-plugin metadata and public agent
SOP skills. It does not contain the RA Compute CLI, Python runtime, worker code,
deployment workflow, tests, or implementation references.

Current public plugin version: `0.6.9`.

Included skills:

- `ra-compute`: GPU and long-running research job SOP.
- `ra-data`: dataset discovery, inspection, query, citation, and download SOP.
- `ra-experiments`: online experiment lifecycle and export SOP.

`mecon` is the CLI used by these workflows; it is not the organizing concept of
the skills. The skills are organized by user task and research object.

## Boundary

- Public: plugin manifests, marketplace metadata, and high-level skill routing.
- Private: CLI, Python package, worker code, deployment scripts, runtime tests, and operational runbooks.

The private implementation lives with RA Suite. Do not add runtime or worker code back to this repository.

## Contents

- `.claude-plugin/`
- `.codex-plugin/`
- `.github/workflows/public-shell-boundary.yml`
- `skills/ra-compute/SKILL.md`
- `skills/ra-data/SKILL.md`
- `skills/ra-experiments/SKILL.md`
