---
name: ra-experiments
description: |
  Manage Maestro experiments through the server-driven mecon exp contract. Use for experiment creation, editing, validation, previews, version review, collaboration, publishing, launches, sessions, and exports.
---

# RA Experiments

Use the authenticated Maestro service as the source of truth. This skill intentionally does not
copy component schemas, props, examples, or lifecycle enums because those contracts evolve with the
deployed runtime.

## Bootstrap

Follow and execute:

`https://ra.maestro.onl/help/experiments`

At minimum:

```bash
python3 -m pip install --upgrade maestro-economics
mecon login
mecon exp capabilities --json
mecon exp schema --json
mecon exp components --json
```

Use `mecon exp components <NAME> --json` before authoring a component. Never invent props from
memory. Use `mecon exp schema <TOPIC> --json` for source-file structure. Pattern guides are optional;
load one with `mecon exp patterns <NAME>` only when it matches the requested study family. Use
`mecon exp init <slug>` for a server-owned scaffold rather than writing a copied schema.

## Existing Experiment

```bash
mecon exp pull <slug>
# edit experiments/<slug>/
mecon exp check experiments/<slug> --journey
mecon exp push experiments/<slug>
mecon exp versions <slug>
```

A normal push creates an immutable preview and does not alter live. Pull before editing. On
`VERSION_CONFLICT`, pull the latest preview and reapply the intended change.

Only after review, publish directly as owner or request owner approval as a collaborator:

```bash
mecon exp publish <slug> --version <N> --summary "What changed and what was verified"

# Equivalent single command after local validation:
mecon exp push experiments/<slug> --publish --summary "What changed and what was verified"
```

The review link identifies an immutable preview version but grants no authority. The signed-in
experiment owner must still confirm publication.

## Collaboration

```bash
mecon exp share <slug> coauthor@example.edu --role read_and_edit
mecon exp collaborators <slug>
mecon exp unshare <slug> coauthor@example.edu
```

`read_and_edit` can inspect, pull, validate, push previews, and request publication. It cannot
manage access. `full_manage` can also pause, close, and manage collaborators, but a new live version
still requires owner approval. Participants do not need accounts.

## Operational Rule

Use `mecon exp --help` and `mecon exp capabilities --json` for the current command surface. Use the
server compiler and journey result as the runtime acceptance gate; do not replace them with a local
interpretation of the protocol.
