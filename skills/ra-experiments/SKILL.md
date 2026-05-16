---
name: ra-experiments
description: |
  SOP for managing online research experiments through mecon exp. Use when the user asks to initialize, inspect, preview, publish, pause, close, export, or delete an experiment; asks about participant sessions, experiment status, live URL, preview URL, dashboard, response export, or experiment lifecycle operations.
---

# RA Experiments

Use this skill for experiment lifecycle operations. Keep preview, production,
session state, and exports tied to server truth.

## Lifecycle SOP

1. Initialize or inspect the experiment source.
2. Validate metadata, session sequence, and app/page structure before publish.
3. Use preview before live collection.
4. Publish only after preview and export shape are acceptable.
5. Monitor sessions while live.
6. Pause or close deliberately.
7. Export response data from server truth after collection.

## Core Commands

- `mecon exp init <slug>`: scaffold a local experiment source project.
- `mecon exp list`: list experiments.
- `mecon exp status <slug>`: show status, session counts, live URL, preview URL,
  and dashboard URL.
- `mecon exp publish <slug>`: make the experiment live.
- `mecon exp pause <slug>`: stop accepting new sessions temporarily.
- `mecon exp close <slug>`: close collection.
- `mecon exp sessions <slug>`: list participant sessions.
- `mecon exp export <slug> --format wide --scope completed -o responses.csv`:
  export completed responses.
- `mecon exp export <slug> --format long --scope all -o responses-long.csv`:
  export event-like long data.
- `mecon exp open <slug>`: open the experiment dashboard.
- `mecon exp delete <slug>`: delete only when explicitly intended.

Use `--json` for machine-readable status/list/session output.

## Publish Gate

Before `publish`, verify:

- metadata has the correct slug, title, and locale
- `session.json` has the intended `app_sequence`
- app/page JSON renders in preview
- required treatments, participant fields, and payoff fields are present
- export shape is known before real collection
- preview URL and live URL are not confused

Do not patch preview-only behavior if production uses the same renderer/runtime.
Fix the shared protocol or runtime contract.

## Status Semantics

- draft/preview: safe for inspection, not public collection.
- live: accepting participants.
- paused: not accepting new participants; existing state should remain
  inspectable.
- closed: collection finished; export from server truth.

When reporting status, include live URL, preview URL if present, dashboard URL,
session count, completed count, and active count.

## Export SOP

For analysis-ready output:

1. Run `mecon exp status <slug>`.
2. Run `mecon exp sessions <slug>` if completion state is unclear.
3. Export completed sessions first:
   `mecon exp export <slug> --format wide --scope completed -o responses.csv`.
4. Use long format when event order, page-level answers, or timing matter.
5. Keep exported files labeled with slug, format, scope, and export date.

Do not treat browser-visible preview state as data truth. Exports must come from
server-side experiment responses.
