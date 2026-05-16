---
name: ra-experiments
description: |
  SOP for managing online research experiments through mecon exp. Use when the user asks to initialize, create, edit, inspect, preview, publish, pause, close, export, or delete an experiment; asks about participant sessions, experiment status, live URL, preview URL, dashboard, response export, or experiment lifecycle operations.
---

# RA Experiments

Use this skill for experiment lifecycle operations. Keep preview, production,
session state, and exports tied to server truth.

## Lifecycle SOP

1. `mecon exp init <slug>` — scaffold local source project.
2. Edit JSON source files locally (experiment.json, apps/*/app.json, apps/*/pages/*.json).
3. `mecon exp push <dir>` — upload → remote compile → create preview.
4. Open preview URL and verify all pages.
5. `mecon exp publish <slug>` — preview → live.
6. Monitor sessions: `mecon exp status <slug>`.
7. Export data: `mecon exp export <slug>`.
8. `mecon exp pause <slug>` or `mecon exp close <slug>` when done.

## Core Commands

- `mecon exp init <slug>`: scaffold a local experiment source project.
- `mecon exp push <dir>`: upload source files → server compile → preview. Use `--publish` to also go live.
- `mecon exp pull <slug> -o <dir>`: download source files for an existing experiment.
- `mecon exp list`: list experiments.
- `mecon exp status <slug>`: show status, session counts, live URL, preview URL, dashboard URL.
- `mecon exp publish <slug>`: make the experiment live.
- `mecon exp pause <slug>`: stop accepting new sessions temporarily.
- `mecon exp close <slug>`: close collection.
- `mecon exp sessions <slug>`: list participant sessions.
- `mecon exp export <slug> --format wide --scope completed -o responses.csv`: export completed responses.
- `mecon exp export <slug> --format long --scope all -o responses-long.csv`: export event-like long data.
- `mecon exp open <slug>`: open the experiment dashboard.
- `mecon exp delete <slug> --yes`: delete experiment and all sessions.

Use `--json` for machine-readable status/list/session output.

## Source Project Schema

An experiment is defined by JSON source files in a directory:

```
experiments/<slug>/
  experiment.json          # project metadata + session config
  apps/<appId>/app.json    # app config + player fields
  apps/<appId>/pages/<pageId>.json  # one file per page
  stimuli.json             # optional: stimulus data
```

### experiment.json (required)

```json
{
  "schema": "experiment-project/1",
  "slug": "my-experiment",
  "title": "My Experiment",
  "locale": "en",
  "session": {
    "app_sequence": ["main"]
  },
  "treatments": {
    "factors": { "group": { "levels": ["control", "treatment"] } },
    "assignment": "between_subjects",
    "method": "hash"
  }
}
```

### apps/<appId>/app.json (one per app)

```json
{
  "schema": "experiment-app/1",
  "id": "main",
  "player": {
    "choice": { "type": "string", "label": "Choice" },
    "score": { "type": "int", "label": "Score", "min": 0, "max": 100 }
  }
}
```

Player fields define what data is collected. All fields are **required by default** unless `"required": false` is set.

### apps/<appId>/pages/<pageId>.json (one per page)

```json
{
  "schema": "experiment-page/1",
  "id": "welcome",
  "type": "display",
  "title": "Welcome",
  "content": "Markdown content here."
}
```

Page types: `display` (read-only), `form` (collects data).

Form pages use `components`:
```json
{
  "schema": "experiment-page/1",
  "id": "q1",
  "type": "form",
  "title": "Question 1",
  "components": [
    { "component": "ChoiceButtons", "field": "choice", "props": { "choices": [["A", "Option A"], ["B", "Option B"]] } },
    { "component": "Slider", "field": "score", "props": { "min": 0, "max": 100, "step": 1 } }
  ]
}
```

### Available Components

| Component | Use for | Key props |
|---|---|---|
| Instruction | Display text | `content` (markdown) |
| ChoiceButtons | Discrete choice | `choices: [value, label][]` |
| Slider | Continuous scale | `min, max, step, label` |
| TextInput | Free text | `placeholder` |
| LikertScale | Rating scale | `min, max, labels` |
| PriceList | MPL / multiple price list | `mode, rows, optionA, optionB` |
| BudgetLine | Budget constraint | `budgetX, budgetY, labelX, labelY` |
| MatrixQuestion | Grid questions | `rows, columns` |
| RankingTask | Rank items | `items` |
| LotteryBox | Lottery visualization | `probabilities, outcomes` |
| ExperimentTimer | Timed pages | `seconds` |

### Compiler Rules

The server compiler (`compileExperimentSourceProject`) validates:

- `experiment.json` must have `schema: "experiment-project/1"`, `slug`, `title`, `session.app_sequence`
- Each app must have `schema: "experiment-app/1"` and `id` matching the directory name
- Each page must have `schema: "experiment-page/1"`, `id`, `type`
- Form pages with required fields must have components that produce values (no empty forms)
- Component `field` names must match `player` field definitions
- `choices` arrays must have `[value, label]` tuples
- Template expressions `{{...}}` are validated at compile time

Compilation errors block push. Warnings are informational.

## Edit → Push → Preview → Publish

```bash
# New experiment
mecon exp init my-experiment
# edit files...
mecon exp push experiments/my-experiment
# open preview URL, verify
mecon exp publish my-experiment

# Edit existing experiment
mecon exp pull my-experiment -o experiments/my-experiment
# edit files...
mecon exp push experiments/my-experiment
# verify preview
mecon exp publish my-experiment

# One-shot edit + publish
mecon exp push experiments/my-experiment --publish
```

## Publish Gate

Before `publish`, verify:

- metadata has the correct slug, title, and locale
- `session.app_sequence` matches the intended flow
- all pages render correctly in preview
- required fields have matching components
- treatments, participant fields, and payoff fields are present
- export shape is known before real collection
- preview URL and live URL are not confused

Do not patch preview-only behavior if production uses the same renderer/runtime.
Fix the shared protocol or runtime contract.

## Status Semantics

- draft/preview: safe for inspection, not public collection.
- live: accepting participants.
- paused: not accepting new participants; existing state should remain inspectable.
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
