---
name: ra-data
description: |
  SOP for discovering, inspecting, querying, citing, and downloading RA Data datasets through mecon data. Use when the user asks about datasets, catalog search, metadata, citation, samples, Parquet/CSV download, DuckDB queries, signed query URLs, dataset description, schema, coverage, source attribution, or RA Data CLI usage.
---

# RA Data

Use this skill when the task is to find, inspect, query, cite, or download
published research datasets. Prefer Parquet as the canonical format; CSV is an
export convenience.

## Discovery SOP

1. Search or list candidates.
2. Inspect metadata and coverage before downloading.
3. Preview rows and schema with `head` or `describe`.
4. Query remotely with DuckDB when only a subset is needed.
5. Download only the needed resolution/version.
6. Capture citation and source attribution in the analysis notes.

## Core Commands

- `mecon data list`: list available datasets.
- `mecon data list --search <term>`: search the catalog.
- `mecon data list --family <family>`: filter by dataset family.
- `mecon data info <slug>`: show metadata, coverage, version, visibility, and
  citation.
- `mecon data cite <slug>`: print citation text.
- `mecon data sample <slug>`: preview a sample.
- `mecon data head <slug> --rows 20`: show first rows.
- `mecon data describe <slug>`: inspect profile, row count, columns, and stats.
- `mecon data query-url <slug>`: get a signed Parquet URL for DuckDB/httpfs.
- `mecon data query <slug> "select ..."`: run a local DuckDB query against the
  dataset.
- `mecon data download <slug> --format parquet`: download canonical data.
- `mecon data download <slug> --format csv`: export CSV only when needed.

Use `--json` when another program or agent needs structured output.

## Query Pattern

For analysis tasks, avoid downloading a full dataset just to inspect it:

```bash
mecon data describe <slug>
mecon data head <slug> --rows 20
mecon data query <slug> "select year, count(*) as n from data group by year order by year"
```

Download after the needed resolution, keys, and time range are clear.

## Dataset Acceptance Checks

Before using a dataset in downstream analysis, verify:

- slug and version
- geographic coverage
- time range
- row count and column count
- merge keys
- unit of observation
- source attribution and license/citation
- whether the selected resolution matches the research question

If metadata is missing or ambiguous, do not invent it. Ask for the dataset page,
source note, or maintainer clarification.

## Output Guidance

When reporting dataset use, include the slug, version, downloaded/query format,
date accessed, and citation. Keep user-facing language plain: "I checked the
coverage and sample rows before downloading" is better than storage or endpoint
details.
