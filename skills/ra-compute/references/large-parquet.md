# Large Parquet First Pass

Use this guide when the user has a large Parquet dataset, only needs a few
fields, or says a laptop cannot run the analysis.

RA Compute can help with first-pass inspection. It is not a data warehouse
replacement.

## Good First-Pass Tasks

- count rows and groups;
- inspect schema and missingness;
- compute summary stats for selected fields;
- sample rows by year, region, entity, or treatment group;
- test whether a later full pipeline is worth running remotely.

## Local First

Try column-pruned DuckDB or Polars before paying for a full remote job. Parquet
is columnar, so a query that touches a few columns from a 1 TB dataset may still
be feasible if it does not materialize the full table.

Bad pattern:

```sql
select * from data
```

Better pattern:

```sql
select year, region_id, avg(outcome) as mean_y, count(*) as n
from read_parquet('data.parquet')
where year between 2010 and 2020
group by year, region_id
```

## Use Remote CPU When

- data already lives in the workspace and repeated upload/download would waste
  time;
- local memory spills or the laptop becomes unusable;
- the professor needs reproducible logs and artifacts;
- the same data will be reused across many runs.

## Workspace Locality

Do not upload stable large data repeatedly. Track stable inputs once, sync
changed code/config, and keep outputs under run artifacts.

For remote first-pass jobs, choose from the current server catalog:

```bash
mecon resources
mecon sync
mecon submit . --resource cpu-8c-32gb --timeout 3600
```

Prefer a modest CPU profile for the first pass, then use `mecon profile`,
logs, and artifacts to decide whether the next run needs more memory.
