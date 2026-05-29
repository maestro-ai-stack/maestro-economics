# Resource Selection

Choose the smallest resource that can answer the research question. Current
profiles and prices come from `mecon resources`; do not copy a static price
table into this guide.

## Default Rule

Default to CPU unless there is evidence the code benefits from GPU.

Use CPU for:

- regressions, fixed effects, IV, DiD, event studies, robustness checks;
- data cleaning, joins, sampling, summaries, and Parquet inspection;
- bootstrap/placebo/specification sweeps where each task is small;
- Python/R/Stata-style work without JAX/PyTorch/CUDA kernels.

Use GPU for:

- JAX/PyTorch/CUDA code with large tensor work;
- structural estimation, SMM, DDC, BLP, simulation, or optimization with heavy
  vectorized kernels;
- runs where CPU smoke passes but runtime is dominated by parallel numeric work.

## Profile Source

Run:

```bash
mecon resources
```

Use the returned resource names exactly. If a profile is not returned by the
server, do not submit it.

## Escalation Evidence

Escalate only after checking server-visible evidence:

- logs;
- `mecon profile <job_id>`;
- progress freshness;
- data size and column shape;
- whether the code is batched/vectorized.

Low GPU utilization usually means host-bound Python orchestration, serial
search, many small kernels, dynamic shapes, or too many callbacks. Improve the
workload before buying a bigger GPU.
