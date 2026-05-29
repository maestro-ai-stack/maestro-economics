# Customer Language

Use plain language for economists and RAs. Mention platform internals only if
the user asks.

## Useful Phrases

CPU CLI issue:

> CPU jobs are available through `--resource`, not `--gpu`. Please upgrade to
> `maestro-economics>=0.7.2`, then run `mecon submit . --resource cpu-4c-16gb`.

Timeout:

> The run reached its time budget. That is not necessarily a crash. I am checking
> whether partial outputs or checkpoints are available before recommending a rerun.

Code failure:

> The logs show a script/runtime error rather than a platform outage. The next
> step is to fix the script and run a smaller smoke job.

Low GPU utilization:

> A bigger GPU is unlikely to help yet. The job is not keeping the current GPU
> busy; batching or vectorization is the better next step.

Large Parquet:

> Because the query only needs a few Parquet columns, we should first try a
> column-pruned DuckDB/Polars query instead of running or uploading the full table.

Cancellation:

> Cancellation has been requested. I am waiting for the terminal `cancelled`
> state before treating it as stopped.

## Avoid

- Do not say "Modal", "callback", "R2", "worker", or "serverless" in client
  updates unless needed.
- Do not say "fixed", "online", or "done" until the live command/output proves it.
- Do not tell the customer to run commands if you can run them.
- Do not claim a rerun is safe until existing artifacts have been checked.
