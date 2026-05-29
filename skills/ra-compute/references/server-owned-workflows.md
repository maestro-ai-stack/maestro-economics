# Server-Owned Workflows

Keep the public skill thin. It should route the agent to supported `mecon`
commands; the server owns product logic, policy, pricing, recommendations, and
runtime state.

## Public Skill Owns

- trigger language;
- safe command order;
- when to read a task-specific reference;
- client-facing wording;
- refusal to use raw API payloads for paid work.

## Server And CLI Own

- current resource catalog and prices: `mecon resources`;
- workspace and auth readiness: `mecon doctor`;
- resource validation and credit holds: `mecon submit`;
- lifecycle truth: `mecon status`, `mecon watch`;
- logs and artifacts: `mecon logs`, `mecon download`;
- utilization and next-action advice: `mecon profile`;
- billing, cancellation settlement, artifact retention, and provider routing.

## Product Data Loop

The useful moat is the server-side run graph, not the markdown:

- what task the user tried to run;
- selected resource and timeout;
- submit failure reason;
- queue/start/terminal timestamps;
- status/log/profile commands after submission;
- artifact download success;
- cancellation, timeout, OOM, or rerun pattern;
- profile recommendation and whether the next run followed it.

Use this to improve default profiles, templates, error messages, cost estimates,
and onboarding. Do not expose raw event internals to the customer.

## Boundary

If the needed answer depends on current catalog, cost, account state, job state,
or provider routing, call `mecon`. Do not guess from the skill.
