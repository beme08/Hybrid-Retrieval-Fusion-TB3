# Failure Analysis — Hybrid Retrieval Fusion

## Candidate retirement

The hybrid retrieval fusion task was retired as a final candidate because
Codex solved v7 with reward 1.0.

**Why it was retired:** the task had a fair, explicit contract with a
non-standard weighted-per-modality RRF formula, ten interacting bugs across
six modules, and a 1024-state partial-fix audit. Despite this, a careful
frontier agent could read the public contract and repair the service through
locally composable changes. The bugs were independent enough that fixing them
sequentially worked.

**Lesson:** hidden tests, oracle/nop validation, and anti-cheat guards are
necessary but not sufficient for task difficulty. When the public contract is a
complete algorithmic repair recipe, a capable agent can implement it directly
without needing to discover the hidden regime through trial and error.

## Historical trial summary

See [`docs/trial-results.md`](trial-results.md) for the full per-version trial
history. The key milestones:

- **v1:** Codex 3/3, Claude 3/3 — both solved it
- **v3:** First Codex 0.0 recorded (v3-codex-run-1)
- **v4:** Second Codex 0.0 recorded, exposed instruction fairness issue
- **v5–v6:** Iterative hardening
- **v7:** Non-standard fusion contract, Codex solved 1/1 — task retired
