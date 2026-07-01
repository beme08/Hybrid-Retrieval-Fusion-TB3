# Benchmark Findings — Hybrid Retrieval Fusion

> Post-submission findings document. This does not modify task logic, verifier logic, public/hidden cases, or instructions.

## Status

- Repo: [Hybrid-Retrieval-Fusion-TB3](https://github.com/beme08/Hybrid-Retrieval-Fusion-TB3) (PUBLIC)
- Branch: `level3-masking` (default branch, submitted)
- Task: `tasks/hybrid-retrieval-fusion`
- Submission status: Submitted on default branch `level3-masking`. Retired as final candidate after Codex solved v7.
- Current status: Post-submission finding. BENCHMARK_HYGIENE.md audit: PASS WITH NOTES.

## Public/private split

Full audit in `BENCHMARK_HYGIENE.md`. Summary:

- Public: 4 visible queries, synthetic corpus (280 documents), vector embeddings (280x96 float64), buggy search/fusion app code in `environment/app/hybrid_search/`. Shipping code with 10 planted bugs across 6 modules.
- Private: 12 hidden queries (`tests/fixtures/hidden_queries.json`), frozen expected rankings (`expected_hidden.json`), reference implementations (reference_bm25.py, reference_dense.py, reference_rrf.py).
- Hidden data lives exclusively in `tests/fixtures/`, never in `/app`.
- Clean hygiene split; no near-duplicate concern.

## Baseline gates

| gate | status | notes |
|---|---|---|
| reference/oracle | **1.0** | confirmed |
| nop/starter | **0.0** | confirmed |
| partial-fix audit | **1023/1024 states fail; only all-fixed passes** | each all-but-one-bug state fails 2+ hidden checks |
| Codex v7 standard | **1/1 solve (reward 1.0)** | solved legitimately via debugging; task retired |
| cheat/adversarial trials | not run | no cheat trials found in repo artifacts |

The all-fail target was not met: a frontier agent solved through the intended debugging/repair path.

## Leakage gates

Full audit in `BENCHMARK_HYGIENE.md`. Verified:

- Hidden queries and expected rankings not in `/app`
- Reference implementations not accessible from agent container
- SHA-256 tree digests guard against data mutation
- Anti-cheat scans for forbidden patterns
- `jobs/` in `.gitignore`

No known leakage risks.

## Agent outcome

- **Codex (gpt-5.5 xhigh):** Solved v7 with reward 1.0. Non-standard weighted-per-modality RRF with 10 interacting bugs across 6 modules. 1024-state partial-fix audit confirms: only all-fixed state passes.
- **Earlier versions (v1–v6):** Mixed results. v1 was too easy (both Codex and Claude solved). v3 recorded first 0.0. v6 leak-probe passed. Progressively hardened through 7 versions.
- Trajectory review (v7): Solve was legitimate — agent debugged through the service, implementing the public contract. No hidden data access or verifier exploit.
- **Result:** Task met all fairness and hygiene standards but failed the strict all-fail target. Retired.

## Lessons learned

- Hygiene/public-private split was mostly clean.
- Hidden retrieval bank was verifier-only.
- Public examples were clean-zone/smoke-style.
- Frontier agents solved through the intended debugging/retrieval path.
- Clean hygiene is not the same as strict all-fail hardness.
- Future hardening should save successful agent artifacts as replay baselines before tuning.

## Quality gates

| gate | status |
|---|---|
| A. Static/build gates | task compiles, Docker builds |
| B. Correctness: reference=1.0, nop=0.0 | confirmed |
| C. Shortcut: partial-fix audit confirmed (1023/1024 fail) | confirmed |
| D. Leakage: public-private isolation | confirmed via BENCHMARK_HYGIENE.md audit |
| E. Agent replay | not applied (task retired before replay baseline was saved) |
