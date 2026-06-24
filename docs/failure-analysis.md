# Check results, trial results, and failure analysis

Authoring/record document. **Do not invent results.** Cells marked `TODO` or
`PENDING` have not been run yet. Local-simulation results are recorded as
observed in the authoring sandbox; Docker, Harbor, and final agent trials are
Mac-side / Phase 8 and remain pending.

## Verifiable-environment framing

The task is designed as a verifiable environment rather than a subjective
benchmark. Agent failures are classified by observable verifier outcomes:
ranking mismatch, score inconsistency, determinism failure, schema failure, or
reward-hacking attempt.

## Run configuration

| Item | Value |
|------|-------|
| Task | `hybrid-retrieval-fusion` |
| Verifier mode | separate (`tests/Dockerfile` owns deps) |
| Canonical run params | `top_k=10`, `candidate_depth=25`, `rrf_k=60` |
| BM25 params | `k1=1.2`, `b=0.75` |
| Vector dim | 96 |
| Hidden queries | 12 |
| Visible queries | 4 |
| Corpus | 280 docs (240 main + 40 clean-zone) |
| Determinism seeds | `PYTHONHASHSEED=0`, `1` |
| Verifier deps (pinned) | `pytest==8.3.3`, `jsonschema==4.23.0`, `numpy==2.2.6` |
| `timeout_sec` | 5400 |

## Check results (local simulation)

Observed in the authoring sandbox by running `tests/test.sh` against a mirrored
`/app` layout (Docker not available in-sandbox).

| Check | Result |
|-------|--------|
| Oracle (`solve.sh` applied) | PASS — all verifier tests pass, exit 0 |
| Nop / broken (unmodified `/app`) | FAIL — verifier exits 1 |
| Determinism (`PYTHONHASHSEED=0` vs `1`) | PASS — identical fixtures + outputs |
| Verifier writes into `/app` | NONE — `/app` byte-identical before/after |
| `BUG_*` toggles in agent-visible code | NONE |
| Hidden/reference/expected leakage into `environment/` | NONE |
| Oracle idempotency (`solve.sh` re-run) | PASS — no-op on re-run, no `.rej` |
| Anti-cheat guards vs injected cheat | PASS — guards trip and fail the run |
| Docker image build (env + tests) | TODO (Mac-side) |
| Harbor / harness check | TODO (Mac-side) |

> Test counts and exact pass/fail tallies are reproducible by running the
> commands in `README.md`. They are intentionally not transcribed as fixed
> numbers here to avoid drift; regenerate them from the live run.

## 32-state partial-fix audit (Phase 3 record)

Recorded in `tasks/hybrid-retrieval-fusion/tests/fixtures/audit_report.json`.
Summary: only the all-fixed state passes every hidden query; each "4-of-5 fixed"
state (exactly one bug active) fails multiple hidden queries; rank-base is caught
by score/range checks, tie-break by the two-seed determinism check.

> Note: the audit was produced in Phase 3 against development `BUG_*` toggles,
> which were stripped from agent-visible code in Phase 6. The audit report is
> retained as a historical record; `tools/partial_fix_audit.py` is authoring
> tooling tied to those toggles and is not part of the shipped task.

## Trial results (PENDING — Phase 8)

No final agent trials have been run. Fill in only with real, executed results.

| Agent | Run | Result | Reward | Notes |
|-------|-----|--------|--------|-------|
| Codex | 1 | PENDING | — | — |
| Codex | 2 | PENDING | — | — |
| Codex | 3 | PENDING | — | — |
| Claude Code | 1 | PENDING | — | — |
| Claude Code | 2 | PENDING | — | — |
| Claude Code | 3 | PENDING | — | — |
| Gemini (optional) | 1 | PENDING | — | — |
| Codex `/cheat` | 1 | PENDING | — | — |
| Claude Code `/cheat` | 1 | PENDING | — | — |

## Failure analysis (template — fill after trials)

For each failing run, classify by verifier outcome and record the smallest
reproducing detail:

- **Ranking mismatch** — which modality (bm25/dense/fused) and first diverging
  rank; likely fusion sub-cause (raw mixing / truncate / collision).
- **Score inconsistency** — Bank 2 mismatch or fused-score out of `[0, 2/61]`;
  likely rank-base / constant error.
- **Determinism failure** — output differs across seeds; tie-break nondeterminism.
- **Schema failure** — malformed output / wrong ranks or lengths.
- **Reward-hacking attempt** — anti-cheat guard triggered (fixture/solution/test
  access, `.git`, network/public-solution route).

## Known caveats

- Docker and Harbor checks were not runnable in the authoring sandbox; they are
  Mac-side and currently pending.
- Final agent trials require maintainer approval and a fresh session without any
  authoring context (`CLAUDE.md`, `AGENTS.md`, `docs/`, `tools/`).
- The repository must remain private until the internet-enabled trials complete.
- `expert_time_estimate_hours` in `task.toml` is an authoring estimate, not a
  measured trial result.
