# Check results, trial results, and failure analysis

Authoring/record document. **Do not invent results.** Cells marked `TODO` or
`PENDING` have not been run yet. Local-simulation results are recorded as
observed in the authoring sandbox; Harbor and final agent trials remain Phase 8
and require maintainer approval.

## Verifiable-environment framing

The task is designed as a verifiable environment rather than a subjective
benchmark. Agent failures are classified by observable verifier outcomes:
ranking mismatch, score inconsistency, determinism failure, schema failure, or
reward-hacking attempt.

v5 was valid and fair, but a Codex run passed it cleanly. v6 added generated
cross-module invariant checks, then a Codex leak-probe run solved v6 cleanly
with normal source patches. v7 reduces instruction handholding and adds a
tenth fair fused-score precision defect. This is development/build validation,
not a final benchmark `/run` trial.

## Run configuration

| Item | Value |
|------|-------|
| Task | `hybrid-retrieval-fusion` |
| Verifier mode | separate (`tests/Dockerfile` owns deps) |
| Canonical run params | `top_k=10`, `candidate_depth=25`, `rrf_k=60` |
| Generated run params | `top_k=4`, `candidate_depth=6`, `rrf_k=7`; `top_k=3`, `candidate_depth=3`, `rrf_k=17` |
| BM25 params | `k1=1.2`, `b=0.75` |
| Vector dim | 96 |
| Static hidden queries | 12 |
| Generated hidden banks | 2 |
| Hidden checks in partial audit | 14 |
| Visible queries | 4 |
| Corpus | 280 docs (240 main + 40 clean-zone) |
| Determinism seeds | `PYTHONHASHSEED=0`, `1` |
| Verifier deps (pinned) | `pytest==8.3.3`, `jsonschema==4.23.0`, `numpy==2.2.6` |
| `timeout_sec` | 5400 |

## Check results (local simulation)

Observed in the authoring sandbox by running `tests/test.sh` against a mirrored
`/app` layout and, where noted, local Docker verifier images.

| Check | Result |
|-------|--------|
| Oracle (`solve.sh` applied) | PASS — 111 verifier tests pass, reward 1.0, exit 0 |
| Nop / broken (unmodified `/app`) | FAIL as expected — 66 failed / 45 passed, reward 0.0, exit 1 |
| Determinism (`PYTHONHASHSEED=0` vs `1`) | PASS — identical fixtures + outputs |
| `/app/data` mutation guard | PASS — digest unchanged before/after verifier run |
| `BUG_*` toggles in agent-visible code | NONE |
| Hidden/reference/expected leakage into `environment/` | NONE |
| Full partial-fix audit | PASS — full 1024-state audit; only all-fixed passes |
| Oracle idempotency (`solve.sh` re-run) | PASS — second run reports already fixed |
| Anti-cheat guards vs injected cheat | PASS — `/tests` mutation smoke exits 1 with reward 0.0 before pytest; full `/cheat` trials remain pending |
| Docker environment image build | PASS — built locally as `hrf-v7-env` |
| Docker verifier image build | PASS — built locally as `hrf-v7-test` |
| Docker verifier oracle run | PASS — 111 verifier tests pass, reward 1.0 |
| Docker verifier nop run | FAIL as expected — verifier exits 1, reward 0.0 |
| Runtime leak audit on environment image | PASS — `/app` contains only app/data files; no hidden/test/solution/repo metadata hits |
| Harbor parser validation (`TaskConfig`) | PENDING in v7 |
| `TaskModel.is_valid_dir(disable_verification=False)` | PENDING in v7 |
| Harbor oracle run | PENDING in v7 |
| Harbor nop run | PENDING in v7 |

> Test counts and exact pass/fail tallies are reproducible by running the
> commands in `README.md`. They are intentionally not transcribed as fixed
> numbers here to avoid drift; regenerate them from the live run.

## 1024-state partial-fix audit (v7 record)

Recorded in `tasks/hybrid-retrieval-fusion/tests/fixtures/audit_report.json`.
Summary: only the all-fixed state `1111111111` passes every hidden check. The
broken state passes 0/14 checks; the all-fixed state passes 14/14. Each
all-but-one fixed state fails at least 2/14 checks, including
`QUERY_ORDER_IDENTITY` and `FUSED_SCORE_PRECISION`. Rank-base and serialization
are caught by score/range checks, tie-break by the two-seed determinism check,
vector alignment by static and generated dense rankings, query identity/order
by duplicate-ID generated inputs, and fused precision by RRF score
recomputation.

## Trial results (PENDING — Phase 8)

Final trial results are reported at the agent-harness configuration level, not
as pure base-model scores. Each run records the agent harness, model, reasoning
setting, timeout, verifier result, and observed failure class.

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
- **Score inconsistency** — Bank 2 mismatch or fused-score out of
  `[0, 2 / (rrf_k + 1)]`;
  likely rank-base / constant error.
- **Determinism failure** — output differs across seeds; tie-break nondeterminism.
- **Schema failure** — malformed output / wrong ranks or lengths.
- **Reward-hacking attempt** — anti-cheat guard triggered (fixture/solution/test
  access, `.git`, network/public-solution route).

## Known caveats

- Harbor checks and final trials have not been rerun for v7 in this turn.
- Final agent trials require maintainer approval and a fresh session without any
  authoring context (`CLAUDE.md`, `AGENTS.md`, `docs/`, `tools/`).
- Official `/run` trials must use only the task's `instruction.md`, with no
  development/build instructions or extra hints.
- The repository must remain private until the internet-enabled trials complete.
- `expert_time_estimate_hours` in `task.toml` is an authoring estimate, not a
  measured trial result.
