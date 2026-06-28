# Trial Results — hybrid-retrieval-fusion

Trial results across all task versions. Historical runs (v1–v7) are documented
for reference; the frozen candidate for public review is the `level3-masking`
branch at tag `retrieval-v7-final`.

## Mac-side validation status

| Check | Result |
|-------|--------|
| Environment Docker image build | ✅ passed |
| Verifier Docker image build | ✅ passed |
| Harbor installation / static checks | ⚠️ blocked — local PyPI CDN TLS failure (`files.pythonhosted.org` SSL EOF) |
| Oracle / nop via Harbor | ⬜ pending |
| Final `/run` and `/cheat` | ⬜ not run |

Oracle job path (for reference): `jobs/2026-06-25__00-20-22/result.json`
Nop job path (for reference):    `jobs/2026-06-25__00-21-17/result.json`

## Benchmark interpretation

This task is a Harbor/TB3-style measurement protocol, not a general claim about
model intelligence. Reported rewards depend on the exact task files, hidden
verifier, Docker runtime, agent harness, model, reasoning setting, and retry
policy.

---

## v1 (commit `481995e`)

Original frozen candidate. Both frontier models solved it 3/3, indicating the
task was easier than intended.

### Codex — 3/3 pass

| Run | Reward | Exceptions | CTRF | Notes |
|-----|--------|------------|------|-------|
| 1/3 | 1.0 | 0 | 82/82 pass | Model: gpt-5.5, reasoning: xhigh |
| 2/3 | 1.0 | 0 | 82/82 pass | Model: gpt-5.5, reasoning: xhigh |
| 3/3 | 1.0 | 0 | 82/82 pass | Model: gpt-5.5, reasoning: xhigh |

### Claude Code — 3/3 pass

| Run | Reward | Exceptions | CTRF | Notes |
|-----|--------|------------|------|-------|
| 1/3 | 1.0 | 0 | 82/82 pass | Initial run timed out during setup; retry succeeded |
| 2/3 | 1.0 | 0 | 82/82 pass | Setup retries hit transient TLS EOF, then succeeded |
| 3/3 | 1.0 | 0 | 82/82 pass | Total runtime: 1406.2s, solve time: 200.6s |

---

## v2 hardening (commit `65e2e728`)

Cross-module defect relocation. Reduced Codex to 2 trials.

### Codex — 2/2 pass

| Run | Reward | Exceptions | CTRF | Notes |
|-----|--------|------------|------|-------|
| 1/2 | 1.0 | 0 | 82/82 pass | 248.3s total (133.1s agent execution) |
| 2/2 | 1.0 | 0 | 82/82 pass | 4m 39s total |

---

## v3 (commit `ba0af7d`)

Hardened with diagnostic complexity. First version to score a model 0.0.

### Codex — 1/2 pass

| Run | Reward | Exceptions | CTRF | Notes |
|-----|--------|------------|------|-------|
| 1/2 | **0.0** | None | 72 pass / 10 fail | Agent execution: 274.0s |
| 2/2 | 1.0 | None | 82/82 pass | 284.5s total (151.8s agent execution) |

### Claude Code — infra failure (not counted)

Trial setup timed out before task execution. Claude native build installation
failed. Not counted as a model result.

---

## v4 additive (commit `4e19b14`)

### Codex — 2/3 counted passes

| Run | Reward | Exceptions | CTRF | Notes |
|-----|--------|------------|------|-------|
| 1/3 | — | Setup failure | — | Infra issue, not counted |
| 2/3 | **0.0** | None | — | Agent interpreted ambiguous BM25 candidate-depth wording literally and padded zero-score BM25 documents. Exposed instruction fairness issue — fixed for v5. |
| 3/3 | 1.0 | None | — | |

---

## v5–v6

Intermediate hardening iterations. Codex leak-probe run solved v6 cleanly with
normal source patches.

---

## v7 contract-level (commit `0be3338`)

Non-standard weighted-per-modality RRF fusion contract (`w_dense=2.0`,
`k_dense=rrf_k//2`), 10 interacting bugs, 1024-state partial-fix audit.

### Codex — 1/1 pass

| Run | Reward | Exceptions | CTRF | Notes |
|-----|--------|------------|------|-------|
| 1/1 | 1.0 | 0 | — | Contract-level debugging pass |

### Final full trial matrix (3× Codex + 3× Claude + cheat)

⬜ **not run** — requires maintainer approval.

## Partial-fix audit (v7)

1024-state bitmask over 10 bugs (`BUG_1`–`BUG_10`). Only the all-fixed state
(`1111111111`) passes all hidden checks.

| Metric | Value |
|--------|-------|
| Total states | 1024 |
| States passing all checks | 1 (all-fixed) |
| States failing | 1023 |
| Minimum failed checks per all-but-one state | 2 |

All-but-one states reliably fail `QUERY_ORDER_IDENTITY` and
`FUSED_SCORE_PRECISION`. Rank-base and serialization caught by score/range
checks, tie-break by two-seed determinism, vector alignment by static and
generated dense rankings, query identity/order by duplicate-ID generated inputs,
and fused precision by RRF score recomputation.
