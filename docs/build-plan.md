# Build plan — hybrid-retrieval-fusion (TB3 task)

This document summarizes the full phase plan. It is authoring guidance only and
is never shown to evaluation agents. See `CLAUDE.md` for the normative contract,
verifier design, and anti-cheat rules.

## Goal

Build one original TB3-compatible task in **separate-verifier mode**. The agent
inherits a realistic BM25 + dense retrieval service whose lexical and cosine
rankings are correct, but whose **fusion layer** is broken. The agent repairs
the fusion layer so hidden BM25, dense, and fused rankings match a
verifier-owned reference. Reward is deterministic (no LLM judge).

## Target repo structure

```
CLAUDE.md
AGENTS.md
README.md
docs/build-plan.md
tools/                         # repo-root build/review tools (NOT task-visible)
  generate_fixtures.py
  validate_fixture_coverage.py
  partial_fix_audit.py
tasks/
  hybrid-retrieval-fusion/
    instruction.md
    task.toml
    environment/
      Dockerfile
      data/
        corpus.json
        visible_queries.json
        vectors.npy
    solution/
      solve.sh
      patches/
        001-remove-raw-score-mixing.patch
        002-fix-rrf-rank-base.patch
        003-fuse-before-truncate.patch
        004-preserve-opaque-doc-ids.patch
        005-deterministic-tie-break.patch
    tests/
      Dockerfile
      test.sh
      test_fusion.py
      fixtures/
        reference_bm25.py
        reference_dense.py
        reference_rrf.py
        hidden_queries.json
        hidden_vectors.npy        # if needed
        expected_hidden.json      # frozen, generated from reference
```

App/service code copied into `/app` by `environment/Dockerfile`:

```
/app/
  hybrid_search/
    __init__.py
    models.py          # Document, Query, RankingEntry
    config.py          # SearchConfig
    pipeline.py        # RetrievalPipeline
    fusion.py          # RRF over typed RankingEntry lists
    serialization.py   # strip internal fields -> output schema
    rankers/
      __init__.py
      bm25.py
      dense.py
  scripts/
    run_search.py
  data/
    corpus.json
    vectors.npy
    visible_queries.json
```

## Phase plan

### Phase 0 — Setup & scaffold
Inspect repo; clone `evals-skills` outside the task/repo; create `CLAUDE.md`,
`AGENTS.md`, `docs/build-plan.md`; initialize TB3 scaffold. No retrieval logic,
no bugs.
**Gate:** task folder exists; scaffold sane; authoring files exist; no
hidden/test/solution leakage into `environment/`.

### Phase 1 — Clean reference + realistic pipeline
Build app structure; clean deterministic BM25/dense/RRF; full CLI; typed
pipeline; serializer. No bugs.
**Gate:** clean app runs on visible data; deterministic; `fusion.py` does not
call rankers directly.

### Phase 2 — Fixtures + independent reference
`tools/generate_fixtures.py`, `tools/validate_fixture_coverage.py`; independent
reference files under `tests/fixtures`; visible + hidden fixtures; frozen
`expected_hidden.json`; coverage + non-degeneracy validated.
**Gate:** reference deterministic across `PYTHONHASHSEED=0/1`; hidden queries
activate intended behaviors; `expected_hidden` from reference, not `/app`.

### Phase 3 — Plant bugs + partial-fix audit
Plant five scoped fusion-layer bugs; dev-only `BUG_*` toggles;
`tools/partial_fix_audit.py` over 32 bitmask states; strip `BUG_*` logic before
packaging.
**Gate:** broken service fails hidden checks for intended reasons; BM25/dense
stay correct; only all-fixed passes; each 4-of-5 fails ≥2–3 hidden queries;
shipped `/app` has natural buggy paths only.

### Phase 4 — Separate verifier
`tests/Dockerfile` (pinned deps); `tests/test.sh`; `tests/test_fusion.py`
(schema + Bank 1 + Bank 2 + determinism + fused range + CTRF guard +
diagnostics).
**Gate:** verifier runs locally; failures informative; verifier never
overwrites `/app`; hidden fixtures only in `tests/`.

### Phase 5 — Oracle solution
`solution/solve.sh`; ordered patches 001–005 with explanatory headers; run
oracle.
**Gate:** oracle = 1.0.

### Phase 6 — Nop + anti-cheat hardening
Run nop; add anti-cheat guards; run explicit cheat attempts.
**Gate:** nop = 0.0; all planned `/cheat` attempts score 0.0; honest fixes pass.

### Phase 7 — Checks, matrix plan, docs
Static checks; harbor check; README compliance table + security/integrity
section + matrix execution plan; final trial commands. No invented results.
**Gate:** static + rubric checks pass or are documented with next fix; README
has placeholders only where trials have not run.

### Phase 8 — Final agent trials (requires maintainer approval)
Codex x3; Claude Code x3; Gemini optional; `/cheat` for Codex + Claude Code;
harbor analyze; failure analysis. Stop and ask before any paid/authenticated
run.

## Planted fusion bugs (Phase 3)

1. **raw-score mixing** — adds raw BM25 scores to cosine similarities instead of
   rank-based RRF. *Masks* the RRF rank-base bug.
2. **wrong RRF rank base / constant** — 0-based rank or wrong constant (correct
   max is `2/61`).
3. **truncate-before-fusion** — truncates each modality to `top_k` before
   fusing instead of `candidate_depth`. *Masks* the ID-collision bug.
4. **suffix-based doc-ID collision** — treats `news_001`, `paper_001`,
   `faq_001` as the same doc via suffix. Organic, not commented as a bug.
5. **nondeterministic tie-break** — dict/set iteration or unstable sort instead
   of `doc_id` ascending. Self-masking.

## Fixture targets (Phase 2)

- Corpus ~240 docs (200–300). Hidden queries ~12; visible ~4. Vector dim ~96
  (64–128), 6-decimal precision, `.npy`. BM25/dense tol `1e-9`; fused RRF tol
  `1e-12`.
- Visible queries test schema/smoke; must not expose suffix-collision,
  truncate-before-fusion, or raw-score/rank-base flips; visible tie uses
  dissimilar prefixes (e.g. `news_042` vs `paper_091`).
- Hidden coverage: raw-score mixing ≥3 activating; wrong rank base ≥2;
  tie nondeterminism ≥2; ID collision ≥3–4; truncate-before-fusion ≥2;
  ≥2–3 queries with 3+ bugs active simultaneously.
