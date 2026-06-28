# CLAUDE.md — Authoring guidance for implementation agents

> **Scope of this file:** This file guides agents that are *building* the
> benchmark task. It is **not** the benchmark instruction shown to evaluation
> agents. The evaluation-facing instruction lives in
> `tasks/hybrid-retrieval-fusion/instruction.md` and must never reference this
> file, the bug list, the phase gates, or any authoring tooling.

## Active repo

- **Repo:** https://github.com/beme08/Hybrid-Retrieval-Fusion-TB3
- This is the repo we update and ship. Do **not** create a new repo unless the
  maintainer explicitly asks. Work in this repo.
- Keep the repo **private** until all required final trials are complete, so
  internet-enabled evaluation agents cannot fetch our solution.

## Task name

`hybrid-retrieval-fusion`

## What we are building

One original Terminal-Bench 3 (TB3) compatible task: a deterministic hybrid
retrieval fusion debugging task. The agent inherits a BM25 + dense vector
retrieval service. BM25 lexical ranking and dense cosine ranking are correct.
The regression lives in the **fusion layer**. The agent must repair the fusion
layer so hidden BM25, dense, and fused rankings match a verifier-owned
reference.

This is a coding assignment. Final delivery is a **standalone GitHub
repo**, not a Terminal-Bench PR. The repo documents check results, trial
results, and a brief failure analysis.

## Phase gates (strict — do not skip or collapse)

Proceed autonomously from one phase to the next **only** if the current gate
passes. If a gate fails, stop and report the blocker. Never run final paid /
authenticated agent trials without explicit maintainer approval.

0. **Setup & scaffold** — repo inspected; TB3 scaffold exists; authoring files
   exist; no hidden/test/solution leakage into `environment/`.
1. **Clean reference + realistic pipeline** — clean deterministic
   BM25/dense/RRF; full CLI; typed pipeline; serializer; runs on visible data;
   `fusion.py` does not call rankers directly. **No bugs yet.**
2. **Fixtures + independent reference** — deterministic synthetic fixtures;
   independent reference under `tests/fixtures`; frozen `expected_hidden.json`
   generated from the reference (not from `/app`); coverage matrix validated;
   deterministic across `PYTHONHASHSEED=0/1`.
3. **Plant bugs + partial-fix audit** — five scoped fusion-layer bugs; 32-state
   bitmask audit; only all-fixed passes; each 4-of-5 fails ≥2–3 hidden queries;
   strip all `BUG_*` toggle logic before packaging.
4. **Separate verifier** — `tests/` owns verifier deps (pinned); schema +
   Bank 1 + Bank 2; determinism check; fused-range guard; CTRF count guard;
   verifier never overwrites `/app`; hidden fixtures only in `tests/`.
5. **Oracle solution** — ordered patches 001–005; oracle scores **1.0**.
6. **Nop + anti-cheat hardening** — nop scores **0.0**; all planned `/cheat`
   attempts score 0.0; honest source fixes still pass.
7. **Checks, matrix plan, docs** — static checks; harbor check; README with
   compliance table, security section, matrix plan; placeholders only where
   trials have not run.
8. **Final agent trials** — *requires maintainer approval.* Codex x3, Claude
   Code x3, Gemini optional, `/cheat` for Codex + Claude Code, harbor analyze,
   failure analysis.

## Architecture rules

- Build a **realistic integration surface before planting bugs**. A full
  rewrite should be more work than repairing the existing service — but the
  structure must stay realistic, not artificial.
- Do **not** collapse the service into one script.
- Required modules: `hybrid_search/{models,config,pipeline,fusion,serialization}.py`
  and `hybrid_search/rankers/{bm25,dense}.py`, plus `scripts/run_search.py`.
- `fusion.py` consumes typed precomputed `RankingEntry` lists + `SearchConfig`.
  It must **not** call `bm25.search()` or `dense.search()` directly.
- `serialization.py` strips internal `RankingEntry` fields and emits the exact
  output schema.
- `RankingEntry` carries internal fields not emitted in JSON: `doc_id`,
  `score`, `modality`, `source_rank`, `raw_score`.

## Core task contract (normative)

- Fusion: `score(doc) = sum over modalities of 1 / (RRF_K + rank_m(doc))`.
- `RRF_K = 60`; ranks are **1-based**; max fused score (two modalities) is
  `2/61`.
- Doc IDs are **opaque strings** — never parse or normalize them.
- Sort fused results by descending fused score; tie-break by `doc_id` ascending.
- Each modality returns `candidate_depth` results; **fuse the union first**,
  then truncate fused ranking to `top_k`.
- BM25: tokenizer lowercase regex `[a-z0-9]+`; `k1=1.2`; `b=0.75`;
  `idf = log(1 + (N - df + 0.5) / (df + 0.5))`; sort by `(-score, doc_id)`.
- Dense: cosine over shipped vectors; sort by `(-score, doc_id)`.

## Verifier rules

- **Separate-verifier mode.** `tests/Dockerfile` owns all verifier deps,
  pinned (pytest, numpy, CTRF plugin, verifier-only packages). No runtime
  installs in `test.sh`.
- `environment/Dockerfile` copies only agent-visible data + app code. It does
  **not** copy `solution/` or `tests/`. `apt-get update` before install; do not
  pin apt versions; clean apt cache.
- Bank 1 (external correctness): emitted BM25 / dense / fused doc-ID rankings
  match expected hidden rankings.
- Bank 2 (internal consistency): recompute fused scores from emitted BM25/dense
  sub-rankings via reference RRF; compare to emitted fused scores, tol `1e-12`.
- Determinism: run twice with `PYTHONHASHSEED=0` and `=1`; outputs identical.
- Fused range guard: `0 <= score <= 2/61 + 1e-9`.
- `expected_hidden.json` is generated from the independent reference, frozen,
  committed, and **loaded** at grade time — never regenerated from `/app`.
- Reward is deterministic. **No LLM judge.** No single-bank verifier. No broad
  source-code grep as the primary verification.

## Anti-cheat rules

Treat reward hacking as an environment-security problem, not a prompt problem.
Add **narrow deterministic** guards/checks (not broad blocklists) for: reading
`/tests` or `/solution`; importing tests/fixtures; references to
`expected_hidden`; output fabrication after a legitimate search; `.git`/history
leakage; public-solution fetch attempts; direct hidden-fixture access. External
RRF knowledge is fair, but should not solve the task unless the agent also
preserves the typed pipeline, serializer, candidate-depth semantics, opaque
doc IDs, correct tie-break, and correct output schema.

## Instruction-completeness rules

`instruction.md` must be complete and fair: full CLI contract, output schema,
RRF formula, candidate-depth semantics, opaque-doc-ID rule, tie-break, and the
scope statement ("BM25 lexical ranking and dense cosine ranking are correctly
implemented. The regression is in the fusion layer."). It must **not** reveal
the bug count or hidden activation patterns. It must end with the exact
required suffix where `N == [agent].timeout_sec`:

> You have N seconds to complete this task. Do not cheat by using online
> solutions or hints specific to this task.

## Hard rules (load-bearing reminders)

- **Do not collapse the pipeline.**
- **Do not plant bugs before a clean reference exists.**
- **Do not start final trials before oracle = 1.0 and nop = 0.0.**
- Do **not** copy `tests/` or `solution/` into `environment/`.
- Do **not** put hidden fixtures, expected outputs, verifier files, or solution
  files into the agent environment.
- Do **not** invent check results or trial results — document only what ran.
- Final `/run` evaluation agents must **not** receive any authoring guidance
  (this file, `AGENTS.md`, `docs/build-plan.md`, `tools/`, evals-skills).

## Authoring-only tooling

`hamelsmu/evals-skills` may be used as authoring/review help only, cloned
**outside** the task folder (ideally outside this repo). Do not copy it into
`tasks/hybrid-retrieval-fusion`, make it a task or verifier dependency, or hand
it to final `/run` or `/cheat` agents.
