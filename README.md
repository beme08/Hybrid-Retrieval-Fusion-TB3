# Hybrid-Retrieval-Fusion-TB3
I spend a week doing this take home task for a YCombinato Ai company below are the details

An original **Terminal-Bench 3 (TB3)** compatible coding task:
`hybrid-retrieval-fusion`. The agent inherits a small hybrid document-retrieval
service (BM25 + dense cosine, merged with Reciprocal Rank Fusion) whose lexical
and dense rankers are correct but whose **fusion layer** contains a regression.
The agent must repair the fusion layer so that hidden BM25, dense, and fused
rankings match a verifier-owned reference.

Reward is **deterministic** — exact ranking comparison plus score recomputation,
schema, range, and determinism checks. There is **no LLM judge**.

## Relationship to / TB3

This is aAI coding assignment, delivered as a **standalone GitHub
repository** (not a Terminal-Bench upstream PR). The repo is laid out in TB3
separate-verifier form under `tasks/hybrid-retrieval-fusion/` and documents its
own check results, trial results, and failure analysis. The repository is kept
**private** until the internet-enabled final agent trials are complete, so that
internet-enabled evaluation agents cannot fetch the solution.

## Repository layout

```
tasks/hybrid-retrieval-fusion/
  instruction.md            # agent-facing task (CLI, schema, RRF contract)
  task.toml                 # TB3 metadata; separate-verifier; allow_internet
  environment/
    Dockerfile              # builds /app from app/ + data/ only
    app/hybrid_search/...    # models, config, pipeline, fusion, serialization, rankers
    app/scripts/run_search.py
    data/                    # corpus.json, vectors.npy, visible_queries.json
  solution/
    solve.sh                # idempotent oracle (applies patches 001..005)
    patches/00{1..5}-*.patch
  tests/                    # SEPARATE verifier (not copied into the env image)
    Dockerfile              # pinned: pytest, jsonschema, numpy
    test.sh                 # runs /app twice (seeds 0/1) then grades; no installs
    test_fusion.py          # schema + Bank 1 + Bank 2 + range + determinism
    test_anticheat.py       # narrow deterministic anti-cheat guards
    conftest.py             # emits CTRF report
    fixtures/               # hidden_queries, expected_hidden, reference_*, reports
docs/
  build-plan.md             # authoring-only phase plan
  failure-analysis.md       # check/trial results + failure analysis (scaffold)
tools/                      # authoring-only build/review tooling (NOT agent-visible)
CLAUDE.md, AGENTS.md        # authoring-only guidance (NOT agent-visible)
```

`tools/`, `docs/`, `CLAUDE.md`, and `AGENTS.md` are **authoring artifacts**.
They are never copied into `environment/` and are never shown to evaluation
agents.

## Expected validation flow

1. **Static / structural checks** — layout, no leakage into `environment/`, no
   `BUG_*` toggles in agent-visible code, executable bits.
2. **Oracle = 1.0** — `solution/solve.sh` applied to a copy of `/app` makes the
   verifier pass every test.
3. **Nop = 0.0** — the unmodified (broken) `/app` fails the verifier.
4. **Docker build** — build the environment and verifier images (Mac-side).
5. **Harbor checks** — TB3 harness static/structural validation (Mac-side).
6. **Final agent trials** — Codex ×3, Claude Code ×3, optional Gemini, plus
   `/cheat`, run in fresh sessions (Phase 8; requires maintainer approval).

## Commands

Local simulation (used in authoring; Docker not required):

```bash
# build a throwaway /app = app code + data
APP=$(mktemp -d)/app; mkdir -p "$APP"
cp -r tasks/hybrid-retrieval-fusion/environment/app/. "$APP/"
cp -r tasks/hybrid-retrieval-fusion/environment/data "$APP/data"

# nop / broken (expect failure, exit 1)
APP_DIR="$APP" TESTS_DIR=tasks/hybrid-retrieval-fusion/tests \
  bash tasks/hybrid-retrieval-fusion/tests/test.sh

# oracle (expect pass, exit 0)
APP_DIR="$APP" bash tasks/hybrid-retrieval-fusion/solution/solve.sh
APP_DIR="$APP" TESTS_DIR=tasks/hybrid-retrieval-fusion/tests \
  bash tasks/hybrid-retrieval-fusion/tests/test.sh

# authoring checks
python3 tools/generate_fixtures.py          # regenerate fixtures (deterministic)
python3 tools/validate_fixture_coverage.py  # Phase 2 coverage/determinism gate
```

Mac-side / harness (not run in this authoring sandbox):

```bash
# Docker image builds
docker build -t hrf-env  tasks/hybrid-retrieval-fusion/environment
docker build -t hrf-test tasks/hybrid-retrieval-fusion/tests

# Harbor / TB3 harness checks and final trials (Phase 8, maintainer-approved)
# tb harbor check    <task>
# tb run   --agent codex        <task>   # x3
# tb run   --agent claude-code  <task>   # x3
# tb cheat --agent codex        <task>
# tb cheat --agent claude-code  <task>
```

> The exact Harbor / `tb` invocations follow the harness in use; the commands
> above are placeholders for the Mac-side run and have **not** been executed
> here.

## Gate status (through Phase 6)

| Phase | Description | Status |
|------:|-------------|--------|
| 0 | Setup & scaffold | ✅ complete |
| 1 | Clean reference + realistic pipeline | ✅ complete |
| 2 | Fixtures + independent reference (frozen `expected_hidden`) | ✅ complete |
| 3 | Plant five fusion bugs + 32-state partial-fix audit | ✅ complete |
| 4 | Separate verifier (schema, Bank 1/2, determinism, range, CTRF) | ✅ complete |
| 5 | Oracle solution (ordered patches 001–005) | ✅ complete |
| 6 | Strip dev toggles + anti-cheat hardening; stripped oracle | ✅ complete |
| 7 | Docs, compliance checklist, packaging | ✅ this change |
| 8 | Final agent trials (`/run`, `/cheat`, Harbor) | ⬜ **not run** |

Local results recorded so far: oracle passes all verifier tests; nop/broken
fails; verifier is deterministic across `PYTHONHASHSEED=0/1`. **No final agent
trial results exist yet** — see `docs/failure-analysis.md`.

## Evaluation design notes

This task follows agent-evaluation best practices: deterministic verifiers,
outcome-based grading, CI-style regression checks, and contamination-resistant
hidden fixtures. The verifier does not use an LLM judge. It grades only the
final `results.json` emitted by the repaired retrieval pipeline.

The hidden checks combine:

- exact ranking comparison against frozen independent-reference outputs;
- RRF score recomputation from emitted BM25/dense sub-rankings;
- schema validation;
- fused-score range checks;
- two-seed determinism checks;
- narrow anti-cheat guards for fixture leakage, solution/test access, `.git`
  leakage, and network/public-solution routes.

Public release and public task discussion were deferred until after local
validation because the assignment uses internet-enabled agent trials.

## Compliance checklist

| Requirement | Status |
|-------------|--------|
| TB3 task metadata finalized (`task.toml`) | ✅ |
| `tests/test.sh` present, no runtime installs | ✅ |
| `tests/Dockerfile` present, deps pinned | ✅ |
| `environment/Dockerfile` copies app + data only (no `solution/`/`tests/`) | ✅ |
| Oracle solution scores 1.0 (local sim) | ✅ |
| Nop / broken state fails (local sim) | ✅ |
| Hidden fixtures only under `tests/fixtures/` | ✅ |
| No hidden/reference/expected/solution leakage into `environment/` | ✅ |
| No `BUG_*` toggles in agent-visible code | ✅ |
| No LLM judge; reward deterministic | ✅ |
| Verifier deterministic (`PYTHONHASHSEED=0/1`) | ✅ |
| Separate verifier mode (`environment_mode = "separate"`) | ✅ |
| `allow_internet = true` set in `task.toml` | ✅ |
| Verifier never writes into `/app` | ✅ |
| Docker environment + verifier image builds | ✅ both built (Mac-side) |
| Harbor parser validation + `TaskModel.is_valid_dir` | ✅ valid / true (Mac-side) |
| Harbor oracle run | ✅ reward 1.0, 0 exceptions (`jobs/2026-06-25__01-11-06`) |
| Harbor nop run | ✅ reward 0.0, 0 exceptions (`jobs/2026-06-25__01-11-24`) |
| Final `/run` and `/cheat` trials | ⬜ pending (Phase 8) |

## Security / integrity

The hidden reference outputs and verifier live only in the verifier image; the
agent environment never contains them. `expected_hidden.json` is generated from
an independent reference implementation under `tests/fixtures/`, frozen, and
loaded (never regenerated from `/app`) at grade time. Anti-cheat is treated as
an environment-security property: narrow deterministic guards flag fixture
leakage, verifier/solution access, `.git` leakage, and network/public-solution
routes, but correctness is graded primarily by artifact outcomes.

In addition to task success, this task treats reward-hacking resistance as a
separate integrity axis. `/cheat` trials are expected to receive zero reward,
and the verifier includes secondary guards against hidden fixture access,
expected-output fabrication, solution/test access, `.git` leakage, and
network/public-solution shortcuts.
