# AGENTS.md

This repo builds a **Terminal-Bench 3 (TB3) compatible task**:
`hybrid-retrieval-fusion`, a deterministic hybrid retrieval fusion debugging
task. BM25 and dense ranking are correct; the regression is in the fusion
layer.

## Rules for agents working in this repo

- **Follow the phase gates** defined in `CLAUDE.md` / `docs/build-plan.md`.
  Proceed to the next phase only when the current gate passes. If a gate fails,
  stop and report the blocker.
- **Do not copy hidden, test, or solution files into the agent environment.**
  `environment/` contains only agent-visible data and app code. `tests/` and
  `solution/` stay out of it.
- **Keep the verifier deterministic.** No LLM judge. Reward is computed from
  exact rankings and recomputed scores with fixed tolerances.
- **Do not plant bugs before a clean reference exists**, and do not collapse
  the service into a single script.
- **Final evaluation trials must be fresh runs** — not the same session that
  implemented the task — and require maintainer approval before running any
  paid/authenticated agent.
- **Authoring guidance is not for evaluation agents.** This file, `CLAUDE.md`,
  `docs/build-plan.md`, and `tools/` must never be exposed to final `/run` or
  `/cheat` agents.

See `CLAUDE.md` for the full normative contract, verifier design, and
anti-cheat rules.
