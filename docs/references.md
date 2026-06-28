# References and Design Rationale

This document collects papers, benchmark docs, and articles that informed the
task design, validation strategy, and failure analysis. These references are
not required at runtime and are not part of the benchmark environment.

## Terminal-Bench / Harbor / Evaluation Infrastructure

- Terminal-Bench 3 repository
  https://github.com/harbor-framework/terminal-bench-3

- Terminal-Bench 3 contributing guide
  https://raw.githubusercontent.com/harbor-framework/terminal-bench-3/main/CONTRIBUTING.md

- Terminal-Bench 3 reviewing guide
  https://raw.githubusercontent.com/harbor-framework/terminal-bench-3/main/REVIEWING.md

- Terminal-Bench 3 task implementation rubric
  https://raw.githubusercontent.com/harbor-framework/terminal-bench-3/main/rubrics/task-implementation.toml

- Terminal-Bench 3 task proposal rubric
  https://raw.githubusercontent.com/harbor-framework/terminal-bench-3/main/rubrics/task-proposal.md

- Terminal-Bench 3 trial analysis rubric
  https://raw.githubusercontent.com/harbor-framework/terminal-bench-3/main/rubrics/trial-analysis.toml

- Terminal-Bench 3 trial analysis job prompt
  https://raw.githubusercontent.com/harbor-framework/terminal-bench-3/main/rubrics/trial-analysis-job.txt

- Harbor documentation
  https://www.harborframework.com/docs

- Harbor run evaluations documentation
  https://www.harborframework.com/docs/run-jobs/run-evals

## Agent Reward Hacking / Benchmark Security

- Cheating Agents / DebugML
  https://debugml.github.io/cheating-agents/

- BenchFlow AI — Awesome Evals
  https://github.com/benchflow-ai/awesome-evals

These references motivated:
- separate verifier mode
- keeping hidden fixtures out of `/app`
- anti-cheat checks for `/tests`, `/solution`, hidden expected outputs, reward
  files, and public solution fetches
- distinguishing valid model failures from infra/provider failures

## Agent Failure Modes and Task Difficulty

- LLMs Have Made Failure Worth Publishing
  https://arxiv.org/html/2604.06236v1

- arXiv:2602.10046
  https://arxiv.org/pdf/2602.10046

- arXiv:2503.16416
  https://arxiv.org/abs/2503.16416

- arXiv:2504.00255
  https://arxiv.org/pdf/2504.00255

- arXiv:2404.12272
  https://arxiv.org/abs/2404.12272

- Hugging Face paper page: 2602.12670
  https://huggingface.co/papers/2602.12670

These references informed the design emphasis on:
- compositional debugging difficulty
- multi-hop localization between cause and symptom
- anchoring on misleading local diagnostics
- confirmation bias around familiar formulas or conventions
- documenting negative results instead of hiding solved candidates

## Retrieval Context

- Overview of the TREC 2024 Retrieval-Augmented Generation Track
  https://arxiv.org/abs/2502.11072

- Reciprocal Rank Fusion (Cormack, Clarke, Buettcher, SIGIR 2009)
  https://doi.org/10.1145/1571941.1572114

These references motivated:
- the BM25 + dense + RRF hybrid architecture
- the use of candidate-depth semantics
- fusion-before-truncation ordering
- opaque document identity handling

## Task-Specific Lessons Learned

### Hybrid Retrieval Fusion

The retrieval-fusion candidate showed that a fair, explicit contract can still be
solved by a strong model when the repair is locally composable. The main lesson
was that hidden tests and oracle/nop validation are not sufficient evidence of
difficulty if a frontier agent can implement the full public contract directly.

The task was retired as a final candidate because Codex solved v7 with reward
1.0. The non-standard weighted-per-modality RRF contract and ten interacting bugs
were not enough to prevent a careful agent from repairing the service.

## Additional References To Verify

The design discussion also considered literature on:
- compositional reasoning failures in LLMs
- irrelevant context distraction
- anchoring and confirmation bias in security analysis
- order sensitivity in tool use and threat-intelligence workflows
- ARC-AGI-style "local success, wrong world model" failures

These were used as conceptual framing during task design. Exact bibliographic
entries should be verified before being cited formally.
