#!/usr/bin/env python3
"""32-state partial-fix audit for the planted fusion bugs (Phase 3).

Repo-root tooling (not shipped, not a verifier dependency). Exercises the five
development ``BUG_*`` toggles across all 2**5 = 32 fix/active combinations and,
for each, checks the shipped ``/app`` fusion output against the frozen
``expected_hidden.json`` produced by the independent reference.

A state's bit ``i`` set => bug ``i`` is FIXED (toggle off); bit clear => bug is
ACTIVE. The all-fixed state (31) must pass every hidden query; no other state
may. Each "4-of-5 fixed" state (exactly one bug active) must fail >= 2 hidden
queries, proving every bug is load-bearing.

Each state is run under PYTHONHASHSEED=0 and =1. A query passes only if, under
BOTH seeds, the fused doc-id order matches expected AND the fused scores match
expected within tolerance. This mirrors the eventual verifier: Bank-1 (doc-id
ranking), Bank-2/range (fused scores) and the determinism guard (two seeds).
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "hybrid-retrieval-fusion"
DATA_DIR = TASK_DIR / "environment" / "data"
APP_DIR = TASK_DIR / "environment" / "app"
FIXTURE_DIR = TASK_DIR / "tests" / "fixtures"

BUGS = ["RAW_MIXING", "RANK_BASE", "TRUNCATE", "SUFFIX_COLLISION", "TIE_BREAK"]
SEEDS = ["0", "1"]
SCORE_TOL = 1e-9


# ---------------------------------------------------------------- emit mode ---
def emit() -> int:
    """Print fused (doc_id, score) per hidden query for the current BUG_* env."""
    import numpy as np
    sys.path.insert(0, str(APP_DIR))
    from hybrid_search.config import SearchConfig
    from hybrid_search.models import Document, Query
    from hybrid_search.pipeline import RetrievalPipeline

    expected = json.loads((FIXTURE_DIR / "expected_hidden.json").read_text())
    cfg = expected["config"]
    corpus = json.loads((DATA_DIR / "corpus.json").read_text())
    vectors = np.load(DATA_DIR / "vectors.npy")
    hidden = json.loads((FIXTURE_DIR / "hidden_queries.json").read_text())

    sc = SearchConfig(top_k=cfg["top_k"], candidate_depth=cfg["candidate_depth"],
                      rrf_k=cfg["rrf_k"], bm25_k1=cfg["bm25_k1"], bm25_b=cfg["bm25_b"])
    pipe = RetrievalPipeline([Document(d["doc_id"], d["text"]) for d in corpus], vectors, sc)
    out = {}
    for q in hidden:
        res = pipe.search(Query(q["query_id"], q["text"], q["embedding"]))
        out[q["query_id"]] = [[e.doc_id, e.score] for e in res.fused]
    json.dump(out, sys.stdout)
    return 0


# ------------------------------------------------------------- driver mode ---
def run_state(fixed_bits) -> dict:
    """Run a state under both seeds; return {qid: passes_bool} plus raw outputs."""
    env = dict(os.environ)
    for i, bug in enumerate(BUGS):
        env[f"BUG_{bug}"] = "0" if fixed_bits[i] else "1"
    runs = {}
    for seed in SEEDS:
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--emit"],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"emit failed: {proc.stderr}")
        runs[seed] = json.loads(proc.stdout)
    return runs


def classify(expected, runs):
    """Return (n_pass, per_query_reason) for one state's two-seed runs."""
    reasons = {}
    n_pass = 0
    for qid, exp in expected["queries"].items():
        exp_ids = exp["fused"]
        exp_scores = exp["fused_scores"]
        ok = True
        reason = set()
        seed_outputs = []
        for seed in SEEDS:
            got = runs[seed][qid]
            ids = [d for d, _ in got]
            scores = [s for _, s in got]
            seed_outputs.append(tuple(ids))
            if ids != exp_ids:
                ok = False
                reason.add("order")
            if len(scores) != len(exp_scores) or any(
                abs(a - b) > SCORE_TOL for a, b in zip(scores, exp_scores)
            ):
                ok = False
                reason.add("score")
            if any(s > (2.0 / 61.0) + 1e-9 for s in scores):
                reason.add("range")
        if seed_outputs[0] != seed_outputs[1]:
            ok = False
            reason.add("nondeterministic")
        if ok:
            n_pass += 1
        else:
            reasons[qid] = sorted(reason)
    return n_pass, reasons


def main() -> int:
    if "--emit" in sys.argv:
        return emit()

    expected = json.loads((FIXTURE_DIR / "expected_hidden.json").read_text())
    n_queries = len(expected["queries"])

    results = {}  # bits tuple -> (n_pass, reasons)
    for bits in itertools.product([0, 1], repeat=5):
        results[bits] = classify(expected, run_state(bits))

    all_fixed = (1, 1, 1, 1, 1)
    all_fixed_pass = results[all_fixed][0]

    # Gate evaluations
    only_all_fixed = all(
        (bits == all_fixed) == (n == n_queries) for bits, (n, _) in results.items()
    )
    four_of_five = {}
    for i, bug in enumerate(BUGS):
        bits = tuple(0 if j == i else 1 for j in range(5))  # only bug i active
        n_pass, reasons = results[bits]
        four_of_five[bug] = (n_queries - n_pass, reasons)

    # Difficulty gradient by number of fixed bugs
    gradient = {}
    for bits, (n, _) in results.items():
        k = sum(bits)
        gradient.setdefault(k, []).append(n)
    gradient_summary = {
        k: {"states": len(v), "min_pass": min(v), "max_pass": max(v),
            "avg_pass": round(sum(v) / len(v), 2)}
        for k, v in sorted(gradient.items())
    }

    report = {
        "n_hidden_queries": n_queries,
        "bug_order": BUGS,
        "all_fixed_pass_count": all_fixed_pass,
        "only_all_fixed_passes": only_all_fixed,
        "four_of_five_fixed": {
            bug: {"fail_count": fc, "fail_reasons": rs}
            for bug, (fc, rs) in four_of_five.items()
        },
        "difficulty_gradient_by_fixed_count": gradient_summary,
        "per_state_pass": {
            "".join(str(b) for b in bits): n for bits, (n, _) in sorted(results.items())
        },
    }
    (FIXTURE_DIR / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # ---- console summary ----
    print("=== 32-state partial-fix audit ===")
    print(f"hidden queries: {n_queries}   bug order: {BUGS}")
    print(f"all-fixed (11111) pass: {all_fixed_pass}/{n_queries}")
    print(f"only all-fixed passes all queries: {only_all_fixed}")
    print("\n4-of-5 fixed (exactly one bug ACTIVE) -> failures:")
    for bug, (fc, rs) in four_of_five.items():
        kinds = sorted({k for r in rs.values() for k in r})
        print(f"  {bug:<18} fails {fc}/{n_queries}  reasons={kinds}")
    print("\ndifficulty gradient (passes by #bugs fixed):")
    for k, s in gradient_summary.items():
        print(f"  fixed={k}: states={s['states']:>2}  pass min/avg/max = "
              f"{s['min_pass']}/{s['avg_pass']}/{s['max_pass']}")

    gate = (
        only_all_fixed
        and all_fixed_pass == n_queries
        and all(fc >= 2 for fc, _ in four_of_five.values())
    )
    # each bug load-bearing == its 4-of-5 state fails (fc>=1), already implied by >=2
    print("\nPHASE 3 AUDIT GATE:", "PASS" if gate else "FAIL")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
