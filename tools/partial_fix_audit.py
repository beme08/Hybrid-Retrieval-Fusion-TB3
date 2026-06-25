#!/usr/bin/env python3
"""Partial-fix audit for the planted retrieval/fusion bugs.

Repo-root tooling only: this is not shipped to evaluation agents and is not a
verifier dependency. The shipped app contains natural buggy code paths, not
``BUG_*`` toggles, so this audit creates a temporary copy of ``environment/app``
for each state and applies independent source-level fix toggles there.

Bug order and locations:
  1. RAW_MIXING       -> hybrid_search/fusion.py
  2. RANK_BASE        -> hybrid_search/fusion.py
  3. CANDIDATE_DEPTH  -> hybrid_search/pipeline.py
  4. DOC_ID_COLLISION -> hybrid_search/candidate_utils.py
  5. TIE_BREAK        -> hybrid_search/fusion.py

A state's bit ``i`` set means bug ``i`` is fixed. By default the reduced audit
runs:
  - all bugs present
  - all bugs fixed
  - each single bug fixed alone
  - all-but-one fixed

Pass ``--full`` to run all 32 states. Each state is executed under
``PYTHONHASHSEED=0`` and ``=1`` and classified against the frozen independent
reference similarly to the verifier: Bank 1 rankings, Bank 2 fused-score
self-consistency, fused-score range, and determinism.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "hybrid-retrieval-fusion"
DATA_DIR = TASK_DIR / "environment" / "data"
APP_SRC = TASK_DIR / "environment" / "app"
FIXTURE_DIR = TASK_DIR / "tests" / "fixtures"

sys.path.insert(0, str(FIXTURE_DIR))
import reference_rrf  # noqa: E402

SEEDS = ("0", "1")
SCORE_TOL = 1e-12


@dataclass(frozen=True)
class Bug:
    name: str
    location: str


BUGS = (
    Bug("RAW_MIXING", "hybrid_search/fusion.py"),
    Bug("RANK_BASE", "hybrid_search/fusion.py"),
    Bug("CANDIDATE_DEPTH", "hybrid_search/pipeline.py"),
    Bug("DOC_ID_COLLISION", "hybrid_search/candidate_utils.py"),
    Bug("TIE_BREAK", "hybrid_search/fusion.py"),
)


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def apply_fix(app_dir: Path, bug: str) -> None:
    """Apply one independent fix toggle to a temporary app copy."""
    fusion = app_dir / "hybrid_search" / "fusion.py"
    pipeline = app_dir / "hybrid_search" / "pipeline.py"
    candidates = app_dir / "hybrid_search" / "candidate_utils.py"

    if bug == "RAW_MIXING":
        text = fusion.read_text(encoding="utf-8")
        text = text.replace(
            "return entry.raw_score + 1.0 / (rrf_k + (rank - 1))",
            "return 1.0 / (rrf_k + (rank - 1))",
        )
        text = text.replace(
            "return entry.raw_score + 1.0 / (rrf_k + rank)",
            "return 1.0 / (rrf_k + rank)",
        )
        fusion.write_text(text, encoding="utf-8")
        if "return entry.raw_score" in text:
            raise RuntimeError("RAW_MIXING fix did not remove raw-score contribution")
    elif bug == "RANK_BASE":
        _replace(fusion, "rrf_k + (rank - 1)", "rrf_k + rank")
    elif bug == "CANDIDATE_DEPTH":
        _replace(pipeline, "depth = self.config.top_k", "depth = self.config.candidate_depth")
    elif bug == "DOC_ID_COLLISION":
        _replace(candidates, 'return doc_id.rsplit("_", 1)[-1]', "return doc_id")
    elif bug == "TIE_BREAK":
        _replace(fusion, "hash(e.doc_id)", "e.doc_id")
    else:
        raise ValueError(f"unknown bug: {bug}")


def states_for_mode(full: bool) -> list[tuple[int, ...]]:
    if full:
        return list(itertools.product((0, 1), repeat=len(BUGS)))

    states: set[tuple[int, ...]] = set()
    zero = (0,) * len(BUGS)
    one = (1,) * len(BUGS)
    states.add(zero)
    states.add(one)
    for i in range(len(BUGS)):
        states.add(tuple(1 if j == i else 0 for j in range(len(BUGS))))
        states.add(tuple(0 if j == i else 1 for j in range(len(BUGS))))
    return sorted(states)


def prepare_app(tmp: Path, fixed_bits: tuple[int, ...]) -> Path:
    app_dir = tmp / "app"
    shutil.copytree(APP_SRC, app_dir)
    shutil.copytree(DATA_DIR, app_dir / "data")
    for bit, bug in zip(fixed_bits, BUGS):
        if bit:
            apply_fix(app_dir, bug.name)
    return app_dir


def run_seed(app_dir: Path, seed: str, expected: dict, tmp: Path) -> dict:
    cfg = expected["config"]
    out = tmp / f"results_seed{seed}.json"
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable,
        str(app_dir / "scripts" / "run_search.py"),
        "--corpus",
        str(app_dir / "data" / "corpus.json"),
        "--vectors",
        str(app_dir / "data" / "vectors.npy"),
        "--queries",
        str(FIXTURE_DIR / "hidden_queries.json"),
        "--top-k",
        str(cfg["top_k"]),
        "--candidate-depth",
        str(cfg["candidate_depth"]),
        "--rrf-k",
        str(cfg["rrf_k"]),
        "--output",
        str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run failed for seed {seed} rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out.read_text(encoding="utf-8"))


def index_results(result: dict) -> dict:
    return {row["query_id"]: row for row in result["results"]}


def ids(ranking: list[dict]) -> list[str]:
    return [entry["doc_id"] for entry in ranking]


def emitted_pairs(row: dict, mod: str) -> list[tuple[str, float]]:
    return [(entry["doc_id"], entry["score"]) for entry in row[mod]]


def classify(expected: dict, runs: dict[str, dict]) -> tuple[int, dict]:
    cfg = expected["config"]
    top_k = cfg["top_k"]
    rrf_k = cfg["rrf_k"]
    max_fused = 2.0 / (rrf_k + 1)

    indexed = {seed: index_results(result) for seed, result in runs.items()}
    reasons: dict[str, list[str]] = {}
    pass_count = 0

    for qid, exp in expected["queries"].items():
        q_reasons: set[str] = set()
        seed_rows = []

        for seed in SEEDS:
            row = indexed[seed].get(qid)
            if row is None:
                q_reasons.add("missing_query")
                continue
            seed_rows.append(row)

            for mod in ("bm25", "dense", "fused"):
                if ids(row[mod]) != exp[mod]:
                    q_reasons.add(mod)

            emitted_fused = emitted_pairs(row, "fused")
            emitted_scores = [score for _, score in emitted_fused]
            if len(emitted_scores) != len(exp["fused_scores"]) or any(
                abs(got - want) > SCORE_TOL
                for got, want in zip(emitted_scores, exp["fused_scores"])
            ):
                q_reasons.add("score")
            if any(score < 0.0 or score > max_fused + 1e-9 for score in emitted_scores):
                q_reasons.add("range")

            bm = [(entry["doc_id"], 0.0) for entry in row["bm25"]]
            dn = [(entry["doc_id"], 0.0) for entry in row["dense"]]
            recomputed = reference_rrf.fuse([bm, dn], k=rrf_k, top_k=top_k)
            if [doc_id for doc_id, _ in recomputed] != ids(row["fused"]):
                q_reasons.add("bank2_order")
            for (rdoc, rscore), (edoc, escore) in zip(recomputed, emitted_fused):
                if rdoc != edoc or abs(rscore - escore) > SCORE_TOL:
                    q_reasons.add("bank2_score")
                    break

        if len(seed_rows) == len(SEEDS):
            for mod in ("bm25", "dense", "fused"):
                first = emitted_pairs(seed_rows[0], mod)
                if any(emitted_pairs(row, mod) != first for row in seed_rows[1:]):
                    q_reasons.add("nondeterministic")

        if q_reasons:
            reasons[qid] = sorted(q_reasons)
        else:
            pass_count += 1

    return pass_count, reasons


def run_state(fixed_bits: tuple[int, ...], expected: dict) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory(prefix="hrf-audit-") as raw_tmp:
        tmp = Path(raw_tmp)
        app_dir = prepare_app(tmp, fixed_bits)
        runs = {seed: run_seed(app_dir, seed, expected, tmp) for seed in SEEDS}
        return classify(expected, runs)


def bit_key(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits)


def bug_names(bits: Iterable[int], fixed: bool) -> list[str]:
    return [bug.name for bit, bug in zip(bits, BUGS) if bool(bit) is fixed]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit partial fixes for hybrid-retrieval-fusion.")
    parser.add_argument("--full", action="store_true", help="run all 32 fix states")
    args = parser.parse_args()

    expected = json.loads((FIXTURE_DIR / "expected_hidden.json").read_text(encoding="utf-8"))
    n_queries = len(expected["queries"])
    states = states_for_mode(args.full)

    results = {}
    for bits in states:
        pass_count, reasons = run_state(bits, expected)
        results[bits] = (pass_count, reasons)

    all_present = (0,) * len(BUGS)
    all_fixed = (1,) * len(BUGS)
    all_present_pass = results[all_present][0]
    all_fixed_pass = results[all_fixed][0]

    single_fixed = {}
    all_but_one_fixed = {}
    for i, bug in enumerate(BUGS):
        single = tuple(1 if j == i else 0 for j in range(len(BUGS)))
        almost = tuple(0 if j == i else 1 for j in range(len(BUGS)))
        single_fixed[bug.name] = {
            "state": bit_key(single),
            "fail_count": n_queries - results[single][0],
            "fail_reasons": results[single][1],
        }
        all_but_one_fixed[bug.name] = {
            "state": bit_key(almost),
            "fail_count": n_queries - results[almost][0],
            "fail_reasons": results[almost][1],
        }

    only_all_fixed = None
    if args.full:
        only_all_fixed = all(
            (bits == all_fixed) == (pass_count == n_queries)
            for bits, (pass_count, _) in results.items()
        )

    report = {
        "mode": "full-32-state" if args.full else "reduced",
        "n_hidden_queries": n_queries,
        "bug_order": [bug.name for bug in BUGS],
        "bug_locations": {bug.name: bug.location for bug in BUGS},
        "all_present_pass_count": all_present_pass,
        "all_fixed_pass_count": all_fixed_pass,
        "only_all_fixed_passes": only_all_fixed,
        "single_bug_fixed_alone": single_fixed,
        "all_but_one_fixed": all_but_one_fixed,
        "per_state": {
            bit_key(bits): {
                "fixed": bug_names(bits, True),
                "active": bug_names(bits, False),
                "pass_count": pass_count,
                "fail_count": n_queries - pass_count,
                "fail_reasons": reasons,
            }
            for bits, (pass_count, reasons) in sorted(results.items())
        },
    }
    (FIXTURE_DIR / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")

    gate = (
        all_present_pass < n_queries
        and all_fixed_pass == n_queries
        and all(item["fail_count"] >= 1 for item in single_fixed.values())
        and all(item["fail_count"] >= 2 for item in all_but_one_fixed.values())
        and (only_all_fixed is not False)
    )

    print("=== partial-fix audit ===")
    print(f"mode: {report['mode']}   hidden queries: {n_queries}")
    print(f"bug order: {report['bug_order']}")
    print(f"all-present (00000) pass: {all_present_pass}/{n_queries}")
    print(f"all-fixed   (11111) pass: {all_fixed_pass}/{n_queries}")
    if only_all_fixed is not None:
        print(f"only all-fixed passes all queries: {only_all_fixed}")
    print("\nsingle bug fixed alone -> failures:")
    for bug in BUGS:
        item = single_fixed[bug.name]
        kinds = sorted({kind for r in item["fail_reasons"].values() for kind in r})
        print(f"  {bug.name:<16} state={item['state']} fails {item['fail_count']}/{n_queries} reasons={kinds}")
    print("\nall-but-one fixed (listed bug ACTIVE) -> failures:")
    for bug in BUGS:
        item = all_but_one_fixed[bug.name]
        kinds = sorted({kind for r in item["fail_reasons"].values() for kind in r})
        print(f"  {bug.name:<16} state={item['state']} fails {item['fail_count']}/{n_queries} reasons={kinds}")
    print("\nPARTIAL-FIX AUDIT GATE:", "PASS" if gate else "FAIL")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
