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
  5. TIE_BREAK        -> hybrid_search/ordering.py
  6. CANDIDATE_UNION_INDEXING -> hybrid_search/fusion.py
  7. SERIALIZATION_SCORE_RANK -> hybrid_search/serialization.py
  8. VECTOR_CORPUS_ALIGNMENT -> hybrid_search/rankers/dense.py
  9. QUERY_ORDER_IDENTITY -> hybrid_search/pipeline.py
 10. FUSED_SCORE_PRECISION -> hybrid_search/fusion.py

A state's bit ``i`` set means bug ``i`` is fixed. By default the reduced audit
runs:
  - all bugs present
  - all bugs fixed
  - each single bug fixed alone
  - all-but-one fixed

Pass ``--full`` to run all 1024 states. Each state is executed under
``PYTHONHASHSEED=0`` and ``=1`` and classified against the frozen independent
reference and the generated hidden banks similarly to the verifier: Bank 1
rankings, Bank 2 fused-score self-consistency, fused-score range, config/order
invariants, and determinism.
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
TESTS_DIR = TASK_DIR / "tests"

sys.path.insert(0, str(FIXTURE_DIR))
sys.path.insert(0, str(TESTS_DIR))
import reference_rrf  # noqa: E402
import generated_cases  # noqa: E402

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
    Bug("TIE_BREAK", "hybrid_search/ordering.py"),
    Bug("CANDIDATE_UNION_INDEXING", "hybrid_search/fusion.py"),
    Bug("SERIALIZATION_SCORE_RANK", "hybrid_search/serialization.py"),
    Bug("VECTOR_CORPUS_ALIGNMENT", "hybrid_search/rankers/dense.py"),
    Bug("QUERY_ORDER_IDENTITY", "hybrid_search/pipeline.py"),
    Bug("FUSED_SCORE_PRECISION", "hybrid_search/fusion.py"),
)


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def apply_fix(app_dir: Path, bug: str) -> None:
    """Apply one independent fix toggle to a temporary app copy."""
    fusion = app_dir / "hybrid_search" / "fusion.py"
    ordering = app_dir / "hybrid_search" / "ordering.py"
    pipeline = app_dir / "hybrid_search" / "pipeline.py"
    candidates = app_dir / "hybrid_search" / "candidate_utils.py"
    serialization = app_dir / "hybrid_search" / "serialization.py"
    dense = app_dir / "hybrid_search" / "rankers" / "dense.py"

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
        _replace(ordering, "hash(entry.doc_id)", "entry.doc_id")
    elif bug == "CANDIDATE_UNION_INDEXING":
        _replace(
            fusion,
            "for candidate_index, key in enumerate(keys, start=1):",
            "for key in keys:",
        )
        _replace(
            fusion,
            "score += _contribution(rrf_k, candidate_index, entry)",
            "score += _contribution(rrf_k, rank, entry)",
        )
    elif bug == "SERIALIZATION_SCORE_RANK":
        text = serialization.read_text(encoding="utf-8")
        old = '''\

def serialize_fused_ranking(
    ranking: Sequence[RankingEntry], config: SearchConfig
) -> List[Dict[str, object]]:
    """Serialize fused rows to the public schema."""
    return [
        serialize_entry(entry, rank, score=1.0 / (config.rrf_k + rank))
        for rank, entry in enumerate(ranking, start=1)
    ]


def serialize_result(result: QueryResult, config: SearchConfig) -> Dict[str, object]:
'''
        new = '''\

def serialize_result(result: QueryResult) -> Dict[str, object]:
'''
        if old not in text:
            raise RuntimeError("expected serialize_fused_ranking block not found")
        text = text.replace(old, new)
        text = text.replace(
            '"fused": serialize_fused_ranking(result.fused, config),',
            '"fused": serialize_ranking(result.fused),',
        )
        text = text.replace(
            "[serialize_result(result, config) for result in results]",
            "[serialize_result(result) for result in results]",
        )
        serialization.write_text(text, encoding="utf-8")
    elif bug == "VECTOR_CORPUS_ALIGNMENT":
        _replace(
            dense,
            "for index, doc in enumerate(self.documents[1:] + self.documents[:1]):",
            "for index, doc in enumerate(self.documents):",
        )
    elif bug == "QUERY_ORDER_IDENTITY":
        old = '''\
        cached: dict[str, QueryResult] = {}
        results: List[QueryResult] = []
        for query in sorted(queries, key=lambda item: item.query_id):
            result = cached.get(query.query_id)
            if result is None:
                result = self.search(query)
                cached[query.query_id] = result
            results.append(result)
        return results
'''
        new = "        return [self.search(query) for query in queries]\n"
        _replace(pipeline, old, new)
    elif bug == "FUSED_SCORE_PRECISION":
        _replace(
            fusion,
            "        score = int(score * 1_000_000) / 1_000_000\n",
            "",
        )
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


def ids(ranking: list[dict]) -> list[str]:
    return [entry["doc_id"] for entry in ranking]


def emitted_pairs(row: dict, mod: str) -> list[tuple[str, float]]:
    return [(entry["doc_id"], entry["score"]) for entry in row[mod]]


def classify_static(
    expected: dict,
    hidden_queries: list[dict],
    runs: dict[str, dict],
) -> tuple[int, dict]:
    cfg = expected["config"]
    top_k = cfg["top_k"]
    rrf_k = cfg["rrf_k"]
    max_fused = 2.0 / (rrf_k + 1)

    reasons: dict[str, list[str]] = {}
    pass_count = 0
    expected_rows = [
        (query["query_id"], expected["queries"][query["query_id"]])
        for query in hidden_queries
    ]

    for index, (qid, exp) in enumerate(expected_rows):
        q_reasons: set[str] = set()
        seed_rows = []

        for seed in SEEDS:
            rows = runs[seed].get("results", [])
            if index >= len(rows):
                q_reasons.add("missing_query")
                continue
            row = rows[index]
            seed_rows.append(row)
            if row.get("query_id") != qid:
                q_reasons.add("query_order")

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


def run_generated_banks(app_dir: Path, tmp: Path) -> dict[str, tuple[generated_cases.GeneratedCase, dict[str, dict]]]:
    generated: dict[str, tuple[generated_cases.GeneratedCase, dict[str, dict]]] = {}
    for case in generated_cases.build_cases():
        case_runs = {
            seed: generated_cases.run_app_case(
                app_dir,
                case,
                tmp / "generated" / case.name / f"seed{seed}",
                seed=seed,
            )
            for seed in SEEDS
        }
        generated[case.name] = (case, case_runs)
    return generated


def generated_reason_codes(failures: list[str]) -> list[str]:
    codes: set[str] = set()
    for failure in failures:
        if ": config " in failure:
            codes.add("generated_config")
        if "query_id" in failure:
            codes.add("generated_query_order")
        if "bm25 ids" in failure:
            codes.add("generated_bm25")
        if "bm25 raw scores" in failure:
            codes.add("generated_bm25_score")
        if "dense ids" in failure:
            codes.add("generated_dense")
        if "dense raw scores" in failure:
            codes.add("generated_dense_score")
        if "fused ids" in failure:
            codes.add("generated_fused")
        if "fused scores" in failure:
            codes.add("generated_score")
        if "Bank2" in failure:
            codes.add("generated_bank2")
        if "differs across seeds" in failure:
            codes.add("generated_nondeterministic")
        if "out of range" in failure:
            codes.add("generated_range")
    return sorted(codes or {"generated_failure"})


def classify_generated(
    generated_runs: dict[str, tuple[generated_cases.GeneratedCase, dict[str, dict]]],
) -> tuple[int, dict]:
    reasons: dict[str, list[str]] = {}
    pass_count = 0
    for name, (case, runs) in generated_runs.items():
        failures = generated_cases.validate_runs(case, runs)
        if failures:
            reasons[f"generated:{name}"] = generated_reason_codes(failures)
        else:
            pass_count += 1
    return pass_count, reasons


def classify(
    expected: dict,
    hidden_queries: list[dict],
    static_runs: dict[str, dict],
    generated_runs: dict[str, tuple[generated_cases.GeneratedCase, dict[str, dict]]],
) -> tuple[int, dict]:
    static_pass, static_reasons = classify_static(expected, hidden_queries, static_runs)
    generated_pass, generated_reasons = classify_generated(generated_runs)
    reasons = dict(static_reasons)
    reasons.update(generated_reasons)
    return static_pass + generated_pass, reasons


def run_state(fixed_bits: tuple[int, ...], expected: dict) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory(prefix="hrf-audit-") as raw_tmp:
        tmp = Path(raw_tmp)
        app_dir = prepare_app(tmp, fixed_bits)
        static_runs = {seed: run_seed(app_dir, seed, expected, tmp) for seed in SEEDS}
        generated_runs = run_generated_banks(app_dir, tmp)
        hidden_queries = json.loads((FIXTURE_DIR / "hidden_queries.json").read_text(encoding="utf-8"))
        return classify(expected, hidden_queries, static_runs, generated_runs)


def bit_key(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits)


def bug_names(bits: Iterable[int], fixed: bool) -> list[str]:
    return [bug.name for bit, bug in zip(bits, BUGS) if bool(bit) is fixed]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit partial fixes for hybrid-retrieval-fusion.")
    parser.add_argument("--full", action="store_true", help="run all fix states")
    args = parser.parse_args()

    expected = json.loads((FIXTURE_DIR / "expected_hidden.json").read_text(encoding="utf-8"))
    hidden_queries = json.loads((FIXTURE_DIR / "hidden_queries.json").read_text(encoding="utf-8"))
    generated_case_names = [case.name for case in generated_cases.build_cases()]
    n_static = len(hidden_queries)
    n_generated = len(generated_case_names)
    n_checks = n_static + n_generated
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
            "fail_count": n_checks - results[single][0],
            "fail_reasons": results[single][1],
        }
        all_but_one_fixed[bug.name] = {
            "state": bit_key(almost),
            "fail_count": n_checks - results[almost][0],
            "fail_reasons": results[almost][1],
        }

    only_all_fixed = None
    if args.full:
        only_all_fixed = all(
            (bits == all_fixed) == (pass_count == n_checks)
            for bits, (pass_count, _) in results.items()
        )

    report = {
        "mode": f"full-{2 ** len(BUGS)}-state" if args.full else "reduced",
        "n_hidden_checks": n_checks,
        "n_static_hidden_queries": n_static,
        "generated_hidden_banks": generated_case_names,
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
                "fail_count": n_checks - pass_count,
                "fail_reasons": reasons,
            }
            for bits, (pass_count, reasons) in sorted(results.items())
        },
    }
    (FIXTURE_DIR / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")

    gate = (
        all_present_pass < n_checks
        and all_fixed_pass == n_checks
        and all(item["fail_count"] >= 1 for item in single_fixed.values())
        and all(item["fail_count"] >= 2 for item in all_but_one_fixed.values())
        and (only_all_fixed is not False)
    )

    print("=== partial-fix audit ===")
    print(
        f"mode: {report['mode']}   hidden checks: {n_checks} "
        f"({n_static} static queries + {n_generated} generated banks)"
    )
    print(f"bug order: {report['bug_order']}")
    print(f"all-present ({bit_key(all_present)}) pass: {all_present_pass}/{n_checks}")
    print(f"all-fixed   ({bit_key(all_fixed)}) pass: {all_fixed_pass}/{n_checks}")
    if only_all_fixed is not None:
        print(f"only all-fixed passes all queries: {only_all_fixed}")
    print("\nsingle bug fixed alone -> failures:")
    for bug in BUGS:
        item = single_fixed[bug.name]
        kinds = sorted({kind for r in item["fail_reasons"].values() for kind in r})
        print(f"  {bug.name:<28} state={item['state']} fails {item['fail_count']}/{n_checks} reasons={kinds}")
    print("\nall-but-one fixed (listed bug ACTIVE) -> failures:")
    for bug in BUGS:
        item = all_but_one_fixed[bug.name]
        kinds = sorted({kind for r in item["fail_reasons"].values() for kind in r})
        print(f"  {bug.name:<28} state={item['state']} fails {item['fail_count']}/{n_checks} reasons={kinds}")
    print("\nPARTIAL-FIX AUDIT GATE:", "PASS" if gate else "FAIL")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
