#!/usr/bin/env python3
"""Independent validation of the generated fixtures (Phase 2 gate).

Repo-root tooling (not shipped, not a verifier dependency). Re-derives every
gate property from the committed fixtures and the *independent* reference, and
additionally confirms the clean shipped ``/app`` agrees with the reference (so
no bugs are planted). Exits non-zero on any failure.

Checks
------
1. expected_hidden.json equals a fresh recompute from tests/fixtures reference.
2. Per-query structure: non-empty bm25 & dense, fused length >= top_k,
   fused scores not all tied, real cross-modality overlap.
3. Coverage matrix: each planned bug meets its activation target; >=2 hidden
   queries activate >=3 structural bug classes.
4. Visible non-leak: no visible query exposes raw_mixing / truncate /
   suffix_collision / rank-base ORDER flips; a clean dissimilar tie exists.
5. Clean /app == reference on hidden and visible queries (Bank-1 doc-ids).
6. Emits a deterministic digest for cross-PYTHONHASHSEED comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "hybrid-retrieval-fusion"
DATA_DIR = TASK_DIR / "environment" / "data"
FIXTURE_DIR = TASK_DIR / "tests" / "fixtures"
APP_DIR = TASK_DIR / "environment" / "app"

sys.path.insert(0, str(FIXTURE_DIR))
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from reference_bm25 import ReferenceBM25  # noqa: E402
from reference_dense import ReferenceDense  # noqa: E402
import reference_rrf  # noqa: E402
import generate_fixtures as gf  # noqa: E402

from hybrid_search.config import SearchConfig  # noqa: E402
from hybrid_search.models import Document, Query  # noqa: E402
from hybrid_search.pipeline import RetrievalPipeline  # noqa: E402

FAILURES = []
OVERLAP_MIN = 3          # min docs shared across modalities per hidden query
OVERLAP_QUERIES_MIN = 8  # min hidden queries meeting OVERLAP_MIN


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
    return cond


def load():
    corpus = json.loads((DATA_DIR / "corpus.json").read_text())
    vectors = np.load(DATA_DIR / "vectors.npy")
    visible = json.loads((DATA_DIR / "visible_queries.json").read_text())
    hidden = json.loads((FIXTURE_DIR / "hidden_queries.json").read_text())
    expected = json.loads((FIXTURE_DIR / "expected_hidden.json").read_text())
    return corpus, vectors, visible, hidden, expected


def main() -> int:
    corpus, vectors, visible, hidden, expected = load()
    docs = [(d["doc_id"], d["text"]) for d in corpus]
    doc_ids = [d for d, _ in docs]
    cfg = expected["config"]
    top_k, depth, rrf_k = cfg["top_k"], cfg["candidate_depth"], cfg["rrf_k"]

    check(vectors.shape[0] == len(docs), "vectors/corpus row mismatch")
    check(len(hidden) == 12, f"expected 12 hidden queries, got {len(hidden)}")
    check(len(visible) == 4, f"expected 4 visible queries, got {len(visible)}")

    bm25 = ReferenceBM25(docs, k1=cfg["bm25_k1"], b=cfg["bm25_b"])
    dense = ReferenceDense(doc_ids, vectors)

    # ---- 1 & 2: expected matches reference; structural checks ----
    overlap_ok = 0
    digest_parts = []
    for q in hidden:
        bm = bm25.rank(q["text"], depth)
        dn = dense.rank(q["embedding"], depth)
        fused = reference_rrf.fuse([bm, dn], k=rrf_k, top_k=top_k)
        exp = expected["queries"][q["query_id"]]
        check([d for d, _ in bm] == exp["bm25"], f"{q['query_id']}: bm25 != expected")
        check([d for d, _ in dn] == exp["dense"], f"{q['query_id']}: dense != expected")
        check([d for d, _ in fused] == exp["fused"], f"{q['query_id']}: fused != expected")
        check(len(bm) > 0, f"{q['query_id']}: empty bm25")
        check(len(dn) > 0, f"{q['query_id']}: empty dense")
        check(len(fused) >= top_k, f"{q['query_id']}: fused < top_k")
        scores = [s for _, s in fused]
        check(len(set(round(s, 12) for s in scores)) > 1, f"{q['query_id']}: all fused scores tied")
        overlap = len(set(d for d, _ in bm) & set(d for d, _ in dn))
        if overlap >= OVERLAP_MIN:
            overlap_ok += 1
        digest_parts.append(q["query_id"] + "|" + ",".join(exp["fused"]))
    check(overlap_ok >= OVERLAP_QUERIES_MIN,
          f"cross-modality overlap weak: {overlap_ok} < {OVERLAP_QUERIES_MIN}")

    # ---- 3: coverage matrix (recomputed independently via gf simulations) ----
    counts = {k: 0 for k in gf.HIDDEN_TARGETS}
    multi = 0
    for q in hidden:
        a = gf.analyze(q["text"], q["embedding"], bm25, dense)
        for k in counts:
            counts[k] += int(a["flags"][k])
        if a["structural_bugs"] >= 3:
            multi += 1
    for bug, target in gf.HIDDEN_TARGETS.items():
        check(counts[bug] >= target, f"coverage {bug}: {counts[bug]} < {target}")
    check(multi >= gf.MULTI_BUG_MIN, f"multi-structural-bug queries {multi} < {gf.MULTI_BUG_MIN}")

    # ---- 4: visible non-leak + clean dissimilar tie ----
    tie_present = False
    for q in visible:
        a = gf.analyze(q["text"], q["embedding"], bm25, dense)
        check(not a["dangerous"], f"{q['query_id']}: visible exposes a dangerous bug {a['flags']}")
        check(a["bm25_n"] > 0 and a["dense_n"] > 0, f"{q['query_id']}: visible empty ranking")
        check(a["fused_len"] >= top_k, f"{q['query_id']}: visible fused < top_k")
        if a["tie_clean"]:
            tie_present = True
    check(tie_present, "no visible query has a clean dissimilar-prefix tie")

    # ---- 5: clean /app == reference (no bugs planted) ----
    app_cfg = SearchConfig(top_k=top_k, candidate_depth=depth, rrf_k=rrf_k,
                           bm25_k1=cfg["bm25_k1"], bm25_b=cfg["bm25_b"])
    app_docs = [Document(doc_id=d, text=t) for d, t in docs]
    pipeline = RetrievalPipeline(app_docs, vectors, app_cfg)
    for q in hidden:
        res = pipeline.search(Query(q["query_id"], q["text"], q["embedding"]))
        exp = expected["queries"][q["query_id"]]
        check([e.doc_id for e in res.bm25] == exp["bm25"], f"{q['query_id']}: app bm25 != reference")
        check([e.doc_id for e in res.dense] == exp["dense"], f"{q['query_id']}: app dense != reference")
        check([e.doc_id for e in res.fused] == exp["fused"], f"{q['query_id']}: app fused != reference")
    for q in visible:
        res = pipeline.search(Query(q["query_id"], q["text"], q["embedding"]))
        ref_bm = [d for d, _ in bm25.rank(q["text"], depth)]
        ref_dn = [d for d, _ in dense.rank(q["embedding"], depth)]
        ref_fu = [d for d, _ in reference_rrf.fuse(
            [bm25.rank(q["text"], depth), dense.rank(q["embedding"], depth)], k=rrf_k, top_k=top_k)]
        check([e.doc_id for e in res.bm25] == ref_bm, f"{q['query_id']}: app bm25 != reference (visible)")
        check([e.doc_id for e in res.dense] == ref_dn, f"{q['query_id']}: app dense != reference (visible)")
        check([e.doc_id for e in res.fused] == ref_fu, f"{q['query_id']}: app fused != reference (visible)")

    # ---- 6: determinism digest ----
    digest = hashlib.sha256("\n".join(digest_parts).encode()).hexdigest()

    print("=== Phase 2 validation ===")
    print(f"corpus={len(docs)} dim={vectors.shape[1]} hidden={len(hidden)} visible={len(visible)}")
    print(f"config: top_k={top_k} candidate_depth={depth} rrf_k={rrf_k}")
    print("coverage counts:", counts, "targets:", gf.HIDDEN_TARGETS)
    print(f"multi-structural(>=3) hidden queries: {multi}")
    print(f"cross-modality overlap>={OVERLAP_MIN}: {overlap_ok}/{len(hidden)} queries")
    print(f"visible clean dissimilar-tie present: {tie_present}")
    print(f"expected/app/reference agreement: {'OK' if not FAILURES else 'FAIL'}")
    print(f"DETERMINISM_DIGEST {digest}")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print("  -", f)
        print("\nPHASE 2 GATE: FAIL")
        return 1
    print("\nPHASE 2 GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
