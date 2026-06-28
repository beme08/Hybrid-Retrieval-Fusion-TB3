#!/usr/bin/env python3
"""Deterministic fixture + expected-output generator for hybrid-retrieval-fusion.

Repo-root build tooling (NOT shipped to evaluation agents, NOT a verifier
dependency). Produces:

Visible (agent-facing) data -> tasks/.../environment/data/
    corpus.json            full corpus (opaque ids; main region has suffix
                           collisions; appended "clean zones" power non-leaky
                           visible queries)
    vectors.npy            aligned dense matrix, dim 96, rounded to 6 decimals
    visible_queries.json   4 non-leaky visible queries (with embeddings)

Hidden (verifier-only) fixtures -> tasks/.../tests/fixtures/
    hidden_queries.json    12 hidden queries (with embeddings)
    expected_hidden.json   frozen expected rankings, generated from the
                           INDEPENDENT reference under tests/fixtures
    coverage_report.json   bug-activation matrix used for selection/auditing

No bugs are planted. To guarantee the (future Phase-3) bugs are catchable, this
generator SIMULATES each planned bug's effect (build-time only) and selects
hidden queries whose correct rankings diverge from the buggy ones. The shipped
reference under tests/fixtures stays clean.

Bug -> verifier detection mapping (honest):
  * raw_mixing        -> fused doc-id ORDER diverges  (Bank 1)
  * truncate          -> fused doc-id ORDER diverges  (Bank 1)
  * suffix_collision  -> fused doc-id ORDER diverges  (Bank 1)
  * rank_base/const   -> fused SCORES diverge / exceed 2/61 (Bank 2 + range
                         guard); applies to every fused-non-empty query and
                         rarely changes doc-id order
  * tie_break         -> nondeterministic order on fused ties (determinism run)

"Visible non-leak" therefore means: the four ORDER/structural bugs do not change
visible doc-id output; only tie-break determinism may be exposed.

Determinism: all randomness seeded; no reliance on set/dict iteration order.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "hybrid-retrieval-fusion"
DATA_DIR = TASK_DIR / "environment" / "data"
FIXTURE_DIR = TASK_DIR / "tests" / "fixtures"

sys.path.insert(0, str(FIXTURE_DIR))
from reference_bm25 import ReferenceBM25, tokenize  # noqa: E402
from reference_dense import ReferenceDense  # noqa: E402
import reference_rrf  # noqa: E402

# ---- Canonical task config (verifier runs the app with these exact values) ----
TOP_K = 10
CANDIDATE_DEPTH = 25
RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75
DIM = 96
SEED = 20240624

PREFIXES = ["blog", "faq", "guide", "news", "paper", "wiki"]
N_SUFFIXES = 40  # 6 prefixes x 40 suffixes = 240 main documents

TOPICS: Dict[str, List[str]] = {
    "monetary": ["inflation", "interest", "rates", "central", "bank", "monetary", "yield", "bond"],
    "weather": ["hurricane", "storm", "coastal", "forecast", "rainfall", "flood", "warning", "wind"],
    "cycling": ["bicycle", "cycling", "lane", "commute", "pedal", "saddle", "gravel", "trail"],
    "transformers": ["transformer", "attention", "neural", "translation", "encoder", "decoder", "tokenizer", "sequence"],
    "retrieval": ["retrieval", "lexical", "bm25", "ranker", "inverted", "okapi", "posting", "term"],
    "embeddings": ["embedding", "cosine", "vector", "semantic", "similarity", "nearest", "latent", "manifold"],
    "accounts": ["password", "account", "login", "reset", "credential", "session", "username", "lockout"],
    "payments": ["payment", "subscription", "invoice", "billing", "checkout", "refund", "card", "upgrade"],
    "dataexport": ["export", "archive", "download", "backup", "dump", "snapshot", "migrate", "tarball"],
    "security": ["security", "encryption", "phishing", "breach", "firewall", "malware", "patch", "audit"],
    "mortgage": ["mortgage", "housing", "loan", "amortization", "escrow", "refinance", "principal", "equity"],
    "qasystems": ["question", "answering", "augmented", "passage", "context", "grounding", "prompt", "rerank"],
    "databases": ["database", "shard", "replica", "consistency", "partition", "throughput", "wal", "compaction"],
    "imaging": ["pixel", "convolution", "segmentation", "grayscale", "filter", "kernel", "resolution", "histogram"],
}
TOPIC_NAMES = sorted(TOPICS)
COMMON = ["overview", "guide", "notes", "summary", "details", "report", "update", "introduction"]

# Bait tokens: a rare token tied to a specific suffix across ALL prefixes, so a
# query containing it retrieves several same-suffix docs (e.g. blog_007,
# faq_007, ...) -> strong suffix-collision pressure.
BAIT_TOKENS: Dict[int, str] = {
    7: "zephyrqx", 13: "quokkabyte", 19: "nimbusvex", 23: "fjordglyph",
    29: "umbracoil", 31: "voltibark", 37: "cobaltwisp",
}


# ============================ corpus / vectors ============================
def build_main_corpus() -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for suffix in range(1, N_SUFFIXES + 1):
        for p_index, prefix in enumerate(PREFIXES):
            doc_id = f"{prefix}_{suffix:03d}"
            rng = random.Random(SEED * 1000 + suffix * 10 + p_index)
            topic = TOPIC_NAMES[(suffix * 7 + p_index * 3) % len(TOPIC_NAMES)]
            tokens = [rng.choice(TOPICS[topic]) for _ in range(rng.randint(9, 15))]
            tokens += [rng.choice(COMMON) for _ in range(rng.randint(1, 3))]
            sec = TOPIC_NAMES[(suffix * 11 + p_index * 5 + 4) % len(TOPIC_NAMES)]
            tokens += [rng.choice(TOPICS[sec]) for _ in range(rng.randint(1, 3))]
            if suffix in BAIT_TOKENS:
                tokens.append(BAIT_TOKENS[suffix])
            rng.shuffle(tokens)
            docs.append((doc_id, " ".join(tokens)))
    return docs


# Clean zones power non-leaky visible queries. Each "identical" doc carries only
# its zone token and is embedded exactly at the zone anchor (cosine 1), so the
# four order/structural bugs are provable no-ops on the zone's query.
SIMPLE_ZONES = [
    ("clnzonealpha", "A", [("blog", 901), ("faq", 902), ("guide", 903), ("news", 904),
                            ("paper", 905), ("wiki", 906), ("blog", 907), ("faq", 908),
                            ("guide", 909), ("news", 910)]),
    ("clnzonebeta", "B", [("blog", 911), ("faq", 912), ("guide", 913), ("news", 914),
                           ("paper", 915), ("wiki", 916), ("blog", 917), ("faq", 918),
                           ("guide", 919), ("news", 920)]),
    ("clnzonegamma", "C", [("blog", 921), ("faq", 922), ("guide", 923), ("news", 924),
                            ("paper", 925), ("wiki", 926), ("blog", 927), ("faq", 928),
                            ("guide", 929), ("news", 930)]),
    ("clnzonedelta", "D", [("blog", 931), ("faq", 932), ("guide", 933), ("news", 934),
                            ("paper", 935), ("wiki", 936), ("blog", 937), ("faq", 938),
                            ("guide", 939), ("news", 940)]),
]
# (Legacy tie zone retained as constants below but no longer emitted.) Tie zone: 8 identical (cosine 1) + P (bm25-only, anti-anchor) + Q (dense-only,
# anchor, no token). P and Q land at symmetric single-modality rank 9 -> exact
# fused tie at positions 9/10 with dissimilar prefixes (news_950 vs paper_951).
TIE_ZONE_TOKEN = "clnzonetie"
TIE_ZONE_IDENTICAL = [("blog", 931), ("blog", 932), ("blog", 933), ("blog", 934),
                      ("blog", 935), ("blog", 936), ("blog", 937), ("blog", 938)]
TIE_ZONE_P = ("news", 950)   # bm25-only
TIE_ZONE_Q = ("paper", 951)  # dense-only


def zone_anchor(zone_key: str) -> np.ndarray:
    rng = np.random.default_rng(SEED + 31337 + sum(ord(c) for c in zone_key))
    return np.round(rng.standard_normal(DIM), 6)


def build_clean_docs():
    """Return (docs, overrides) where overrides maps doc_id -> embedding row."""
    docs: List[Tuple[str, str]] = []
    overrides: Dict[str, np.ndarray] = {}
    for token, key, slots in SIMPLE_ZONES:
        anchor = zone_anchor(key)
        for prefix, suffix in slots:
            doc_id = f"{prefix}_{suffix:03d}"
            docs.append((doc_id, " ".join([token] * 3)))
            overrides[doc_id] = anchor
    return docs, overrides


def build_token_vectors(vocab: Sequence[str]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED + 99)
    return {tok: rng.standard_normal(DIM) for tok in sorted(set(vocab))}


def embed(tokens, token_vecs, noise_rng) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float64)
    for tok in tokens:
        v = token_vecs.get(tok)
        if v is not None:
            vec += v
    vec += 0.12 * noise_rng.standard_normal(DIM)
    return vec


def build_vectors(docs, token_vecs, overrides) -> np.ndarray:
    matrix = np.zeros((len(docs), DIM), dtype=np.float64)
    for i, (doc_id, text) in enumerate(docs):
        if doc_id in overrides:
            matrix[i] = overrides[doc_id]
        else:
            noise = np.random.default_rng(SEED + 7 + i)
            matrix[i] = embed(tokenize(text), token_vecs, noise)
    return np.round(matrix, 6)


# ============================ query pool (hidden) ============================
def build_query_pool() -> List[Tuple[str, str]]:
    pool: List[Tuple[str, str]] = []
    rng = random.Random(SEED + 500)
    for topic in TOPIC_NAMES:
        vocab = TOPICS[topic]
        for j in range(4):
            n = rng.randint(2, 4)
            pool.append((f"t_{topic}_{j}", " ".join(vocab[(j + t) % len(vocab)] for t in range(n))))
    for a in range(len(TOPIC_NAMES)):
        b = (a + 5) % len(TOPIC_NAMES)
        pool.append((f"c_{TOPIC_NAMES[a]}_{TOPIC_NAMES[b]}",
                     " ".join(TOPICS[TOPIC_NAMES[a]][:2] + TOPICS[TOPIC_NAMES[b]][:2])))
    for suffix, tok in BAIT_TOKENS.items():
        pool.append((f"b_{suffix}_solo", tok))
        pool.append((f"b_{suffix}_mix", f"{tok} {TOPICS[TOPIC_NAMES[suffix % len(TOPIC_NAMES)]][0]}"))
    for topic in TOPIC_NAMES:
        pool.append((f"s_{topic}", TOPICS[topic][-1]))
    for a in range(0, len(TOPIC_NAMES), 2):
        b, c = (a + 3) % len(TOPIC_NAMES), (a + 7) % len(TOPIC_NAMES)
        pool.append((f"d_{a}", " ".join(TOPICS[TOPIC_NAMES[a]][:2] + TOPICS[TOPIC_NAMES[b]][:1] + TOPICS[TOPIC_NAMES[c]][:1])))
    return pool


# ============================ bug simulations ============================
def suffix_of(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[-1]


def prefix_of(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]


def correct_ids(bm25_cand, dense_cand):
    return [d for d, _ in reference_rrf.fuse([bm25_cand, dense_cand], k=RRF_K, top_k=TOP_K)]


def correct_full(bm25_cand, dense_cand):
    return reference_rrf.fuse([bm25_cand, dense_cand], k=RRF_K, top_k=10 ** 9)


def sim_raw_mixing(bm25_cand, dense_cand):
    bm = {d: s for d, s in bm25_cand}
    dn = {d: s for d, s in dense_cand}
    scored = [(d, bm.get(d, 0.0) + dn.get(d, 0.0)) for d in sorted(set(bm) | set(dn))]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scored[:TOP_K]]


def sim_rank_base_zero(bm25_cand, dense_cand):
    maps = []
    for cand in (bm25_cand, dense_cand):
        m = {}
        for pos, (d, _) in enumerate(cand, start=1):
            m.setdefault(d, pos)
        maps.append(m)
    scored = []
    for d in sorted(set(maps[0]) | set(maps[1])):
        s = sum(1.0 / (RRF_K + (m[d] - 1)) for m in maps if d in m)
        scored.append((d, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scored[:TOP_K]]


def sim_truncate(bm25_full, dense_full):
    return [d for d, _ in reference_rrf.fuse([bm25_full[:TOP_K], dense_full[:TOP_K]], k=RRF_K, top_k=TOP_K)]


def sim_suffix_collision(bm25_cand, dense_cand):
    maps = []
    for cand in (bm25_cand, dense_cand):
        m = {}
        for pos, (d, _) in enumerate(cand, start=1):
            key = suffix_of(d)
            if key not in m:
                m[key] = (pos, d)
        maps.append(m)
    scored = []
    for key in sorted(set(maps[0]) | set(maps[1])):
        s, rep = 0.0, None
        for m in maps:
            if key in m:
                pos, d = m[key]
                s += 1.0 / (RRF_K + pos)
                rep = d if rep is None else min(rep, d)
        scored.append((rep, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scored[:TOP_K]]


def tie_info(bm25_cand, dense_cand):
    full = correct_full(bm25_cand, dense_cand)
    groups: Dict[float, List[Tuple[int, str]]] = {}
    for pos, (d, s) in enumerate(full):
        groups.setdefault(s, []).append((pos, d))
    activates, dissimilar = False, False
    for members in groups.values():
        if len(members) < 2:
            continue
        if min(p for p, _ in members) < TOP_K:
            activates = True
            ids = [d for _, d in members]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if prefix_of(ids[i]) != prefix_of(ids[j]) and suffix_of(ids[i]) != suffix_of(ids[j]):
                        dissimilar = True
    return activates, dissimilar


def analyze(text, embedding, bm25, dense):
    bm25_full = bm25.rank(text, 10 ** 9)
    dense_full = dense.rank(embedding, 10 ** 9)
    bm25_cand, dense_cand = bm25_full[:CANDIDATE_DEPTH], dense_full[:CANDIDATE_DEPTH]
    correct = correct_ids(bm25_cand, dense_cand)
    tie_act, tie_clean = tie_info(bm25_cand, dense_cand)
    flags = {
        "raw_mixing": sim_raw_mixing(bm25_cand, dense_cand) != correct,
        "truncate": sim_truncate(bm25_full, dense_full) != correct,
        "suffix_collision": sim_suffix_collision(bm25_cand, dense_cand) != correct,
        "rank_base_order": sim_rank_base_zero(bm25_cand, dense_cand) != correct,
        "rank_base_detect": len(correct) > 0,  # Bank-2/range-guard catches it everywhere
        "modality_weight": [d for d, _ in reference_rrf.fuse([bm25_cand, dense_cand], k=RRF_K, top_k=TOP_K, weights=[1.0, 1.0])] != correct,
    }
    dangerous = flags["raw_mixing"] or flags["truncate"] or flags["suffix_collision"] or flags["rank_base_order"] or flags["modality_weight"]
    structural = sum(1 for k in ("raw_mixing", "truncate", "suffix_collision", "modality_weight") if flags[k])
    return {
        "bm25_n": len(bm25_cand), "dense_n": len(dense_cand), "fused_len": len(correct),
        "overlap": len(set(d for d, _ in bm25_cand) & set(d for d, _ in dense_cand)),
        "flags": flags, "dangerous": dangerous, "structural_bugs": structural,
        "tie_clean": tie_clean,
    }


# ============================ selection (hidden) ============================
HIDDEN_TARGETS = {"raw_mixing": 3, "truncate": 2, "suffix_collision": 3, "rank_base_detect": 2, "modality_weight": 3}
N_HIDDEN = 12
MULTI_BUG_MIN = 2  # >=2 hidden queries with >=3 simultaneous STRUCTURAL bug classes


def select_hidden(an: Dict[str, dict]) -> List[str]:
    valid = [q for q, a in sorted(an.items()) if a["bm25_n"] > 0 and a["dense_n"] > 0 and a["fused_len"] >= TOP_K]
    chosen: List[str] = []
    counts = {k: 0 for k in HIDDEN_TARGETS}
    multi = 0
    for q in sorted(valid, key=lambda x: (-an[x]["structural_bugs"], x)):
        if multi >= MULTI_BUG_MIN:
            break
        if an[q]["structural_bugs"] >= 3:
            chosen.append(q)
            multi += 1
            for k in counts:
                counts[k] += int(an[q]["flags"][k])
    for bug, target in HIDDEN_TARGETS.items():
        for q in valid:
            if counts[bug] >= target or len(chosen) >= N_HIDDEN:
                break
            if q not in chosen and an[q]["flags"][bug]:
                chosen.append(q)
                for k in counts:
                    counts[k] += int(an[q]["flags"][k])
    for q in sorted(valid, key=lambda x: (-an[x]["structural_bugs"], x)):
        if len(chosen) >= N_HIDDEN:
            break
        if q not in chosen:
            chosen.append(q)
    return chosen[:N_HIDDEN]


# ============================ expected (from reference) ============================
def expected_for(text, embedding, bm25, dense):
    bm25_cand = bm25.rank(text, CANDIDATE_DEPTH)
    dense_cand = dense.rank(embedding, CANDIDATE_DEPTH)
    fused = reference_rrf.fuse([bm25_cand, dense_cand], k=RRF_K, top_k=TOP_K)
    return {
        "bm25": [d for d, _ in bm25_cand],
        "dense": [d for d, _ in dense_cand],
        "fused": [d for d, _ in fused],
        "fused_scores": [round(s, 12) for _, s in fused],
    }


# ============================ main ============================
def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    main_docs = build_main_corpus()
    clean_docs, overrides = build_clean_docs()
    docs = sorted(main_docs + clean_docs, key=lambda d: d[0])
    doc_ids = [d for d, _ in docs]

    all_tokens: List[str] = []
    for _, t in docs:
        all_tokens += tokenize(t)
    pool = build_query_pool()
    for _, t in pool:
        all_tokens += tokenize(t)
    token_vecs = build_token_vectors(all_tokens)
    vectors = build_vectors(docs, token_vecs, overrides)

    bm25 = ReferenceBM25(docs, k1=BM25_K1, b=BM25_B)
    dense = ReferenceDense(doc_ids, vectors)

    # ---- hidden: analyze pool, select ----
    qnoise = np.random.default_rng(SEED + 4242)
    pool_emb, pool_text, an = {}, {}, {}
    for qid, text in pool:
        emb = [round(float(x), 6) for x in embed(tokenize(text), token_vecs, qnoise)]
        pool_emb[qid], pool_text[qid] = emb, text
    for qid, text in pool:
        an[qid] = analyze(text, pool_emb[qid], bm25, dense)
    hidden_ids = select_hidden(an)

    hidden_queries = [
        {"query_id": f"hq_{i:03d}", "text": pool_text[q], "embedding": pool_emb[q], "_src": q}
        for i, q in enumerate(hidden_ids, start=1)
    ]

    # ---- visible: constructed clean-zone queries ----
    visible_specs = [
        (SIMPLE_ZONES[0][0], "A"), (SIMPLE_ZONES[1][0], "B"),
        (SIMPLE_ZONES[2][0], "C"), (SIMPLE_ZONES[3][0], "D"),
    ]
    visible_queries, visible_an = [], {}
    for i, (token, key) in enumerate(visible_specs, start=1):
        emb = [round(float(x), 6) for x in zone_anchor(key)]
        a = analyze(token, emb, bm25, dense)
        qid = f"vq_{i:03d}"
        visible_queries.append({"query_id": qid, "text": token, "embedding": emb})
        visible_an[qid] = a

    # ---- expected hidden (independent reference) ----
    expected = {
        "config": {"top_k": TOP_K, "candidate_depth": CANDIDATE_DEPTH, "rrf_k": RRF_K,
                   "bm25_k1": BM25_K1, "bm25_b": BM25_B},
        "queries": {q["query_id"]: expected_for(q["text"], q["embedding"], bm25, dense)
                    for q in hidden_queries},
    }

    # ---- coverage report ----
    counts = {k: 0 for k in HIDDEN_TARGETS}
    multi = []
    cov_hidden = {}
    for q in hidden_queries:
        a = an[q["_src"]]
        cov_hidden[q["query_id"]] = {"source": q["_src"], "flags": a["flags"],
                                     "structural_bugs": a["structural_bugs"], "overlap": a["overlap"]}
        for k in counts:
            counts[k] += int(a["flags"][k])
        if a["structural_bugs"] >= 3:
            multi.append(q["query_id"])
    coverage = {
        "config": expected["config"],
        "targets": HIDDEN_TARGETS, "counts": counts,
        "multi_structural_bug_queries": multi,
        "detection_notes": {
            "raw_mixing": "fused doc-id order divergence (Bank 1)",
            "truncate": "fused doc-id order divergence (Bank 1)",
            "suffix_collision": "fused doc-id order divergence (Bank 1)",
            "rank_base_detect": "fused score divergence / >2/61 range guard (Bank 2 + range)",
            "modality_weight": "fused doc-id order divergence under unweighted RRF (Bank 1)",
        },
        "hidden": cov_hidden,
        "visible": {q["query_id"]: {"token": q["text"], "flags": visible_an[q["query_id"]]["flags"],
                                    "dangerous": visible_an[q["query_id"]]["dangerous"]}
                    for q in visible_queries},
    }

    for q in hidden_queries:
        q.pop("_src", None)

    # ---- write ----
    (DATA_DIR / "corpus.json").write_text(
        json.dumps([{"doc_id": d, "text": t} for d, t in docs], indent=2) + "\n", encoding="utf-8")
    np.save(DATA_DIR / "vectors.npy", vectors)
    (DATA_DIR / "visible_queries.json").write_text(json.dumps(visible_queries, indent=2) + "\n", encoding="utf-8")
    (FIXTURE_DIR / "hidden_queries.json").write_text(json.dumps(hidden_queries, indent=2) + "\n", encoding="utf-8")
    (FIXTURE_DIR / "expected_hidden.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    (FIXTURE_DIR / "coverage_report.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    # ---- summary ----
    print(f"corpus: {len(docs)} docs ({len(main_docs)} main + {len(clean_docs)} clean) | vectors {vectors.shape} | pool {len(pool)}")
    print(f"hidden {len(hidden_queries)} | visible {len(visible_queries)}")
    print("hidden coverage counts:", counts, "targets:", HIDDEN_TARGETS)
    print("multi-structural (>=3) hidden:", multi)
    vis_safe = all(not visible_an[q["query_id"]]["dangerous"] for q in visible_queries)
    print("visible non-leaky:", vis_safe)
    ok = all(counts[k] >= v for k, v in HIDDEN_TARGETS.items()) and len(multi) >= MULTI_BUG_MIN and vis_safe
    print("SELECTION OK:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
