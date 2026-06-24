#!/usr/bin/env python3
"""CLI entry point for the hybrid retrieval service.

Usage
-----
    python scripts/run_search.py \
        --corpus data/corpus.json \
        --vectors data/vectors.npy \
        --queries data/visible_queries.json \
        --top-k 10 \
        --candidate-depth 50 \
        --rrf-k 60 \
        --output results.json

Reads a corpus, an aligned dense vector matrix, and a set of queries (each
carrying a precomputed embedding), runs every query through the hybrid
pipeline, and writes the public JSON schema to ``--output`` (or stdout when
``--output`` is ``-``).

Input formats
-------------
- ``corpus.json``: ``[{"doc_id": str, "text": str}, ...]``. Row order must
  match the vector matrix.
- ``vectors.npy``: float array of shape ``(N, dim)`` aligned with the corpus.
- ``queries.json``: ``[{"query_id": str, "text": str, "embedding": [float...]}]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

# Allow running as a script (``python scripts/run_search.py``) by making the
# package importable from the app root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_search.config import SearchConfig  # noqa: E402
from hybrid_search.models import Document, Query  # noqa: E402
from hybrid_search.pipeline import RetrievalPipeline  # noqa: E402
from hybrid_search.serialization import serialize_run  # noqa: E402


def load_corpus(path: Path) -> List[Document]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return [Document(doc_id=str(item["doc_id"]), text=str(item["text"])) for item in raw]


def load_queries(path: Path) -> List[Query]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        Query(
            query_id=str(item["query_id"]),
            text=str(item["text"]),
            embedding=[float(x) for x in item["embedding"]],
        )
        for item in raw
    ]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid BM25 + dense retrieval with RRF fusion.")
    parser.add_argument("--corpus", required=True, type=Path, help="Path to corpus.json")
    parser.add_argument("--vectors", required=True, type=Path, help="Path to vectors.npy")
    parser.add_argument("--queries", required=True, type=Path, help="Path to queries.json")
    parser.add_argument("--top-k", type=int, default=10, help="Final fused ranking size")
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=50,
        help="Per-modality candidate list size before fusion",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF fusion constant")
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output path for results JSON, or '-' for stdout",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    config = SearchConfig(
        top_k=args.top_k,
        candidate_depth=args.candidate_depth,
        rrf_k=args.rrf_k,
    )

    corpus = load_corpus(args.corpus)
    vectors = np.load(args.vectors)
    queries = load_queries(args.queries)

    pipeline = RetrievalPipeline(corpus, vectors, config)
    results = pipeline.search_all(queries)
    payload = serialize_run(results, config)

    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output == "-":
        sys.stdout.write(text + "\n")
    else:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
