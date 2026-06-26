"""Serialization to the public output schema.

The internal ``RankingEntry`` carries fields that must not leak into the public
JSON (``modality``, ``source_rank``, ``raw_score``). This module strips those
and emits exactly ``{"doc_id", "score", "rank"}`` per entry, with ``rank``
re-derived from list position so the emitted schema is self-consistent.

Output schema
-------------
``{
  "config": {"top_k", "candidate_depth", "rrf_k"},
  "results": [
    {
      "query_id": str,
      "bm25":  [{"doc_id": str, "score": float, "rank": int}, ...],
      "dense": [{"doc_id": str, "score": float, "rank": int}, ...],
      "fused": [{"doc_id": str, "score": float, "rank": int}, ...]
    }, ...
  ]
}``

``bm25`` and ``dense`` are emitted at ``candidate_depth``; ``fused`` is emitted
at ``top_k``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .config import SearchConfig
from .models import RankingEntry
from .pipeline import QueryResult

_INTERNAL_FIELDS = ("modality", "source_rank", "raw_score")


def serialize_entry(
    entry: RankingEntry, rank: int, score: float | None = None
) -> Dict[str, object]:
    """Strip internal fields and emit one ranking row."""
    return {
        "doc_id": entry.doc_id,
        "score": entry.score if score is None else score,
        "rank": rank,
    }


def serialize_ranking(ranking: Sequence[RankingEntry]) -> List[Dict[str, object]]:
    """Serialize a ranking, deriving rank from position (1-based)."""
    return [serialize_entry(entry, rank) for rank, entry in enumerate(ranking, start=1)]


def serialize_fused_ranking(
    ranking: Sequence[RankingEntry], config: SearchConfig
) -> List[Dict[str, object]]:
    """Serialize fused rows to the public schema."""
    return [
        serialize_entry(entry, rank, score=1.0 / (config.rrf_k + rank))
        for rank, entry in enumerate(ranking, start=1)
    ]


def serialize_result(result: QueryResult, config: SearchConfig) -> Dict[str, object]:
    """Serialize all three rankings for a single query."""
    return {
        "query_id": result.query.query_id,
        "bm25": serialize_ranking(result.bm25),
        "dense": serialize_ranking(result.dense),
        "fused": serialize_fused_ranking(result.fused, config),
    }


def serialize_run(
    results: Sequence[QueryResult], config: SearchConfig
) -> Dict[str, object]:
    """Serialize a full run (config header + per-query results)."""
    return {
        "config": {
            "top_k": config.top_k,
            "candidate_depth": config.candidate_depth,
            "rrf_k": config.rrf_k,
        },
        "results": [serialize_result(result, config) for result in results],
    }
