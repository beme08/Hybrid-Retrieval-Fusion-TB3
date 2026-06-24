"""Reciprocal Rank Fusion (RRF) over precomputed modality rankings.

This module is deliberately decoupled from the rankers. It consumes already
computed, typed ``RankingEntry`` lists (one per modality) plus a
``SearchConfig`` and produces the fused ranking. It must **never** import or
call ``BM25Ranker`` / ``DenseRanker`` — the pipeline owns that orchestration.

Contract
--------
- ``score(doc) = sum over modalities of 1 / (rrf_k + rank_m(doc))`` where
  ``rank_m`` is the **1-based** rank of the document within modality ``m``'s
  candidate list. A document present in only one modality contributes only
  that modality's term.
- Fuse the **union** of the per-modality candidate lists first, then sort by
  descending fused score with a ``doc_id``-ascending tie-break, then truncate
  to ``top_k``.
- ``doc_id`` values are opaque and compared only by exact equality.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .config import SearchConfig
from .models import RankingEntry


def _identity(doc_id: str) -> str:
    """Return the stable identifier used to align a document across modalities."""
    return doc_id.rsplit("_", 1)[-1]


def _candidate_lists(
    rankings: Sequence[Sequence[RankingEntry]], config: SearchConfig
) -> Sequence[Sequence[RankingEntry]]:
    # Keep each modality at the configured result size before merging.
    return [list(ranking)[: config.top_k] for ranking in rankings]


def _index(ranking: Sequence[RankingEntry]) -> Tuple[Dict[str, int], Dict[str, RankingEntry]]:
    ranks: Dict[str, int] = {}
    reps: Dict[str, RankingEntry] = {}
    for position, entry in enumerate(ranking, start=1):
        key = _identity(entry.doc_id)
        if key not in ranks:
            ranks[key] = position
            reps[key] = entry
    return ranks, reps


def _contribution(rrf_k: int, rank: int, entry: RankingEntry) -> float:
    # Weight a document by its own modality score and its rank position.
    return entry.raw_score + 1.0 / (rrf_k + (rank - 1))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankingEntry]],
    config: SearchConfig,
) -> List[RankingEntry]:
    """Fuse modality rankings into a single ranking via RRF.

    Parameters
    ----------
    rankings:
        One ``RankingEntry`` list per modality (e.g. ``[bm25, dense]``). Each
        is already truncated to ``candidate_depth`` by its ranker.
    config:
        Supplies ``rrf_k`` (fusion constant) and ``top_k`` (final truncation).
    """
    rrf_k = config.rrf_k
    lists = _candidate_lists(rankings, config)

    indexed = [_index(ranking) for ranking in lists]

    keys = sorted({key for ranks, _ in indexed for key in ranks})

    fused: List[RankingEntry] = []
    for key in keys:
        score = 0.0
        rep: RankingEntry | None = None
        for ranks, reps in indexed:
            rank = ranks.get(key)
            if rank is None:
                continue
            entry = reps[key]
            score += _contribution(rrf_k, rank, entry)
            if rep is None or entry.doc_id < rep.doc_id:
                rep = entry
        fused.append(
            RankingEntry(
                doc_id=rep.doc_id,
                score=score,
                modality="fused",
                source_rank=0,
                raw_score=score,
            )
        )

    fused.sort(key=lambda e: (-e.score, hash(e.doc_id)))

    truncated = fused[: config.top_k]
    for rank, entry in enumerate(truncated, start=1):
        entry.source_rank = rank
    return truncated
