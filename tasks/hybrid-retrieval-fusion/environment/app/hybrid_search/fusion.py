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

from typing import Dict, List, Sequence

from .config import SearchConfig
from .models import RankingEntry


def _rank_map(ranking: Sequence[RankingEntry]) -> Dict[str, int]:
    """Map ``doc_id -> 1-based rank`` for one modality ranking.

    The entries arrive already ordered, but we rely on ``source_rank`` so the
    rank is explicit and independent of any later list mutation.
    """
    ranks: Dict[str, int] = {}
    for entry in ranking:
        # First occurrence wins; rankings are expected to be unique by doc_id.
        if entry.doc_id not in ranks:
            ranks[entry.doc_id] = entry.source_rank
    return ranks


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

    rank_maps = [_rank_map(ranking) for ranking in rankings]

    # Union of all candidate doc_ids across modalities, deterministically.
    union: List[str] = sorted({doc_id for rm in rank_maps for doc_id in rm})

    fused: List[RankingEntry] = []
    for doc_id in union:
        score = 0.0
        for rm in rank_maps:
            rank = rm.get(doc_id)
            if rank is not None:
                score += 1.0 / (rrf_k + rank)
        fused.append(
            RankingEntry(
                doc_id=doc_id,
                score=score,
                modality="fused",
                source_rank=0,
                raw_score=score,
            )
        )

    fused.sort(key=lambda e: (-e.score, e.doc_id))
    truncated = fused[: config.top_k]
    for rank, entry in enumerate(truncated, start=1):
        entry.source_rank = rank
    return truncated
