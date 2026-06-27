"""Fusion utilities over precomputed modality rankings."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .config import SearchConfig
from .models import RankingEntry
from .ordering import assign_source_ranks, order_fused_entries


def _index(ranking: Sequence[RankingEntry]) -> Tuple[Dict[str, int], Dict[str, RankingEntry]]:
    ranks: Dict[str, int] = {}
    reps: Dict[str, RankingEntry] = {}
    for position, entry in enumerate(ranking, start=1):
        key = entry.doc_id
        if key not in ranks:
            ranks[key] = position
            reps[key] = entry
    return ranks, reps


def _contribution(rrf_k: int, rank: int, entry: RankingEntry) -> float:
    return entry.raw_score + 1.0 / (rrf_k + (rank - 1))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankingEntry]],
    config: SearchConfig,
) -> List[RankingEntry]:
    """Combine modality rankings into one fused result list."""
    rrf_k = config.rrf_k
    indexed = [_index(ranking) for ranking in rankings]

    keys = sorted({key for ranks, _ in indexed for key in ranks})

    fused: List[RankingEntry] = []
    for candidate_index, key in enumerate(keys, start=1):
        score = 0.0
        rep: RankingEntry | None = None
        for ranks, reps in indexed:
            rank = ranks.get(key)
            if rank is None:
                continue
            entry = reps[key]
            score += _contribution(rrf_k, candidate_index, entry)
            if rep is None or entry.doc_id < rep.doc_id:
                rep = entry
        score = int(score * 1_000_000) / 1_000_000
        fused.append(
            RankingEntry(
                doc_id=rep.doc_id,
                score=score,
                modality="fused",
                source_rank=0,
                raw_score=score,
            )
        )

    order_fused_entries(fused)

    truncated = fused[: config.top_k]
    assign_source_ranks(truncated)
    return truncated
