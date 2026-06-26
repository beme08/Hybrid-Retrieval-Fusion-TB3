"""Candidate preparation helpers used by the retrieval pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

from .models import RankingEntry


def _candidate_key(doc_id: str) -> str:
    """Return a grouping key for candidate bookkeeping."""
    return doc_id.rsplit("_", 1)[-1]


def coalesce_candidate_identities(
    rankings: Sequence[Sequence[RankingEntry]],
) -> List[List[RankingEntry]]:
    """Prepare ranked candidates for downstream scoring."""
    representatives: Dict[str, str] = {}
    for ranking in rankings:
        for entry in ranking:
            key = _candidate_key(entry.doc_id)
            representative = representatives.get(key)
            if representative is None or entry.doc_id < representative:
                representatives[key] = entry.doc_id

    prepared: List[List[RankingEntry]] = []
    for ranking in rankings:
        prepared_ranking: List[RankingEntry] = []
        for entry in ranking:
            representative = representatives[_candidate_key(entry.doc_id)]
            if representative == entry.doc_id:
                prepared_ranking.append(entry)
            else:
                prepared_ranking.append(replace(entry, doc_id=representative))
        prepared.append(prepared_ranking)
    return prepared
