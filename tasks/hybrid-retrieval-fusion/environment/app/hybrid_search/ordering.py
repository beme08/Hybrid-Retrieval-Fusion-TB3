"""Ordering and rank helpers for retrieval outputs."""

from __future__ import annotations

from typing import Iterable, List, MutableSequence, Tuple

from .models import RankingEntry


def score_desc_doc_id_key(entry: RankingEntry) -> Tuple[float, str]:
    """Return a reusable ordering key for ranked entries."""
    return (-entry.score, entry.doc_id)


def sort_by_score_then_doc_id(entries: MutableSequence[RankingEntry]) -> None:
    """Sort entries in place."""
    entries.sort(key=score_desc_doc_id_key)


def assign_source_ranks(entries: Iterable[RankingEntry], start: int = 1) -> None:
    """Assign contiguous source ranks to an already ordered ranking."""
    for rank, entry in enumerate(entries, start=start):
        entry.source_rank = rank


def _fused_order_key(entry: RankingEntry) -> Tuple[float, object]:
    return (-entry.score, hash(entry.doc_id))


def order_fused_entries(entries: List[RankingEntry]) -> None:
    """Order fused entries in place."""
    entries.sort(key=_fused_order_key)
