"""Retrieval pipeline orchestration.

The pipeline is the only place that knows about both the rankers and the
fusion layer. For each query it:

1. asks each modality ranker for ``candidate_depth`` results,
2. hands the typed rankings to the fusion layer, and
3. returns a typed per-query result bundle.

The fusion layer is invoked through ``reciprocal_rank_fusion`` and never the
other way around, keeping the dependency direction one-way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .candidate_utils import coalesce_candidate_identities
from .config import SearchConfig
from .fusion import reciprocal_rank_fusion
from .models import Document, Query, RankingEntry
from .rankers import BM25Ranker, DenseRanker


@dataclass(frozen=True)
class QueryResult:
    """The three rankings produced for a single query."""

    query: Query
    bm25: List[RankingEntry]
    dense: List[RankingEntry]
    fused: List[RankingEntry]


class RetrievalPipeline:
    """Builds the modality rankers once and runs queries through fusion."""

    def __init__(
        self,
        corpus: Sequence[Document],
        vectors: np.ndarray,
        config: SearchConfig,
    ) -> None:
        self.corpus: List[Document] = list(corpus)
        self.config = config
        self._bm25 = BM25Ranker(
            self.corpus, k1=config.bm25_k1, b=config.bm25_b
        )
        self._dense = DenseRanker(self.corpus, vectors)

    def search(self, query: Query) -> QueryResult:
        """Run one query end to end and return its typed rankings."""
        depth = self.config.top_k
        bm25_ranking = self._bm25.search(query, depth)
        dense_ranking = self._dense.search(query, depth)
        fusion_inputs = coalesce_candidate_identities([bm25_ranking, dense_ranking])
        fused_ranking = reciprocal_rank_fusion(
            fusion_inputs, self.config
        )
        return QueryResult(
            query=query,
            bm25=bm25_ranking,
            dense=dense_ranking,
            fused=fused_ranking,
        )

    def search_all(self, queries: Sequence[Query]) -> List[QueryResult]:
        """Run every query, preserving input order."""
        return [self.search(query) for query in queries]
