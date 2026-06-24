"""Hybrid retrieval service: BM25 + dense cosine retrieval fused with RRF.

The public surface is intentionally small. Callers build a
:class:`~hybrid_search.pipeline.RetrievalPipeline` from a corpus, a vector
matrix, and a :class:`~hybrid_search.config.SearchConfig`, then run queries
through it. Ranking results are typed :class:`~hybrid_search.models.RankingEntry`
lists; serialization to the public JSON schema is handled by
:mod:`hybrid_search.serialization`.
"""

from .config import SearchConfig
from .models import Document, Query, RankingEntry

__all__ = ["SearchConfig", "Document", "Query", "RankingEntry"]
