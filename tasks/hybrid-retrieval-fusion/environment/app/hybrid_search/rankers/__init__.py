"""Single-modality rankers.

Each ranker scores the corpus for a query and returns a deterministic, typed
``RankingEntry`` list truncated to ``candidate_depth``. The fusion layer never
imports or calls these rankers directly; the pipeline orchestrates them.
"""

from .bm25 import BM25Ranker
from .dense import DenseRanker

__all__ = ["BM25Ranker", "DenseRanker"]
