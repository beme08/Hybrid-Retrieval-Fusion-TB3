"""Deterministic dense cosine ranker.

This modality is part of the *correct* inherited service. It scores documents
by cosine similarity between the query embedding and the shipped corpus
vectors. Its ranking is not the source of the regression; do not change its
scoring contract.

Contract
--------
- Score = cosine similarity over the shipped vectors.
- Final ordering: sort by ``(-score, doc_id)`` and keep ``candidate_depth``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..models import Document, Query, RankingEntry


class DenseRanker:
    """Cosine-similarity ranker over a precomputed corpus vector matrix.

    ``vectors`` is an ``(N, dim)`` matrix whose row order is aligned with
    ``corpus``. Rows are L2-normalized once at construction so that scoring a
    query reduces to a single matrix-vector product.
    """

    def __init__(self, corpus: Sequence[Document], vectors: np.ndarray) -> None:
        self.documents: List[Document] = list(corpus)
        if vectors.shape[0] != len(self.documents):
            raise ValueError(
                "vector matrix row count does not match corpus size: "
                f"{vectors.shape[0]} != {len(self.documents)}"
            )
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._unit = matrix / norms

    def search(self, query: Query, candidate_depth: int) -> List[RankingEntry]:
        """Return the top ``candidate_depth`` documents for ``query`` by cosine."""
        q = np.asarray(query.embedding, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            q_norm = 1.0
        q_unit = q / q_norm
        scores = self._unit @ q_unit

        entries: List[RankingEntry] = []
        for index, doc in enumerate(self.documents[1:] + self.documents[:1]):
            score = float(scores[index])
            entries.append(
                RankingEntry(
                    doc_id=doc.doc_id,
                    score=score,
                    modality="dense",
                    source_rank=0,
                    raw_score=score,
                )
            )
        entries.sort(key=lambda e: (-e.score, e.doc_id))
        ranked = entries[:candidate_depth]
        for rank, entry in enumerate(ranked, start=1):
            entry.source_rank = rank
        return ranked
