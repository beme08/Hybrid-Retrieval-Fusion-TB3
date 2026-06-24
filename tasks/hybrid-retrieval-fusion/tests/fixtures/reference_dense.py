"""Independent reference dense cosine ranker (verifier-owned).

Separate implementation of the dense contract from ``/app``. Scores documents
by cosine similarity between a query embedding and the shipped corpus vectors,
ordering by ``(-score, doc_id)``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class ReferenceDense:
    def __init__(self, doc_ids: Sequence[str], vectors: np.ndarray) -> None:
        self.doc_ids: List[str] = list(doc_ids)
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.shape[0] != len(self.doc_ids):
            raise ValueError("vector rows must match number of doc_ids")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.unit = matrix / norms

    def rank(self, embedding: Sequence[float], depth: int) -> List[Tuple[str, float]]:
        q = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            norm = 1.0
        sims = self.unit @ (q / norm)
        results = [(self.doc_ids[i], float(sims[i])) for i in range(len(self.doc_ids))]
        results.sort(key=lambda pair: (-pair[1], pair[0]))
        return results[:depth]
