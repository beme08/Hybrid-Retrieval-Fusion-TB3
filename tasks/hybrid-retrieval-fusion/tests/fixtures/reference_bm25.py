"""Independent reference BM25 (verifier-owned).

This is a deliberately *separate* implementation of the BM25 contract from the
one shipped in ``/app``. The verifier uses it to generate frozen expected
rankings, so it must not import from ``hybrid_search``. The two implementations
share only the normative contract:

- tokenizer: lowercase then ``[a-z0-9]+``
- ``k1 = 1.2``, ``b = 0.75``
- ``idf = log(1 + (N - df + 0.5) / (df + 0.5))``
- order by ``(-score, doc_id)``; documents scoring 0 are dropped.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class ReferenceBM25:
    def __init__(
        self,
        documents: Sequence[Tuple[str, str]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = [doc_id for doc_id, _ in documents]
        self.term_freqs: List[Counter] = [Counter(tokenize(text)) for _, text in documents]
        self.lengths: List[int] = [sum(tf.values()) for tf in self.term_freqs]
        self.n_docs = len(self.doc_ids)
        self.avg_len = (sum(self.lengths) / self.n_docs) if self.n_docs else 0.0

        doc_freq: Counter = Counter()
        for tf in self.term_freqs:
            for term in tf:
                doc_freq[term] += 1
        self.idf: Dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def rank(self, text: str, depth: int) -> List[Tuple[str, float]]:
        query = Counter(tokenize(text))
        results: List[Tuple[str, float]] = []
        for i, doc_id in enumerate(self.doc_ids):
            tf = self.term_freqs[i]
            length = self.lengths[i]
            denom_len = (
                self.k1 * (1.0 - self.b + self.b * (length / self.avg_len))
                if self.avg_len
                else self.k1
            )
            score = 0.0
            for term, _ in query.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term)
                if idf is None:
                    continue
                score += idf * (freq * (self.k1 + 1.0)) / (freq + denom_len)
            if score > 0.0:
                results.append((doc_id, score))
        results.sort(key=lambda pair: (-pair[1], pair[0]))
        return results[:depth]
