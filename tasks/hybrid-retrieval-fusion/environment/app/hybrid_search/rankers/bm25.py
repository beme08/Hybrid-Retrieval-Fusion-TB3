"""Deterministic Okapi BM25 lexical ranker.

This modality is part of the *correct* inherited service. Its lexical ranking
is not the source of the regression; do not change its scoring contract.

Contract
--------
- Tokenizer: lowercase, then the regex ``[a-z0-9]+``.
- ``k1 = 1.2``, ``b = 0.75``.
- ``idf(term) = log(1 + (N - df + 0.5) / (df + 0.5))`` (always non-negative).
- Final ordering: sort by ``(-score, doc_id)`` and keep ``candidate_depth``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

from ..models import Document, Query, RankingEntry

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase then extract ``[a-z0-9]+`` runs, in order."""
    return _TOKEN_RE.findall(text.lower())


class BM25Ranker:
    """Okapi BM25 over a fixed corpus.

    The index (document frequencies, idf, term frequencies, lengths) is built
    once at construction so that repeated queries are cheap and deterministic.
    """

    def __init__(
        self,
        corpus: Sequence[Document],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = list(corpus)
        self.doc_count = len(self.documents)

        self._doc_tokens: List[Counter] = []
        self._doc_len: List[int] = []
        df: Counter = Counter()
        for doc in self.documents:
            tokens = tokenize(doc.text)
            tf = Counter(tokens)
            self._doc_tokens.append(tf)
            self._doc_len.append(len(tokens))
            for term in tf:
                df[term] += 1

        total_len = sum(self._doc_len)
        self.avg_doc_len = (total_len / self.doc_count) if self.doc_count else 0.0

        self._idf: Dict[str, float] = {}
        for term, term_df in df.items():
            self._idf[term] = math.log(
                1.0 + (self.doc_count - term_df + 0.5) / (term_df + 0.5)
            )

    def _score_doc(self, query_terms: Counter, doc_index: int) -> float:
        tf = self._doc_tokens[doc_index]
        doc_len = self._doc_len[doc_index]
        denom_len = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len)) \
            if self.avg_doc_len else self.k1
        score = 0.0
        for term in query_terms:
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue
            idf = self._idf.get(term)
            if idf is None:
                continue
            score += idf * (term_tf * (self.k1 + 1.0)) / (term_tf + denom_len)
        return score

    def search(self, query: Query, candidate_depth: int) -> List[RankingEntry]:
        """Return the top ``candidate_depth`` documents for ``query``.

        Documents with a zero score are omitted. Ties are broken by ascending
        ``doc_id`` so the ranking is fully deterministic.
        """
        query_terms = Counter(tokenize(query.text))
        scored: List[RankingEntry] = []
        for index, doc in enumerate(self.documents):
            score = self._score_doc(query_terms, index)
            if score <= 0.0:
                continue
            scored.append(
                RankingEntry(
                    doc_id=doc.doc_id,
                    score=score,
                    modality="bm25",
                    source_rank=0,
                    raw_score=score,
                )
            )
        scored.sort(key=lambda e: (-e.score, e.doc_id))
        ranked = scored[:candidate_depth]
        for rank, entry in enumerate(ranked, start=1):
            entry.source_rank = rank
        return ranked
