"""Small text encoding utilities used by GEMS retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def compact_schema_text(schema: object) -> str:
    if not schema:
        return ""
    return normalize_text(schema)


def lexical_key(text: str, max_tokens: int = 10) -> str:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    return "_".join(tokens[:max_tokens]) or "empty"


@dataclass
class TfidfTextIndex:
    """TF-IDF encoder with cosine similarity over fixed documents."""

    doc_ids: list[str]
    documents: list[str]
    vectorizer: TfidfVectorizer
    matrix: object
    id_to_row: dict[str, int]

    @classmethod
    def fit(cls, doc_ids: Sequence[str], documents: Sequence[str]) -> "TfidfTextIndex":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
        )
        docs = [doc if doc.strip() else "empty" for doc in documents]
        matrix = vectorizer.fit_transform(docs)
        ids = list(doc_ids)
        return cls(ids, docs, vectorizer, matrix, {doc_id: idx for idx, doc_id in enumerate(ids)})

    def similarities(self, query: str) -> dict[str, float]:
        if not self.doc_ids:
            return {}
        q = self.vectorizer.transform([query if query.strip() else "empty"])
        values = cosine_similarity(q, self.matrix).ravel()
        return {doc_id: float(score) for doc_id, score in zip(self.doc_ids, values)}

    def score_subset(self, query: str, doc_ids: Sequence[str]) -> dict[str, float]:
        rows = [self.id_to_row[doc_id] for doc_id in doc_ids if doc_id in self.id_to_row]
        if not rows:
            return {doc_id: 0.0 for doc_id in doc_ids}
        q = self.vectorizer.transform([query if query.strip() else "empty"])
        values = cosine_similarity(q, self.matrix[rows]).ravel()
        row_scores = {self.doc_ids[row]: float(score) for row, score in zip(rows, values)}
        return {doc_id: row_scores.get(doc_id, 0.0) for doc_id in doc_ids}


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))
