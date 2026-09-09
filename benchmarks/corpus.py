"""Synthetic labelled retrieval corpus for the benchmark harness.

Reproducible (seeded) single-chunk documents plus eval queries carrying
their known-relevant document ids, shaped so the retrieval styles under
test have genuinely different strengths instead of all collapsing to the
same ranking:

  * Every relevant doc for a query contains one rare entity token
    ("entity<NNNNN>") that appears nowhere else -> BM25 gets a very high
    IDF signal, i.e. exact-term territory.
  * Relevant docs also share a short topic-word phrase with the query ->
    dense (bag-of-words) similarity has something to latch onto.
  * Distractor docs are drawn from the same vocabulary but never contain an
    entity token, so they are near-misses rather than random noise.

The vocabulary is a Zipf-distributed pool of pseudo-words: a few words are
frequent and most are rare, as in real text. Query topic words come from a
mid-frequency band so each one matches only a small slice of the corpus --
without that, the BM25 arm rescans most of the corpus on every query and
its latency reflects the toy vocabulary rather than the engine.

Embeddings are a deterministic hashed bag-of-words (the "hashing trick"),
L2-normalised -- enough structure for cosine ANN to be meaningful without
pulling in a real embedding model.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass

_VOCAB_SIZE = 800
_VOCAB = [f"w{i:04d}" for i in range(_VOCAB_SIZE)]
# Zipf-ish weights: flat enough (exponent 0.5) that even the most frequent
# word lands in well under a third of documents at the doc lengths below.
_WEIGHTS = [1.0 / (i + 1) ** 0.5 for i in range(_VOCAB_SIZE)]
# Query topic words are sampled from this rank band -- frequent enough that
# dense similarity has signal, rare enough (~1% of docs each) that the BM25
# arm stays selective. The most frequent word overall lands in ~20% of docs.
_TOPIC_BAND = (200, 560)

_DISTRACTOR_LEN = 12
_RELEVANT_FILLER_LEN = 6
_TOPIC_WORDS = 4


@dataclass
class LabeledCorpus:
    documents: list[dict]  # each {"id": str, "text": str}
    embeddings: list[list[float]]  # parallel to documents
    queries: list[dict]  # each {"text": str, "embedding": [...], "relevant_ids": set[str]}
    dim: int


def embed_text(text: str, dim: int) -> list[float]:
    """Deterministic L2-normalised hashed bag-of-words vector."""
    vec = [0.0] * dim
    for token in text.lower().split():
        vec[zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def build_corpus(
    num_docs: int,
    num_queries: int,
    *,
    dim: int = 64,
    relevant_per_query: int = 3,
    seed: int = 1234,
) -> LabeledCorpus:
    """Builds `num_docs` documents; the first groups are the labelled
    relevant sets for up to `num_queries` queries, the rest are distractors.
    `num_queries` is clamped so the relevant sets fit inside `num_docs`.
    """
    if num_docs < relevant_per_query + 1:
        raise ValueError(f"num_docs={num_docs} too small for relevant_per_query={relevant_per_query}")

    rng = random.Random(seed)
    num_queries = max(1, min(num_queries, num_docs // (relevant_per_query + 1)))
    topic_pool = _VOCAB[_TOPIC_BAND[0] : _TOPIC_BAND[1]]

    documents: list[dict] = []
    embeddings: list[list[float]] = []
    queries: list[dict] = []

    def add(doc_id: str, text: str) -> None:
        documents.append({"id": doc_id, "text": text})
        embeddings.append(embed_text(text, dim))

    for q in range(num_queries):
        entity = f"entity{q:05d}"
        topic_words = rng.sample(topic_pool, _TOPIC_WORDS)
        relevant_ids: list[str] = []
        for _ in range(relevant_per_query):
            filler = rng.choices(_VOCAB, weights=_WEIGHTS, k=_RELEVANT_FILLER_LEN)
            add(f"doc{len(documents):06d}", " ".join([entity, *topic_words, *filler]))
            relevant_ids.append(documents[-1]["id"])
        query_text = " ".join([entity, *topic_words])
        queries.append(
            {
                "text": query_text,
                "embedding": embed_text(query_text, dim),
                "relevant_ids": set(relevant_ids),
            }
        )

    while len(documents) < num_docs:
        text = " ".join(rng.choices(_VOCAB, weights=_WEIGHTS, k=_DISTRACTOR_LEN))
        add(f"doc{len(documents):06d}", text)

    return LabeledCorpus(documents=documents, embeddings=embeddings, queries=queries, dim=dim)
