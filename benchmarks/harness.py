"""Benchmark harness: measure one dataset size end to end.

`benchmark_size()` ingests a synthetic labelled corpus (benchmarks/corpus.py)
into a fresh on-disk RetrievalEngine, then for each retrieval approach
  - dense        : search_dense()        -- cosine ANN only
  - sparse       : search_sparse()       -- FTS5 BM25 only
  - hybrid       : search_hybrid()       -- RRF fusion of the two
  - hybrid_decay : search_memory()       -- hybrid + exponential recency decay
measures retrieval quality (Recall@10, nDCG@10, MRR@10), warm-cache latency
(p50/p95/p99, mean, throughput), cold-cache first-query latency, and the
process memory / on-disk footprint.

Ingestion goes through the native _retrieval_engine_ext types rather than
the friendly Engine.add() wrapper because the decay approach needs per-chunk
timestamps, which the wrapper does not expose (see docs/DECISIONS.md).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from typing import Callable

import retrieval_engine  # noqa: F401  -- ensures the extension is importable
from corpus import LabeledCorpus, build_corpus
from metrics import mrr_at_k, ndcg_at_k, recall_at_k
from retrieval_engine import _retrieval_engine_ext as _ext

APPROACHES = ("dense", "sparse", "hybrid", "hybrid_decay")
K = 10
_DISTRACTOR_AGE_SECONDS = 30 * 86_400  # distractors are "old", relevant docs are "now"


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = pct / 100.0 * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def _peak_rss_mb() -> float:
    """Process-wide peak resident set size. ru_maxrss is bytes on macOS,
    kibibytes on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _current_rss_mb() -> float:
    """Current RSS via `ps` (portable across macOS/Linux, KiB)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip()) / 1024
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def _ingest(engine_native, corpus: LabeledCorpus, now: int, batch: int = 5_000) -> None:
    relevant = {cid for query in corpus.queries for cid in query["relevant_ids"]}
    pending: list = []
    for doc, embedding in zip(corpus.documents, corpus.embeddings):
        chunk = _ext.DocumentChunkInput()
        chunk.text = doc["text"]
        chunk.embedding = embedding
        chunk.start_token = 0
        chunk.end_token = len(doc["text"].split())
        chunk.created_at_unix_seconds = now if doc["id"] in relevant else now - _DISTRACTOR_AGE_SECONDS

        native_doc = _ext.DocumentInput()
        native_doc.document_id = doc["id"]
        native_doc.metadata = ""
        native_doc.chunks = [chunk]
        pending.append(native_doc)
        if len(pending) >= batch:
            engine_native.add_documents(pending)
            pending.clear()
    if pending:
        engine_native.add_documents(pending)


def _approach_runner(engine_native, approach: str, decay_lambda: float) -> Callable[[dict], list[str]]:
    if approach == "dense":
        return lambda q: [r.document_id for r in engine_native.search_dense(q["embedding"], K)]
    if approach == "sparse":
        return lambda q: [r.document_id for r in engine_native.search_sparse(q["text"], K)]
    if approach == "hybrid":
        return lambda q: [r.document_id for r in engine_native.search_hybrid(q["text"], q["embedding"], K)]
    if approach == "hybrid_decay":
        return lambda q: [
            r.document_id for r in engine_native.search_memory(q["text"], q["embedding"], K, decay_lambda)
        ]
    raise ValueError(f"unknown approach {approach!r}")


def _quality(runner, queries: list[dict]) -> dict:
    recall = ndcg = mrr = 0.0
    for query in queries:
        retrieved = runner(query)
        recall += recall_at_k(retrieved, query["relevant_ids"], K)
        ndcg += ndcg_at_k(retrieved, query["relevant_ids"], K)
        mrr += mrr_at_k(retrieved, query["relevant_ids"], K)
    n = len(queries)
    return {"recall_at_10": recall / n, "ndcg_at_10": ndcg / n, "mrr_at_10": mrr / n}


def _warm_latency(runner, queries: list[dict], min_samples: int = 200, max_passes: int = 20) -> dict:
    for query in queries:  # warm-up pass, discarded
        runner(query)

    samples_ms: list[float] = []
    passes = 0
    while len(samples_ms) < min_samples and passes < max_passes:
        for query in queries:
            start = time.perf_counter()
            runner(query)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
        passes += 1

    samples_ms.sort()
    total_s = sum(samples_ms) / 1000.0
    return {
        "latency_p50_ms": _percentile(samples_ms, 50),
        "latency_p95_ms": _percentile(samples_ms, 95),
        "latency_p99_ms": _percentile(samples_ms, 99),
        "latency_mean_ms": sum(samples_ms) / len(samples_ms),
        "throughput_qps": len(samples_ms) / total_s if total_s > 0 else 0.0,
    }


def _cold_latency(db_path: str, dim: int, approach: str, decay_lambda: float, query: dict) -> dict:
    """First-query latency against a freshly reopened engine (usearch index
    rebuilt from SQLite, prepared statements cold)."""
    start = time.perf_counter()
    reopened = _ext.NativeEngine(db_path, dim)
    reopen_ms = (time.perf_counter() - start) * 1000.0

    runner = _approach_runner(reopened, approach, decay_lambda)
    start = time.perf_counter()
    runner(query)
    return {"reopen_ms": reopen_ms, "cold_first_query_ms": (time.perf_counter() - start) * 1000.0}


def benchmark_size(dataset_size: int, num_queries: int, dim: int, decay_lambda: float, tmp_dir: str) -> list[dict]:
    corpus = build_corpus(dataset_size, num_queries, dim=dim)
    db_path = os.path.join(tmp_dir, f"bench_{dataset_size}.sqlite3")
    if os.path.exists(db_path):
        os.remove(db_path)

    now = int(time.time())
    rss_before = _current_rss_mb()
    engine_native = _ext.NativeEngine(db_path, dim)

    ingest_start = time.perf_counter()
    _ingest(engine_native, corpus, now)
    ingest_s = time.perf_counter() - ingest_start

    rss_delta_index_mb = max(0.0, _current_rss_mb() - rss_before)
    db_size_mb = os.path.getsize(db_path) / 1_000_000
    docs_per_sec = len(corpus.documents) / ingest_s if ingest_s > 0 else 0.0

    records: list[dict] = []
    for approach in APPROACHES:
        runner = _approach_runner(engine_native, approach, decay_lambda)
        record = {
            "dataset_size": dataset_size,
            "approach": approach,
            "num_queries": len(corpus.queries),
            "index_docs_per_sec": docs_per_sec,
            "db_size_mb": db_size_mb,
            "rss_delta_index_mb": rss_delta_index_mb,
        }
        record.update(_quality(runner, corpus.queries))
        record.update(_warm_latency(runner, corpus.queries))
        record.update(_cold_latency(db_path, dim, approach, decay_lambda, corpus.queries[0]))
        records.append(record)

    peak = _peak_rss_mb()
    for record in records:
        record["peak_memory_mb"] = peak
    return records
