"""Agent memory: recency decay + temporal reranking, from Python.

Tests engine.search_memory(query_text, query_vec, top_k, recency_weight)
and engine.search_memory_explained(...), neither of which exists on the
Python Engine wrapper yet (python/retrieval_engine/__init__.py). Calling
either is expected to make the test FAIL with an uncaught AttributeError --
that is the correct, expected result for this pass. Do not "fix" it by
implementing the Python-side API; that is Phase 2 (GREEN).

The detailed decay-math and reranking behavior (the exact recency formula,
demonstrating that decay actually reorders results) is verified in the C++
test suite (core/tests/), which has precise control over chunk timestamps.
This file's job is only to confirm the same capability is reachable from
the Python API once it exists.
"""

import retrieval_engine


def _make_engine_with_one_document(tmp_path, db_filename):
    db_path = str(tmp_path / db_filename)
    engine = retrieval_engine.Engine(db_path, dim=4)
    engine.add(
        documents=[{"id": "doc-0", "text": "the quick brown fox"}],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
    )
    return engine


def test_engine_search_memory_ranks_recent_document_above_older_one(tmp_path):
    engine = _make_engine_with_one_document(tmp_path, "memory_test.sqlite3")

    results = engine.search_memory("fox", [1.0, 0.0, 0.0, 0.0], top_k=1, recency_weight=0.1)
    assert results


def test_engine_search_memory_explained_includes_recency_fields(tmp_path):
    engine = _make_engine_with_one_document(tmp_path, "memory_test_explained.sqlite3")

    explanations = engine.search_memory_explained("fox", [1.0, 0.0, 0.0, 0.0], top_k=1, recency_weight=0.1)
    assert explanations
    explanation = explanations[0]
    for key in ("timestamp", "age_seconds", "recency_factor", "decayed_score"):
        assert key in explanation
