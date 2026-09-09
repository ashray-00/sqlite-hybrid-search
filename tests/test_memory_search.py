"""Agent memory: recency decay + temporal reranking, from Python.

The detailed decay-math and reranking behavior (the exact formula,
demonstrating that decay actually reorders results) is verified thoroughly
in the C++ test suite (core/tests/test_recency_decay_and_temporal_reranking.cpp),
which has precise control over chunk timestamps. This file is a lighter
integration-level check that the same capability is reachable from the
Python API and the CLI.
"""

import subprocess
import time
from pathlib import Path

import retrieval_engine
from retrieval_engine import _retrieval_engine_ext as _ext

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_CLI = REPO_ROOT / ".venv" / "bin" / "engine"

_DIM = 4
_ONE_DAY_SECONDS = 86400


def _add_two_chunk_corpus_with_explicit_timestamps(engine: retrieval_engine.Engine) -> None:
    """Ingests the same "old" (higher raw similarity, one day old) vs.
    "recent" (slightly lower raw similarity, brand new) corpus the C++ test
    uses -- built directly against the native module, since the friendly
    Engine.add() wrapper doesn't expose per-chunk timestamps (see
    docs/DECISIONS.md).
    """
    now = int(time.time())

    old_chunk = _ext.DocumentChunkInput()
    old_chunk.text = "an old memory"
    old_chunk.embedding = [1.0, 0.0, 0.0, 0.0]
    old_chunk.start_token = 0
    old_chunk.end_token = 3
    old_chunk.created_at_unix_seconds = now - _ONE_DAY_SECONDS

    old_document = _ext.DocumentInput()
    old_document.document_id = "old"
    old_document.metadata = ""
    old_document.chunks = [old_chunk]

    recent_chunk = _ext.DocumentChunkInput()
    recent_chunk.text = "a recent memory"
    recent_chunk.embedding = [0.999, 0.001, 0.0, 0.0]
    recent_chunk.start_token = 0
    recent_chunk.end_token = 3
    recent_chunk.created_at_unix_seconds = now

    recent_document = _ext.DocumentInput()
    recent_document.document_id = "recent"
    recent_document.metadata = ""
    recent_document.chunks = [recent_chunk]

    engine._native.add_documents([old_document, recent_document])


def test_search_memory_ranks_recent_document_above_slightly_more_similar_older_one(tmp_path):
    engine = retrieval_engine.Engine(str(tmp_path / "memory_test.sqlite3"), dim=_DIM)
    _add_two_chunk_corpus_with_explicit_timestamps(engine)

    query_vec = [1.0, 0.0, 0.0, 0.0]

    # decay_lambda=0 disables decay: raw similarity wins, "old" ranks first.
    undecayed = engine.search_memory("unrelated", query_vec, top_k=2, decay_lambda=0.0)
    assert undecayed[0]["document_id"] == "old"

    # A non-zero decay_lambda lets the fresher document overtake it.
    decayed = engine.search_memory("unrelated", query_vec, top_k=2, decay_lambda=0.1)
    assert decayed[0]["document_id"] == "recent"


def test_search_memory_explained_includes_recency_fields(tmp_path):
    engine = retrieval_engine.Engine(str(tmp_path / "memory_test_explained.sqlite3"), dim=_DIM)
    _add_two_chunk_corpus_with_explicit_timestamps(engine)

    explanations = engine.search_memory_explained("unrelated", [1.0, 0.0, 0.0, 0.0], top_k=2, decay_lambda=0.1)
    assert len(explanations) == 2

    for key in ("created_at_unix_seconds", "age_seconds", "recency_factor", "decayed_score"):
        assert key in explanations[0]

    old_explanation = next(e for e in explanations if e["document_id"] == "old")
    recent_explanation = next(e for e in explanations if e["document_id"] == "recent")

    assert old_explanation["fused_score"] > recent_explanation["fused_score"]  # raw similarity favors "old"
    assert old_explanation["decayed_score"] < recent_explanation["decayed_score"]  # decay flips it
    assert old_explanation["age_seconds"] == pytest_approx(_ONE_DAY_SECONDS, abs=2.0)
    assert recent_explanation["age_seconds"] == pytest_approx(0.0, abs=2.0)


def pytest_approx(expected, abs):  # noqa: A002 -- matches pytest.approx's own kwarg name
    import pytest

    return pytest.approx(expected, abs=abs)


def test_cli_query_supports_decay_flag(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("the quick brown fox", encoding="utf-8")

    ingest = subprocess.run(
        [str(ENGINE_CLI), "ingest", str(docs_dir)], capture_output=True, text=True, cwd=tmp_path
    )
    assert ingest.returncode == 0, f"ingest failed: {ingest.stderr}"

    query = subprocess.run(
        [str(ENGINE_CLI), "query", "fox", "--decay", "0.05"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert query.returncode == 0, f"query failed: {query.stderr}"
    assert query.stdout.strip() != ""
