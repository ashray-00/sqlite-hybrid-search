"""Stage 3 (BUILD_PLAN.md) -- TDD RED phase.

Tests two not-yet-implemented pieces of Stage 3:

  1. Bindings.* -- `import retrieval_engine`, the nanobind extension module
     wrapping retrieval_engine::RetrievalEngine
     (core/include/retrieval_engine/retrieval_engine.hpp). No bindings/
     directory, no pyproject.toml/setup.py, and nothing installed into
     .venv yet -- `import retrieval_engine` is expected to fail with
     ModuleNotFoundError.
  2. Cli.* -- the `engine` console-script CLI (`engine ingest`,
     `engine query`). No CLI code or packaging exists yet, so
     .venv/bin/engine doesn't exist -- invoking it via subprocess is
     expected to fail (FileNotFoundError, or a non-zero/absent exit code).

Do not "fix" either by writing the bindings/CLI/packaging -- that is
Phase 2 (GREEN). This file's job right now is only to fail for the right
reason.

Python-facing Engine API this file specifies -- resolved by re-reading
core/include/retrieval_engine/retrieval_engine.hpp before writing this (see
docs/DECISIONS.md for the full mapping from what was asked to what the C++
engine actually exposes):

    Engine(db_path: str, dim: int)
        Mirrors RetrievalEngine(db_path, dim) exactly. No index_path: the
        core engine has no file-based index persistence, only SQLite +
        rebuild-on-open (BUILD_PLAN.md section 5).
    engine.add(documents, embeddings)
        documents: list of {"id": str, "text": str, "metadata": str}.
        embeddings: a parallel list of list[float], one per document.
        Maps each (document, embedding) pair onto one DocumentInput with a
        single DocumentChunkInput (the whole document as one chunk) --
        add_documents() functionality, exposed at a simpler granularity.
        (Multi-chunk documents are chunk_text()'s own territory, already
        covered by Stage 1's tests -- not what this smoke test is for.)
    engine.search(query_vec, top_k) -> search_dense()
    engine.search_hybrid(query_text, query_vec, top_k) -> search_hybrid()
        (name and shape already match the C++ method exactly)
    engine.search_explained(query_text, query_vec, top_k) -> search_explained()
        (name and shape already match the C++ method exactly)
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_CLI = REPO_ROOT / ".venv" / "bin" / "engine"


def _sample_documents_and_embeddings():
    documents = [
        {"id": "doc-0", "text": "the quick brown fox", "metadata": "source:test"},
        {"id": "doc-1", "text": "a completely unrelated sentence", "metadata": "source:test"},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    return documents, embeddings


def test_import_retrieval_engine_bindings():
    import retrieval_engine  # noqa: F401 -- the import itself is the assertion


def test_engine_add_and_search(tmp_path):
    import retrieval_engine

    db_path = str(tmp_path / "stage3_test.sqlite3")
    engine = retrieval_engine.Engine(db_path, 4)

    documents, embeddings = _sample_documents_and_embeddings()
    engine.add(documents, embeddings)

    # doc-0's embedding is an exact match for the query; doc-1's is
    # orthogonal, so this is unambiguous.
    dense_results = engine.search([1.0, 0.0, 0.0, 0.0], 1)
    assert len(dense_results) == 1
    assert dense_results[0]["document_id"] == "doc-0"


def test_engine_search_hybrid_and_explained(tmp_path):
    import retrieval_engine

    db_path = str(tmp_path / "stage3_test_hybrid.sqlite3")
    engine = retrieval_engine.Engine(db_path, 4)

    documents, embeddings = _sample_documents_and_embeddings()
    engine.add(documents, embeddings)

    hybrid_results = engine.search_hybrid("fox", [1.0, 0.0, 0.0, 0.0], 2)
    assert len(hybrid_results) >= 1
    assert hybrid_results[0]["document_id"] == "doc-0"

    explanations = engine.search_explained("fox", [1.0, 0.0, 0.0, 0.0], 2)
    assert len(explanations) >= 1
    explanation = explanations[0]
    for key in (
        "document_id",
        "dense_present",
        "dense_rank",
        "sparse_present",
        "sparse_rank",
        "fused_score",
        "final_rank",
    ):
        assert key in explanation


def test_cli_ingest_then_query(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text(
        "the quick brown fox jumps over the lazy dog", encoding="utf-8"
    )

    ingest = subprocess.run(
        [str(ENGINE_CLI), "ingest", str(docs_dir)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert ingest.returncode == 0, f"ingest failed: {ingest.stderr}"
    assert ingest.stdout.strip() != ""

    query = subprocess.run(
        [str(ENGINE_CLI), "query", "fox"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert query.returncode == 0, f"query failed: {query.stderr}"
    assert query.stdout.strip() != ""
