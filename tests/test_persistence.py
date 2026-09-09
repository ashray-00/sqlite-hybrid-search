"""USearch sidecar persistence, from the Python API and the CLI.

The C++ suite (core/tests/test_persistence.cpp) covers the load/rebuild/
stale/corrupt matrix in detail. This is an integration-level check that the
sidecar is created and reloaded automatically through the bindings and the
`hybrid-search` console script.
"""

import subprocess
from pathlib import Path

import sqlite_hybrid_search

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_CLI = REPO_ROOT / ".venv" / "bin" / "hybrid-search"

_DIM = 4


def test_reopening_engine_loads_sidecar(tmp_path):
    db_path = str(tmp_path / "persist.sqlite3")

    engine = sqlite_hybrid_search.Engine(db_path, dim=_DIM)
    engine.add(
        documents=[{"id": f"doc-{i}", "text": f"document {i}"} for i in range(6)],
        embeddings=[[(1.0 + i) if j == i % _DIM else 0.1 for j in range(_DIM)] for i in range(6)],
    )
    assert not engine.loaded_index_from_sidecar()  # fresh build
    assert Path(db_path + ".usearch").exists()

    reopened = sqlite_hybrid_search.Engine(db_path, dim=_DIM)
    assert reopened.loaded_index_from_sidecar()
    assert reopened.chunk_count() == 6
    assert reopened.search([1.0, 0.1, 0.1, 0.1], top_k=1)[0]["document_id"] == "doc-0"


def test_cli_ingest_creates_sidecar_and_query_reuses_it(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "note.txt").write_text("I live in Munich near the river", encoding="utf-8")

    ingest = subprocess.run([str(ENGINE_CLI), "ingest", str(docs_dir)], capture_output=True, text=True, cwd=tmp_path)
    assert ingest.returncode == 0, ingest.stderr
    assert (tmp_path / ".hybrid_search.sqlite3.usearch").exists()

    query = subprocess.run([str(ENGINE_CLI), "query", "where do I live"], capture_output=True, text=True, cwd=tmp_path)
    assert query.returncode == 0, query.stderr
    assert "note.txt" in query.stdout
