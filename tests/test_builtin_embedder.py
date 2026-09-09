"""Built-in local embedding inference ("just give it text"), from Python.

Tests four methods that do not exist on the Python Engine wrapper yet
(python/retrieval_engine/__init__.py):

    engine.load_embedding_model(model_path)
    engine.embed(text) -> list[float]
    engine.add_text(documents)          # documents: [{"id", "text", "metadata"}]
    engine.search_text(query_text, top_k)

Calling any of them is expected to make the test FAIL with an uncaught
AttributeError -- that is the correct, expected result for this pass
(Phase 1, RED). Do NOT fix it by implementing the Python-side API or the
native embedder; that is Phase 2 (GREEN).

The detailed embedding behavior (exact dimension, semantic-similarity
ordering) is covered more thoroughly in the C++ suite
(core/tests/test_builtin_embedder.cpp). This file confirms the same capability is
reachable from the Python API once it exists.

Model file: these tests write a tiny *mock* model file rather than
downloading a real ONNX/GGUF model -- see WriteMockModel() below and
docs/DECISIONS.md for the documented contract GREEN is expected to honor.
"""

import math
import subprocess
from pathlib import Path

import pytest

import retrieval_engine

_EMBEDDING_DIM = 384

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINE_CLI = _REPO_ROOT / ".venv" / "bin" / "engine"


def _write_mock_model(path) -> str:
    """Writes the mock embedding-model file (see module docstring / DECISIONS.md).

    First line selects a deterministic, dependency-free test embedder in
    place of a real ONNX/GGUF runtime.
    """
    path.write_text(
        f"RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1\ndim={_EMBEDDING_DIM}\n",
        encoding="utf-8",
    )
    return str(path)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _engine_with_mock_model(tmp_path, db_filename):
    model_path = _write_mock_model(tmp_path / "mock_model.txt")
    engine = retrieval_engine.Engine(str(tmp_path / db_filename), dim=_EMBEDDING_DIM)
    engine.load_embedding_model(model_path)
    return engine


def test_embed_returns_non_empty_vector_of_model_dimension(tmp_path):
    engine = _engine_with_mock_model(tmp_path, "embed.sqlite3")

    embedding = engine.embed("Hello world")

    assert len(embedding) == _EMBEDDING_DIM
    assert any(value != 0.0 for value in embedding)


def test_add_text_and_search_text_without_precomputed_vectors(tmp_path):
    engine = _engine_with_mock_model(tmp_path, "search_text.sqlite3")

    engine.add_text(
        [
            {"id": "install-guide", "text": "python package installation guide for beginners"},
            {"id": "weather-report", "text": "mountain weather forecast cold and snowy today"},
            {"id": "cooking-notes", "text": "recipe roasted vegetables garlic thyme dinner"},
        ]
    )
    assert engine.chunk_count() == 3

    results = engine.search_text("the python package installation guide", top_k=2)

    assert results
    assert results[0]["document_id"] == "install-guide"


def test_semantically_related_text_scores_higher(tmp_path):
    engine = _engine_with_mock_model(tmp_path, "cosine.sqlite3")

    query = engine.embed("the python package installation guide")
    related = engine.embed("python package installation guide for beginners")
    unrelated = engine.embed("mountain weather forecast cold and snowy")

    assert _cosine(query, related) > _cosine(query, unrelated)
    assert _cosine(query, related) > 0.5


def test_cli_routes_text_through_the_builtin_embedder(tmp_path):
    """`engine ingest/query --model <m> --model-dim <n>` generates vectors
    under the hood -- no caller-supplied floats at the CLI at all.
    """
    model_path = _write_mock_model(tmp_path / "mock_model.txt")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "install.txt").write_text(
        "python package installation guide for beginners", encoding="utf-8"
    )
    (docs_dir / "weather.txt").write_text(
        "mountain weather forecast cold and snowy today", encoding="utf-8"
    )

    model_flags = ["--model", model_path, "--model-dim", str(_EMBEDDING_DIM)]

    ingest = subprocess.run(
        [str(_ENGINE_CLI), "ingest", str(docs_dir), *model_flags],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert ingest.returncode == 0, f"ingest failed: {ingest.stderr}"

    query = subprocess.run(
        [str(_ENGINE_CLI), "query", "the python package installation guide", *model_flags],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert query.returncode == 0, f"query failed: {query.stderr}"
    assert "install.txt" in query.stdout


def test_load_embedding_model_with_missing_file_raises(tmp_path):
    engine = retrieval_engine.Engine(str(tmp_path / "missing.sqlite3"), dim=_EMBEDDING_DIM)

    with pytest.raises(RuntimeError):
        engine.load_embedding_model(str(tmp_path / "does_not_exist.bin"))
    assert not engine.has_embedding_model()


def test_load_embedding_model_with_wrong_dimension_raises(tmp_path):
    # Mock model advertises a different dimension than the engine was built for.
    model_path = tmp_path / "wrong_dim_model.txt"
    model_path.write_text(
        f"RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1\ndim={_EMBEDDING_DIM + 1}\n", encoding="utf-8"
    )
    engine = retrieval_engine.Engine(str(tmp_path / "wrong_dim.sqlite3"), dim=_EMBEDDING_DIM)

    with pytest.raises(ValueError):
        engine.load_embedding_model(str(model_path))
    assert not engine.has_embedding_model()


def test_cli_model_flag_requires_model_dim(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("hello world", encoding="utf-8")

    result = subprocess.run(
        [str(_ENGINE_CLI), "ingest", str(docs_dir), "--model", "whatever.bin"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "--model-dim" in result.stderr
