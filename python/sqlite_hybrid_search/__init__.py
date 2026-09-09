"""sqlite_hybrid_search: an embeddable, local-first hybrid (dense + sparse)
retrieval and agent-memory engine.

This package is a thin Python layer over `_sqlite_hybrid_search_ext`, the
compiled nanobind extension module (bindings/python_bindings.cpp). The
native module is deliberately close to the C++ API; this layer adapts it
to a friendlier Python shape -- dict-based documents in, dict-based
results out -- and is what `import sqlite_hybrid_search` actually gives you.
Do not import `sqlite_hybrid_search._sqlite_hybrid_search_ext` directly.

Quickstart:

    import sqlite_hybrid_search

    engine = sqlite_hybrid_search.Engine("my_index.sqlite3", dim=4)
    engine.add(
        documents=[{"id": "doc-1", "text": "the quick brown fox"}],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
    )
    print(engine.search([1.0, 0.0, 0.0, 0.0], top_k=1))
"""

from __future__ import annotations

from typing import Any

from . import _sqlite_hybrid_search_ext as _ext

__all__ = ["Engine"]


def _chunk_result_to_dict(result: _ext.ChunkSearchResult) -> dict[str, Any]:
    return {
        "document_id": result.document_id,
        "chunk_index": result.chunk_index,
        "text": result.text,
        "score": result.score,
    }


def _explanation_to_dict(explanation: _ext.SearchExplanation) -> dict[str, Any]:
    return {
        "document_id": explanation.document_id,
        "chunk_index": explanation.chunk_index,
        "text": explanation.text,
        "dense_present": explanation.dense_present,
        "dense_distance": explanation.dense_distance,
        "dense_rank": explanation.dense_rank,
        "sparse_present": explanation.sparse_present,
        "sparse_bm25_score": explanation.sparse_bm25_score,
        "sparse_rank": explanation.sparse_rank,
        "fused_score": explanation.fused_score,
        "final_rank": explanation.final_rank,
        "created_at_unix_seconds": explanation.created_at_unix_seconds,
        "age_seconds": explanation.age_seconds,
        "recency_factor": explanation.recency_factor,
        "decayed_score": explanation.decayed_score,
    }


class Engine:
    """Friendly Python wrapper over the native (nanobind) RetrievalEngine.

    Adapts the C++ engine's structured DocumentInput/DocumentChunkInput
    ingestion API to a simpler (documents, embeddings) shape: each
    (document, embedding) pair becomes one document with a single chunk
    (the whole document text). For multi-chunk ingestion (long documents
    split via chunk_text()), build DocumentInput/DocumentChunkInput objects
    directly against the native module instead -- see cli.py's `ingest`
    command for an example. See docs/dev-log.md for why this mapping
    exists rather than a 1:1 mirror of the C++ API.
    """

    def __init__(self, db_path: str, dim: int) -> None:
        self._native = _ext.NativeEngine(db_path, dim)

    def add(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        """Adds `documents` (each a single chunk) with their `embeddings`.

        `documents[i]` must have an "id" key and may have "text" and
        "metadata" keys (both default to ""). `embeddings[i]` is that
        document's embedding. Raises ValueError if the two lists have
        different lengths.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                "documents and embeddings must have the same length "
                f"(got {len(documents)} documents and {len(embeddings)} embeddings)"
            )

        native_documents = []
        for document, embedding in zip(documents, embeddings):
            chunk = _ext.DocumentChunkInput()
            chunk.text = document.get("text", "")
            chunk.embedding = embedding
            chunk.start_token = 0
            chunk.end_token = len(chunk.text.split())

            native_document = _ext.DocumentInput()
            native_document.document_id = document["id"]
            native_document.metadata = document.get("metadata", "")
            native_document.chunks = [chunk]
            native_documents.append(native_document)

        self._native.add_documents(native_documents)

    def search(self, query_vec: list[float], top_k: int) -> list[dict[str, Any]]:
        """Dense-only search -- see RetrievalEngine::search_dense()."""
        return [_chunk_result_to_dict(r) for r in self._native.search_dense(query_vec, top_k)]

    def search_hybrid(self, query_text: str, query_vec: list[float], top_k: int) -> list[dict[str, Any]]:
        """Hybrid (dense + sparse, RRF-fused) search -- see RetrievalEngine::search_hybrid()."""
        return [_chunk_result_to_dict(r) for r in self._native.search_hybrid(query_text, query_vec, top_k)]

    def search_explained(self, query_text: str, query_vec: list[float], top_k: int) -> list[dict[str, Any]]:
        """Hybrid search with the full score breakdown -- see RetrievalEngine::search_explained()."""
        return [_explanation_to_dict(e) for e in self._native.search_explained(query_text, query_vec, top_k)]

    def search_memory(
        self, query_text: str, query_vec: list[float], top_k: int, decay_lambda: float
    ) -> list[dict[str, Any]]:
        """The agent memory layer: search_hybrid()'s fused score, discounted
        by exponential recency decay and re-ranked by the result -- see
        RetrievalEngine::search_memory()'s doc comment for the formula.
        `decay_lambda=0` disables decay entirely (identical ranking to
        search_hybrid()).
        """
        return [
            _chunk_result_to_dict(r) for r in self._native.search_memory(query_text, query_vec, top_k, decay_lambda)
        ]

    def search_memory_explained(
        self, query_text: str, query_vec: list[float], top_k: int, decay_lambda: float
    ) -> list[dict[str, Any]]:
        """Like search_memory(), but with the full score breakdown (as
        search_explained(), plus created_at_unix_seconds, age_seconds,
        recency_factor, decayed_score) -- see
        RetrievalEngine::search_memory_explained()'s doc comment.
        """
        return [
            _explanation_to_dict(e)
            for e in self._native.search_memory_explained(query_text, query_vec, top_k, decay_lambda)
        ]

    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""
        return self._native.chunk_count()

    def loaded_index_from_sidecar(self) -> bool:
        """True if this engine loaded its dense index from the on-disk
        ``<db_path>.usearch`` sidecar at open (the fast path), False if it
        rebuilt from SQLite. Diagnostic only -- results are identical
        either way, and the sidecar is written and refreshed automatically.
        """
        return self._native.loaded_index_from_sidecar()

    # --- Built-in local embedding model --------------------------------

    def load_embedding_model(self, model_path: str) -> None:
        """Loads a local embedding model from `model_path` and attaches it,
        enabling embed(), add_text() and search_text(). The model's output
        dimension must equal the `dim` this Engine was constructed with.
        Raises RuntimeError if the file is missing/unreadable/unrecognized,
        ValueError on a dimension mismatch.
        """
        self._native.load_embedding_model(model_path)

    def has_embedding_model(self) -> bool:
        """True once load_embedding_model() has attached a model."""
        return self._native.has_embedding_model()

    def embedding_dim(self) -> int:
        """Output dimension of the attached model. Raises if none attached."""
        return self._native.embedding_dim()

    def embed(self, text: str) -> list[float]:
        """Embeds `text` with the attached model -- see
        RetrievalEngine::embed(). Raises if no model is attached.
        """
        return list(self._native.embed(text))

    def add_text(self, documents: list[dict[str, Any]]) -> None:
        """Raw-text ingestion: embeds each document's text with the attached
        model (no caller-supplied vectors). `documents[i]` must have an
        "id" key and may have "text", "metadata" (default "") and
        "created_at_unix_seconds" (default 0) keys. Raises if no model is
        attached.
        """
        native_documents = []
        for document in documents:
            native_document = _ext.TextDocumentInput()
            native_document.document_id = document["id"]
            native_document.text = document.get("text", "")
            native_document.metadata = document.get("metadata", "")
            native_document.created_at_unix_seconds = document.get("created_at_unix_seconds", 0)
            native_documents.append(native_document)
        self._native.add_text(native_documents)

    def search_text(self, query_text: str, top_k: int, decay_lambda: float = 0.0) -> list[dict[str, Any]]:
        """Raw-text query: embeds `query_text` with the attached model and
        runs the memory search with it -- see RetrievalEngine::search_text().
        `decay_lambda=0` (the default) disables recency decay. Raises if no
        model is attached.
        """
        return [_chunk_result_to_dict(r) for r in self._native.search_text(query_text, top_k, decay_lambda)]
