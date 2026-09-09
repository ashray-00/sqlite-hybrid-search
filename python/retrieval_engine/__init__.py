"""retrieval_engine: an embeddable, local-first hybrid (dense + sparse)
retrieval engine (BUILD_PLAN.md).

This package is a thin Python layer over `_retrieval_engine_ext`, the
compiled nanobind extension module (bindings/python_bindings.cpp). The
native module is deliberately close to the C++ API; this layer adapts it
to a friendlier Python shape -- dict-based documents in, dict-based
results out -- and is what `import retrieval_engine` actually gives you.
Do not import `retrieval_engine._retrieval_engine_ext` directly.

Quickstart:

    import retrieval_engine

    engine = retrieval_engine.Engine("my_index.sqlite3", dim=4)
    engine.add(
        documents=[{"id": "doc-1", "text": "the quick brown fox"}],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
    )
    print(engine.search([1.0, 0.0, 0.0, 0.0], top_k=1))
"""

from __future__ import annotations

from typing import Any

from . import _retrieval_engine_ext as _ext

__all__ = ["Engine"]


def _chunk_result_to_dict(result: "_ext.ChunkSearchResult") -> dict[str, Any]:
    return {
        "document_id": result.document_id,
        "chunk_index": result.chunk_index,
        "text": result.text,
        "score": result.score,
    }


def _explanation_to_dict(explanation: "_ext.SearchExplanation") -> dict[str, Any]:
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
    }


class Engine:
    """Friendly Python wrapper over the native (nanobind) RetrievalEngine.

    Adapts the C++ engine's structured DocumentInput/DocumentChunkInput
    ingestion API to a simpler (documents, embeddings) shape: each
    (document, embedding) pair becomes one document with a single chunk
    (the whole document text). For multi-chunk ingestion (long documents
    split via chunk_text()), build DocumentInput/DocumentChunkInput objects
    directly against the native module instead -- see cli.py's `ingest`
    command for an example. See docs/DECISIONS.md ("Stage 3") for why this
    mapping exists rather than a 1:1 mirror of the C++ API.
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

    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""
        return self._native.chunk_count()
