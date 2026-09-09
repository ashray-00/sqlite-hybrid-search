"""Command-line interface for retrieval_engine (Stage 3, BUILD_PLAN.md):

    engine ingest <folder_path>   Chunk and index every .txt file in a folder.
    engine query "<text query>"   Run a hybrid (dense + sparse) search.

Both commands operate on an index file in the *current working directory*
(see _DB_FILENAME) -- `ingest` creates/updates it, `query` reads it. Run
`query` from the same directory you ran `ingest` from.

Stage 3's scope (BUILD_PLAN.md) is packaging/bindings/CLI, not embeddings:
"embeddings supplied by caller for now" (BUILD_PLAN.md Stage 1) is still in
effect, and no embedding model is wired in yet (that's Stage 5). Both
commands use a small deterministic hashing-trick "embedding" (_hash_embed
below) purely so the CLI has *something* to feed the dense index end to
end; it is not a real semantic embedding, and search quality will improve
once a later stage adds a real model.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from . import _retrieval_engine_ext as _ext

_DEFAULT_DIM = 64
_CHUNK_WINDOW_TOKENS = 200
_CHUNK_OVERLAP_TOKENS = 40
_DB_FILENAME = ".retrieval_engine.sqlite3"


def _hash_embed(text: str, dim: int = _DEFAULT_DIM) -> list[float]:
    """A deterministic, dependency-free stand-in for a real embedding model
    (feature hashing / the "hashing trick"): each token votes for one
    dimension based on a hash of itself. The same text always maps to the
    same vector, and texts sharing words with a query get *some*
    dense-similarity signal -- enough to exercise the ingest -> index ->
    query pipeline end to end, though far weaker than a real embedding
    model (see this module's docstring).
    """
    vector = [0.0] * dim
    for token in text.split():
        bucket = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16) % dim
        vector[bucket] += 1.0
    return vector


def _cmd_ingest(args: argparse.Namespace) -> int:
    folder = Path(args.folder_path)
    if not folder.exists():
        print(f"engine ingest: error: {folder} does not exist", file=sys.stderr)
        return 1
    if not folder.is_dir():
        print(f"engine ingest: error: {folder} is not a directory", file=sys.stderr)
        return 1

    text_files = sorted(folder.glob("*.txt"))
    if not text_files:
        print(f"engine ingest: error: no .txt files found in {folder}", file=sys.stderr)
        return 1

    documents = []
    total_chunks = 0
    for file_path in text_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"engine ingest: error: could not read {file_path}: {error}", file=sys.stderr)
            return 1

        document = _ext.DocumentInput()
        document.document_id = file_path.name
        document.metadata = str(file_path)

        chunks = []
        for chunk in _ext.chunk_text(text, _CHUNK_WINDOW_TOKENS, _CHUNK_OVERLAP_TOKENS):
            native_chunk = _ext.DocumentChunkInput()
            native_chunk.text = chunk.text
            native_chunk.embedding = _hash_embed(chunk.text)
            native_chunk.start_token = chunk.start_token
            native_chunk.end_token = chunk.end_token
            chunks.append(native_chunk)
        document.chunks = chunks

        total_chunks += len(chunks)
        documents.append(document)

    db_path = Path.cwd() / _DB_FILENAME
    try:
        engine = _ext.NativeEngine(str(db_path), _DEFAULT_DIM)
        engine.add_documents(documents)
    except Exception as error:  # noqa: BLE001 -- surface any engine failure as a clean CLI error
        print(f"engine ingest: error: {error}", file=sys.stderr)
        return 1

    print(f"Ingested {len(documents)} document(s), {total_chunks} chunk(s), into {db_path}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        # Caught here, not left to the native binding: NativeEngine.search_hybrid()
        # takes an unsigned C++ size_t, so a negative --top-k fails with a
        # nanobind type-mismatch message ("incompatible function arguments
        # ... Invoked with types: ...") that leaks internal binding details
        # instead of explaining the actual mistake.
        print(f"engine query: error: --top-k must be a positive integer (got {args.top_k})", file=sys.stderr)
        return 1

    db_path = Path.cwd() / _DB_FILENAME
    if not db_path.exists():
        print(
            f"engine query: error: no index found at {db_path} -- run 'engine ingest <folder>' first",
            file=sys.stderr,
        )
        return 1

    try:
        engine = _ext.NativeEngine(str(db_path), _DEFAULT_DIM)
        results = engine.search_hybrid(args.text, _hash_embed(args.text), args.top_k)
    except Exception as error:  # noqa: BLE001 -- surface any engine failure as a clean CLI error
        print(f"engine query: error: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No results.")
        return 0

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result.document_id}] (score={result.score:.4f}) {result.text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine", description="Local-first hybrid retrieval engine CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Chunk and index every .txt file in a folder.")
    ingest_parser.add_argument("folder_path", help="Folder containing .txt files to ingest.")
    ingest_parser.set_defaults(func=_cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Search the current directory's index.")
    query_parser.add_argument("text", help="Query text.")
    query_parser.add_argument("--top-k", type=int, default=5, dest="top_k", help="Number of results (default: 5).")
    query_parser.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
