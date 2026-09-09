"""Command-line interface for retrieval_engine:

    engine ingest <folder_path>                      Chunk and index every .txt
                                                       file in a folder.
    engine query "<text query>" [--decay] [--explain] Run a hybrid (dense +
                                                       sparse) search, optionally
                                                       discounted by recency and/or
                                                       printed with a full score
                                                       breakdown.

Both commands operate on an index file in the *current working directory*
(see _DB_FILENAME) -- `ingest` creates/updates it, `query` reads it. Run
`query` from the same directory you ran `ingest` from. A companion
`<index>.usearch` sidecar holding the vector graph is written and reloaded
automatically so `query` starts instantly instead of rebuilding the index.

`query`'s `--decay` (decay_lambda) applies exponential recency decay on top
of the hybrid score -- e.g. `--decay 0.05` discounts results at
e^(-0.05 * age_days); the default, 0, disables decay entirely (identical
results to a plain hybrid search). See
RetrievalEngine::search_memory()'s doc comment for the exact formula.

`query`'s `--explain` prints each result's full per-ranking score breakdown
(dense distance/rank, sparse bm25/rank, fused score, age/recency
factor/decayed score) instead of just the final score -- backed by
RetrievalEngine::search_memory_explained().

By default both commands use a small deterministic hashing-trick
"embedding" (_hash_embed below) purely so the CLI has *something* to feed
the dense index end to end; it is not a real semantic embedding.

Pass `--model <path> --model-dim <n>` to both `ingest` and `query` to route
text through the engine's built-in embedder instead (RetrievalEngine::embed()) --
then `engine query "Where do I live?"` generates the query vector under the
hood with no caller-supplied floats. Use the *same* `--model`/`--model-dim`
for `query` as you did for `ingest`: the index is built for one embedding
space and one dimensionality.
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


def _resolve_model_args(args: argparse.Namespace, command: str) -> int | None:
    """Validates the shared --model / --model-dim pair. Returns an exit code
    to fail with, or None if the arguments are consistent.
    """
    if args.model is not None and args.model_dim is None:
        print(f"engine {command}: error: --model requires --model-dim <n>", file=sys.stderr)
        return 1
    return None


def _embed_dim(args: argparse.Namespace) -> int:
    return args.model_dim if args.model is not None else _DEFAULT_DIM


def _make_embedder(engine: _ext.NativeEngine, args: argparse.Namespace):
    """Returns the text->vector callable for this run: the engine's built-in
    embedder when --model was given (loaded once, here), else the
    dependency-free hashing stand-in.
    """
    if args.model is not None:
        engine.load_embedding_model(args.model)
        return engine.embed
    return _hash_embed


def _cmd_ingest(args: argparse.Namespace) -> int:
    model_args_error = _resolve_model_args(args, "ingest")
    if model_args_error is not None:
        return model_args_error

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

    db_path = Path.cwd() / _DB_FILENAME
    try:
        engine = _ext.NativeEngine(str(db_path), _embed_dim(args))
        embed = _make_embedder(engine, args)
    except Exception as error:  # noqa: BLE001 -- surface any engine failure as a clean CLI error
        print(f"engine ingest: error: {error}", file=sys.stderr)
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
            native_chunk.embedding = embed(chunk.text)
            native_chunk.start_token = chunk.start_token
            native_chunk.end_token = chunk.end_token
            chunks.append(native_chunk)
        document.chunks = chunks

        total_chunks += len(chunks)
        documents.append(document)

    try:
        engine.add_documents(documents)
    except Exception as error:  # noqa: BLE001 -- surface any engine failure as a clean CLI error
        print(f"engine ingest: error: {error}", file=sys.stderr)
        return 1

    print(f"Ingested {len(documents)} document(s), {total_chunks} chunk(s), into {db_path}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    model_args_error = _resolve_model_args(args, "query")
    if model_args_error is not None:
        return model_args_error

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
        engine = _ext.NativeEngine(str(db_path), _embed_dim(args))
        query_vec = _make_embedder(engine, args)(args.text)
        # decay=0 is mathematically a no-op (e^0 == 1), so search_memory()/
        # search_memory_explained() with the default --decay are identical
        # to a plain hybrid search -- no need to branch between decayed and
        # non-decayed native calls.
        if args.explain:
            results = engine.search_memory_explained(args.text, query_vec, args.top_k, args.decay)
        else:
            results = engine.search_memory(args.text, query_vec, args.top_k, args.decay)
    except Exception as error:  # noqa: BLE001 -- surface any engine failure as a clean CLI error
        print(f"engine query: error: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No results.")
        return 0

    if args.explain:
        for rank, explanation in enumerate(results, start=1):
            print(
                f"{rank}. [{explanation.document_id}] {explanation.text}\n"
                f"   dense_present={explanation.dense_present} dense_distance={explanation.dense_distance:.4f} "
                f"dense_rank={explanation.dense_rank}\n"
                f"   sparse_present={explanation.sparse_present} sparse_bm25_score={explanation.sparse_bm25_score:.6f} "
                f"sparse_rank={explanation.sparse_rank}\n"
                f"   fused_score={explanation.fused_score:.4f}  "
                f"age_seconds={explanation.age_seconds:.1f}  "
                f"recency_factor={explanation.recency_factor:.4f}  "
                f"decayed_score={explanation.decayed_score:.4f}"
            )
        return 0

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result.document_id}] (score={result.score:.4f}) {result.text}")
    return 0


def _add_model_arguments(subparser: argparse.ArgumentParser) -> None:
    """The shared --model / --model-dim pair: route text through the
    engine's built-in embedder instead of the hashing stand-in. Both
    commands must be given the same values (the index is built for one
    embedding space).
    """
    subparser.add_argument(
        "--model",
        default=None,
        help="Path to a local embedding model file. Requires --model-dim. "
        "Without this, a deterministic hashing stand-in is used.",
    )
    subparser.add_argument(
        "--model-dim",
        type=int,
        default=None,
        dest="model_dim",
        help="Output dimension of --model (required whenever --model is given).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine", description="Local-first hybrid retrieval engine CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Chunk and index every .txt file in a folder.")
    ingest_parser.add_argument("folder_path", help="Folder containing .txt files to ingest.")
    _add_model_arguments(ingest_parser)
    ingest_parser.set_defaults(func=_cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Search the current directory's index.")
    query_parser.add_argument("text", help="Query text.")
    _add_model_arguments(query_parser)
    query_parser.add_argument("--top-k", type=int, default=5, dest="top_k", help="Number of results (default: 5).")
    query_parser.add_argument(
        "--decay",
        type=float,
        default=0.0,
        dest="decay",
        help="Recency decay lambda (default: 0, i.e. no decay). Larger values discount older results more.",
    )
    query_parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the full per-result score breakdown (dense/sparse/fused/decay) instead of just the score.",
    )
    query_parser.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
