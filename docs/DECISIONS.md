# Architecture Decision Records

The significant design decisions behind this engine, in standard ADR form
(Context / Decision / Consequences). The blow-by-blow development history —
review findings, false starts, environment friction — lives in
[`dev-log.md`](dev-log.md); this file is the curated record.

| # | Decision | Status |
|---|---|---|
| [ADR-1](#adr-1-in-process-library-not-a-vector-database-server) | In-process library, not a vector-database server | Accepted |
| [ADR-2](#adr-2-sqlite-is-authoritative-usearch-is-a-rebuildable-sidecar) | SQLite authoritative, usearch a rebuildable sidecar | Accepted |
| [ADR-3](#adr-3-rebuild-the-hnsw-index-from-sqlite-on-open-no-index-file) | Rebuild the HNSW index from SQLite on open | Accepted |
| [ADR-4](#adr-4-hybrid-retrieval-via-reciprocal-rank-fusion-k60) | Hybrid retrieval via Reciprocal Rank Fusion (k=60) | Accepted |
| [ADR-5](#adr-5-agent-memory-as-exponential-recency-decay-on-the-fused-score) | Agent memory as exponential recency decay | Accepted |
| [ADR-6](#adr-6-caller-supplied-embeddings-by-default-built-in-model-optional) | Caller-supplied embeddings by default | Accepted |
| [ADR-7](#adr-7-quote-and-or-sanitisation-of-fts5-match-input) | Quote-and-OR sanitisation of FTS5 MATCH input | Accepted |
| [ADR-8](#adr-8-layered-python-bindings-thin-nanobind-shell--pure-python-ergonomics) | Layered Python bindings | Accepted |
| [ADR-9](#adr-9-single-threaded-instances-no-internal-locking) | Single-threaded instances, no internal locking | Accepted |

---

## ADR-1: In-process library, not a vector-database server

**Context.** Adding retrieval or memory to a local application normally
means running a separate vector database (Qdrant, Milvus, Chroma in server
mode) plus glue code. That assumes an operator, a network hop, and a
deployment surface — overkill for a desktop app, a CLI, or an offline agent
that runs on one machine, and a non-starter for "data never leaves the
device" requirements.

**Decision.** Ship the engine as an embeddable C++17 library with Python
bindings. All state lives in a single SQLite file plus an in-memory usearch
index. No daemon, no port, no container. The retrieval pipeline
(chunk → dense + sparse → fusion → optional decay → explain) and the memory
semantics run in the caller's process.

**Consequences.**
- No operational surface: no service to supervise, secure, upgrade, or
  scale; startup is a constructor call.
- Latency is a function call, not a round trip — sub-millisecond dense
  queries (see `BENCHMARKS.md`).
- The ceiling is one machine's RAM and one process's throughput. There is
  no horizontal scale, no multi-writer story, and no cross-language wire
  protocol. Workloads that need those should use a dedicated vector DB.
- The value has to come from the *whole* packaged pipeline plus memory
  semantics, not from the index itself — `sqlite-vec` already occupies the
  bare "embeddable vector index" slot.

---

## ADR-2: SQLite is authoritative, usearch is a rebuildable sidecar

**Context.** The engine keeps two stores: SQLite (documents, chunks,
metadata, timestamps, FTS5) and a usearch HNSW index (dense vectors).
usearch has no transactions. A naive `add_documents()` that writes both in
the same loop can leave a vector live in the index with no backing SQLite
row if a later document in the batch fails and rolls the transaction back —
an inconsistency with no recovery path.

**Decision.** SQLite is the single source of truth. Every vector is also
stored as a `BLOB` in the `chunks` row it belongs to. `add_documents()`
commits the entire SQLite transaction (documents + chunks + FTS5) *first*,
and only then adds the corresponding vectors to usearch. Index mutation
never precedes the commit it depends on.

**Consequences.**
- The only possible drift is the index lagging SQLite (recoverable by
  rebuilding it — see ADR-3), never the reverse.
- A corrupt or lost index file is not data loss.
- Each vector is stored twice (once in SQLite, once in the in-memory
  index), costing roughly one extra copy of the raw floats on disk.
- `add_documents()` validates the whole batch up front, so one bad chunk
  rejects the call without partially ingesting earlier documents.

---

## ADR-3: Rebuild the HNSW index from SQLite on open (no index file)

**Context.** usearch can serialise its HNSW graph to disk. The alternative
is to persist nothing and reconstruct the index from the vectors in SQLite
each time the engine opens.

**Decision.** Do not persist the index. On construction, `ChunkStore`
streams every stored `(chunk_id, embedding)` from SQLite and re-inserts it
into a fresh usearch index. This runs unconditionally, not just after a
detected corruption.

**Consequences.**
- No index-file format to version, migrate, or keep in sync with the
  schema; no risk of a stale or partially written sidecar. Recovery from
  index corruption is automatic and needs no code.
- Open cost is O(N) HNSW insertions. Measured: ~40 ms at 1k chunks,
  ~0.7 s at 10k, ~15 s at 100k (see `BENCHMARKS.md`). Fine for a
  long-lived process; a real cost for short-lived CLI invocations over a
  large corpus.
- Peak memory during open includes building the graph. There is no
  mmap-the-index-and-go option.
- **Revisit when** the rebuild time at a target corpus size becomes a
  user-visible problem. The seam is small: `DenseIndex` already isolates
  usearch, so an optional "load from sidecar, fall back to rebuild" path
  can slot in behind it without touching callers.

---

## ADR-4: Hybrid retrieval via Reciprocal Rank Fusion (k=60)

**Context.** Dense vector search captures meaning but is weak on exact
tokens — names, identifiers, rare terms. Sparse BM25 (SQLite FTS5) is the
opposite. Combining them requires fusing two result lists whose scores are
not comparable: cosine distance and BM25 are different scales, different
directions (both lower-is-better), and BM25 is unbounded and
corpus-dependent.

**Decision.** Fuse by rank, not by score. Reciprocal Rank Fusion assigns
each chunk `Σ 1 / (k + rank_i)` over the rankings it appears in, with
`k = 60` (the value from the original Cormack et al. RRF paper, widely
adopted since). Score normalisation (min-max, z-score, etc.) was rejected:
it is sensitive to outliers and to the score distribution of each query,
and BM25's unbounded range makes any fixed normalisation fragile.

**Consequences.**
- No score calibration, no per-query tuning, no distribution assumptions.
  Only the ordering of each input list matters.
- Robust failure mode: when one retriever returns garbage, its
  contribution is bounded by `1/(k+1)` per result, so the other retriever
  still dominates. Measured: hybrid holds Recall@10 = 1.000 even where
  dense-only collapses (see `BENCHMARKS.md`).
- `k = 60` is effectively a constant, not a knob; it flattens the
  contribution of rank differences beyond the first ~10 positions.
- Fusion discards score magnitude, so a result that is overwhelmingly
  better by one retriever's score is not rewarded for the margin — only
  for being rank 1. On queries where one signal is already perfect, fusing
  in a noisier signal can cost a little top-rank precision.
- Ties on the fused score are broken deterministically by
  `(document_id, chunk_index)` so `search_explained()` is reproducible.

---

## ADR-5: Agent memory as exponential recency decay on the fused score

**Context.** An agent accumulates memories over many sessions. When an old
fact and a newer, slightly-less-similar fact both match a query, raw
relevance ranking surfaces the stale one. The memory layer needs a
time-aware tie-breaker without a separate ranking model.

**Decision.** Multiply each result's fused RRF score by an exponential
recency factor:

```
decayed_score = fused_score * exp(-decay_lambda * age_days)
age_days      = max(now - created_at, 0) / 86400
```

Age is normalised to days so `decay_lambda` values around 0.05–0.5
correspond to meaningful day-scale half-lives. `decay_lambda = 0` is an
exact no-op (`exp(0) = 1`), so the memory search degrades to plain hybrid
search. The factor is guarded on three edges: the age used in the exponent
is clamped to `>= 0` (a future `created_at` or backward clock step must
never *inflate* a score); the factor is floored at `0.01` (an old but
critical memory is heavily discounted, never zeroed out); and a
non-finite `exp()` result (pathological `decay_lambda`) returns the floor
rather than propagating NaN/Inf into the ranking.

**Consequences.**
- One interpretable parameter. Callers who do not want decay pay nothing.
- `search_memory()` is a pure read — it decays against `created_at` and
  never refreshes `last_accessed_at`, so identical calls return identical
  results. Access-pattern-aware forgetting is deliberately out of scope.
- The raw, unclamped age is still reported by
  `search_memory_explained()` for debugging; only the decay math clamps.
- Decay is applied *after* fusion over the top-k candidate set, so a
  memory that never entered the fused candidate list cannot be surfaced by
  recency alone.

---

## ADR-6: Caller-supplied embeddings by default, built-in model optional

**Context.** Bundling an embedding model makes the library heavy and ties
it to a runtime (ONNX, llama.cpp) and a model licence. Requiring callers to
supply vectors keeps the core small but raises the barrier to a first
result.

**Decision.** The primary API (`add_documents()`, `search_dense()`,
`search_hybrid()`, `search_memory()`) takes caller-supplied `float`
vectors and never computes embeddings. A built-in path
(`load_embedding_model()` + `add_text()` / `search_text()`) is layered on
top behind a `TextEmbedder` interface, with an ONNX Runtime backend and a
dependency-free mock. The model file's format is detected by extension, and
the ONNX backend degrades gracefully: if ONNX Runtime is absent at build
time the engine still builds with the mock only.

**Consequences.**
- The core has no ML dependency; it links only SQLite and header-only
  usearch.
- Callers integrating an existing embedding pipeline pass vectors straight
  through with no conversion.
- Two ingestion entry points to document and keep behaviourally aligned.
- The built-in model is a convenience, not the contract — the
  vector-in path is always available whether or not a model is attached.

---

## ADR-7: Quote-and-OR sanitisation of FTS5 MATCH input

**Context.** FTS5's `MATCH` operand is parsed by FTS5's own query grammar
(`AND`/`OR`/`NOT`, quoted phrases, leading `-`, `*` prefixes, `column:`
filters, `NEAR()`). Passing raw user text through means an ordinary
hyphenated word or an unbalanced quote raises a SQLite error instead of
matching literal text.

**Decision.** Split the query on whitespace, wrap each token in its own
double-quoted phrase (escaping embedded quotes by doubling), and join the
phrases with `OR`. Quoting removes every character from FTS5's operator
grammar; `OR` matches typical keyword-search intent (any term, ranked by
BM25) rather than requiring every term.

**Consequences.**
- Any input string is a valid query; nothing the caller types is
  interpreted as query syntax.
- Advanced FTS5 operators are unavailable through this path by design.
- `OR` semantics mean a query with common words also matches documents
  that contain only those common words; BM25's IDF weighting is what keeps
  rare-term matches ranked on top.

---

## ADR-8: Layered Python bindings (thin nanobind shell + pure-Python ergonomics)

**Context.** A direct 1:1 nanobind wrapper of the C++ API exposes
`DocumentInput` / `DocumentChunkInput` struct construction to Python, which
is verbose for the common "here are some docs and their vectors" case. A
hand-written binding that also reshapes the API mixes two jobs in C++.

**Decision.** Two layers. `_retrieval_engine_ext` (nanobind, in
`bindings/`) mirrors the C++ types and methods closely and is exposed as
`NativeEngine`. The public `retrieval_engine.Engine` is pure Python: it
adapts dict-shaped documents and parallel embedding lists onto the native
structs, and returns dict-shaped results. Every potentially slow native
method releases the GIL.

**Consequences.**
- The ergonomic surface is editable without recompiling.
- Power users can drop to `NativeEngine` for multi-chunk documents and
  per-chunk timestamps.
- One extra indirection per call on the Python side (negligible next to
  the native work).
- Two API descriptions to keep in sync (native and wrapper).

---

## ADR-9: Single-threaded instances, no internal locking

**Context.** The engine holds a SQLite connection and a mutable usearch
index. Making a single instance safe for concurrent calls would mean
locking around every method or a more granular scheme.

**Decision.** A `RetrievalEngine` instance is not thread-safe. Concurrent
calls into the same instance are not synchronised. Callers confine an
instance to one thread or add external locking. The attached embedder,
once loaded, is only ever read, so concurrent raw-text *queries* do not
race on it — but `load_embedding_model()` is a mutation like any other.

**Consequences.**
- No lock contention, no locking bugs, simplest possible reasoning about
  state.
- No query parallelism within one instance. A server-style caller runs one
  instance per thread (each with its own SQLite connection) or serialises
  access.
- **Revisit when** an embedding target needs concurrent throughput from a
  shared index; that is a deliberate, separate design effort.
