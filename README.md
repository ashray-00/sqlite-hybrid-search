# retrieval-engine

**Ultra-fast, in-process hybrid search & agent memory engine in C++17 with Python bindings.**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)
![PyPI](https://img.shields.io/badge/PyPI-unpublished-lightgrey.svg)

Drop it **inside** an app to give a local LLM two things it lacks: knowledge of
your private data (RAG) and memory that survives across sessions — with **no
vector database to run and no cloud**. SQLite is the source of truth; the
usearch HNSW graph is persisted to a `.usearch` sidecar and memory-loaded on
open, so startup is instant regardless of corpus size.

---

## Why this exists

| | |
|---|---|
| **Embedded / zero-service** | One process, one SQLite file plus a `.usearch` sidecar. No Docker, no daemon, no port. A query is a function call — sub-millisecond for dense search (see [benchmarks](#empirical-benchmarks-verified)). |
| **Instant startup** | The vector index is loaded from disk, not rebuilt — ~25 ms to open a 100k-chunk store, versus ~15 s to reconstruct it from SQLite. |
| **Hybrid retrieval** | Dense vector search (usearch, cosine HNSW) **+** sparse keyword search (SQLite FTS5 / BM25), fused by **Reciprocal Rank Fusion** (RRF, k=60). Rank-based fusion — no score normalisation, no per-query tuning. |
| **Agent memory** | Exponential recency decay — `score × e^(−λ·age_days)` — so a fresher, slightly-less-similar memory can outrank a stale one. `λ = 0` is an exact no-op. |
| **Explainable** | `search_explained()` returns the full per-result score trail: dense distance & rank, BM25 score & rank, fused score, recency factor, decayed score. |
| **Bring your own embeddings** | The core takes caller-supplied vectors and computes nothing. An optional built-in embedder (ONNX Runtime, e.g. all-MiniLM-L6-v2) is layered on top for a text-in path. |

---

## Install

Not yet published to PyPI. Build from source (needs a C++17 toolchain, CMake ≥ 3.24,
and SQLite; usearch is fetched automatically):

```console
git clone https://github.com/<owner>/retrieval-engine   # your fork/repo URL
cd retrieval-engine
python -m venv .venv && .venv/bin/pip install -e .
```

On macOS/ARM64, install SQLite via Homebrew first (`brew install sqlite`) and
configure with `-DCMAKE_PREFIX_PATH=/opt/homebrew`. The built-in ONNX embedder
is optional — without ONNX Runtime installed, the engine still builds and the
caller-supplied-vector path is unaffected.

---

## Quickstart (Python)

```python
import retrieval_engine

engine = retrieval_engine.Engine("memory.sqlite3", dim=3)

# Ingest documents with caller-supplied embeddings (one vector per document).
engine.add(
    documents=[
        {"id": "home", "text": "I live in Munich, Germany."},
        {"id": "pet",  "text": "My cat is named Pixel."},
    ],
    embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
)

# Hybrid search: dense vector + BM25 keyword, RRF-fused.
hits = engine.search_hybrid("Where do I live?", [1.0, 0.0, 0.0], top_k=2)
print(hits[0]["document_id"], hits[0]["score"])          # -> home 0.0328

# Memory retrieval: same, discounted by recency (decay_lambda=0 disables it).
recall = engine.search_memory("Where do I live?", [1.0, 0.0, 0.0], top_k=2, decay_lambda=0.1)

# Full score breakdown for one result.
trail = engine.search_explained("Where do I live?", [1.0, 0.0, 0.0], top_k=1)[0]
print(trail["dense_rank"], trail["sparse_rank"], trail["fused_score"], trail["recency_factor"])
```

Per-chunk timestamps (so recency decay can actually reorder results) are set
through the native `retrieval_engine._retrieval_engine_ext` types — see
`tests/test_memory_search.py`.

## Quickstart (CLI)

The `engine` console script chunks a folder of `.txt` files into an index in the
current directory. It uses a deterministic hashing stand-in for embeddings unless
you pass `--model` / `--model-dim`.

```console
$ engine ingest ./docs
Ingested 2 document(s), 2 chunk(s), into /path/to/cwd/.retrieval_engine.sqlite3

$ engine query "where do I live" --decay 0.1
1. [notes.txt] (score=0.0328) I live in Munich. My office is near the river.
2. [work.txt] (score=0.0161) The quarterly report is due on Friday.

$ engine query "where do I live" --decay 0.1 --explain    # full score trail
```

---

## Empirical benchmarks (verified)

Reproduce with `python benchmarks/run_eval.py`; full method and raw data in
[`BENCHMARKS.md`](BENCHMARKS.md) and [`benchmarks/results.json`](benchmarks/results.json).
Machine: Apple Silicon (macOS arm64), single thread, embedding dim 64, 100 labelled queries.

> **Read the corpus honestly.** It is synthetic, and the `dense` column uses a
> 64-dim hashed bag-of-words, not a trained model — so `sparse` is a best case
> and `dense` a worst case. Latency and memory are model-independent and
> transfer directly; the quality table shows *how RRF behaves when the two
> signals disagree*, not an absolute score for dense retrieval.

### Retrieval quality (Recall@10 / nDCG@10)

| Approach | 1k | 10k | 100k |
|---|---|---|---|
| Dense only | 0.987 / 0.944 | 0.783 / 0.752 | 0.493 / 0.507 |
| Sparse only (BM25) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| **Hybrid (RRF)** | **1.000 / 0.992** | **1.000 / 0.969** | **1.000 / 0.951** |
| Hybrid + recency decay | 1.000 / 0.998 | 1.000 / 0.998 | 1.000 / 0.997 |

Hybrid recovers **every point of recall the dense path loses at scale** (1.000
vs 0.493 at 100k) — RRF lets the keyword side carry the query when the vector
side degrades.

### Latency (warm cache, single thread)

| Approach | 1k p50 / p99 | 100k p50 / p99 | 100k throughput |
|---|---|---|---|
| Dense | 0.090 / 0.097 ms | 0.173 / 0.211 ms | 5,705 q/s |
| Hybrid | 0.524 / 0.558 ms | 49.4 / 50.5 ms | 20 q/s |

Dense stays **sub-0.2 ms p50 at 100k**. Sparse/hybrid latency is dominated by
FTS5 and grows with corpus size — the main query-time bottleneck.

### Startup: instant, disk-backed index

The usearch graph is serialised to a `<db>.usearch` sidecar and memory-loaded
on the next open, instead of being rebuilt from SQLite:

| | 1k | 10k | 100k |
|---|---|---|---|
| Index load on open | 0.7 ms | 3.2 ms | **24.5 ms** |
| _(previously: rebuild from SQLite)_ | 40 ms | 0.7 s | **14.6 s** |

SQLite stays authoritative — a missing, truncated, or out-of-sync sidecar is
rejected and the engine rebuilds transparently.

### Memory & storage footprint

| | 1k | 10k | 100k |
|---|---|---|---|
| SQLite on disk | 0.57 MB | 5.2 MB | 52 MB |
| `.usearch` sidecar | ~0.5 MB | ~4 MB | 40.5 MB |
| Peak RSS (Python-driven) | 35 MB | 79 MB | 430 MB |

The native C++ cross-check (`benchmarks/run_benchmarks.cpp`) puts the
**engine-only peak RSS at ~38 MB for 20,000 documents** — most of the
Python-driven figure is the benchmark driver holding the corpus, not the engine.

---

## Comparison

| | **retrieval-engine** | sqlite-vec + glue | ChromaDB | Qdrant / Milvus / Weaviate |
|---|---|---|---|---|
| Deployment | in-process library, 1 file | in-process (SQLite ext) | embedded lib **or** server | separate server / cluster |
| Process to run | none | none | none (embedded) / one (server) | one+ |
| Dense + sparse hybrid | built in (RRF) | DIY (wire up FTS5 + fusion) | dense-first | built in (server-side) |
| Recency / memory semantics | built in (`search_memory`) | DIY | DIY | DIY (metadata + custom scoring) |
| Score-trail / explainability | `search_explained()` | DIY | limited | varies |
| Text + metadata storage | SQLite (authoritative) | second table you design | built in | built in |
| Chunking | token-window built in | DIY | some | DIY / integrations |
| Vector-index persistence | `.usearch` sidecar, auto-managed | rows in SQLite | persisted | persisted |
| Ops surface | none | none | small | real (scaling, backups, upgrades) |
| Best fit | desktop / CLI / edge agents, local-first, privacy | you already live in SQLite | Python RAG prototypes | multi-tenant, large-scale, networked |

**Reach for a dedicated vector DB instead** if you need horizontal scale,
multi-writer concurrency, or sub-10 ms keyword search over millions of
documents.

---

## C++ integration

The public header exposes no usearch or SQLite types (Pimpl idiom); the only
hard dependency is SQLite (usearch is fetched by CMake).

```cmake
include(FetchContent)
FetchContent_Declare(retrieval_engine
    GIT_REPOSITORY https://github.com/<owner>/retrieval-engine
    GIT_TAG main)
FetchContent_MakeAvailable(retrieval_engine)

target_link_libraries(your_target PRIVATE retrieval_engine::core)
```

```cpp
#include "retrieval_engine/retrieval_engine.hpp"

retrieval_engine::RetrievalEngine engine("memory.sqlite3", /*dim=*/384);

retrieval_engine::DocumentInput doc;
doc.document_id = "home";
doc.chunks.push_back({ "I live in Munich.", embedding /*std::vector<float>*/, 0, 4 });
engine.add_documents({ doc });

auto hits = engine.search_hybrid("Where do I live?", query_vec, /*k=*/5);
auto memory = engine.search_memory("Where do I live?", query_vec, /*k=*/5, /*decay_lambda=*/0.1f);
```

A single instance is **not thread-safe** — confine it to one thread or lock
externally (one instance per thread, each with its own SQLite connection).

---

## Project layout

```
core/        C++17 engine (chunking, dense index, FTS5, RRF fusion, recency decay)
bindings/    nanobind extension module
python/      retrieval_engine package (friendly wrapper + `engine` CLI)
benchmarks/  reproducible recall / latency / memory harness
docs/        architecture decision records + development log
```

## Building & testing

```console
cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build build
ctest --test-dir build --output-on-failure      # C++ suite
.venv/bin/pytest                                 # Python + CLI suite
```

## Contributing

Issues and pull requests welcome. The project follows a strict TDD workflow
(failing test first, then implementation, then an independent review pass) and
enforces formatting with `.clang-format` (C++) and `ruff` (Python). Run both
test suites before opening a PR. Design rationale is recorded as ADRs in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## License

[MIT](LICENSE) © 2026 Ashray Adhikari
