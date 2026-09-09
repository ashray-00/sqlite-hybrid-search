# sqlite-hybrid-search

**Ultra-fast, in-process hybrid search & agent memory engine in C++17 with Python bindings.**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)
[![PyPI](https://img.shields.io/pypi/v/sqlite-hybrid-search.svg)](https://pypi.org/project/sqlite-hybrid-search/)

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
| **Instant startup** | The vector index is loaded from disk, not rebuilt — ~40 ms to open a 100k-chunk store, versus ~15 s to reconstruct it from SQLite. |
| **Hybrid retrieval** | Dense vector search (usearch, cosine HNSW) **+** sparse keyword search (SQLite FTS5 / BM25), fused by **Reciprocal Rank Fusion** (RRF, k=60). Rank-based fusion — no score normalisation, no per-query tuning. |
| **Agent memory** | Exponential recency decay — `score × e^(−λ·age_days)` — so a fresher, slightly-less-similar memory can outrank a stale one. `λ = 0` is an exact no-op. |
| **Explainable** | `search_explained()` returns the full per-result score trail: dense distance & rank, BM25 score & rank, fused score, recency factor, decayed score. |
| **Bring your own embeddings** | The core takes caller-supplied vectors and computes nothing. An optional built-in embedder (ONNX Runtime, e.g. all-MiniLM-L6-v2) is layered on top for a text-in path. |

---

## Install

```console
pip install sqlite-hybrid-search
```

Prebuilt wheels cover **Linux x86-64** and **macOS (Apple Silicon)**, CPython
3.9–3.13. On any other platform `pip` builds from the sdist — that needs the
toolchain in [From source](#from-source-other-platforms--development) below.

## From source (other platforms / development)

You need a C++17 toolchain, CMake ≥ 3.24, Python ≥ 3.9 with `venv`, and SQLite
(with FTS5 — the default on mainstream builds). `usearch` and GoogleTest are
fetched automatically.

```console
git clone https://github.com/ashray-00/sqlite-hybrid-search
cd sqlite-hybrid-search
python3 -m venv .venv && .venv/bin/pip install -e .
```

**Dependencies:**

| | macOS (Homebrew) | Debian / Ubuntu | Fedora / RHEL | Arch |
|---|---|---|---|---|
| Toolchain + CMake | `xcode-select --install`<br>`brew install cmake` | `sudo apt install build-essential cmake python3-venv` | `sudo dnf install gcc-c++ cmake python3-devel` | `sudo pacman -S base-devel cmake` |
| SQLite | `brew install sqlite` | `sudo apt install libsqlite3-dev` | `sudo dnf install sqlite-devel` | `sudo pacman -S sqlite` |

- **macOS only:** Homebrew's `sqlite` is keg-only, so configure the C++ build
  with `-DCMAKE_PREFIX_PATH=/opt/homebrew` (the CMake project also autodetects
  it via `brew --prefix`). On Linux the system SQLite is found with no extra
  flags.
- **Ubuntu 22.04** ships CMake 3.22; either `.venv/bin/pip install "cmake>=3.24"`
  or add the [Kitware APT repo](https://apt.kitware.com/).

**Optional built-in ONNX embedder.** Without ONNX Runtime the engine still
builds and every caller-supplied-vector path works unchanged; only
`load_embedding_model()` / `add_text()` / `search_text()` are unavailable.

| | Install ONNX Runtime |
|---|---|
| macOS | `brew install onnxruntime` |
| Linux | Download a release from [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/releases) (`onnxruntime-linux-x64-*.tgz`), then `sudo cp -r onnxruntime-linux-x64-*/include/* /usr/local/include/` and `sudo cp -rP onnxruntime-linux-x64-*/lib/* /usr/local/lib/ && sudo ldconfig`. Or point CMake at it directly: `-DONNXRUNTIME_INCLUDE_DIR=<dir> -DONNXRUNTIME_LIBRARY=<dir>/libonnxruntime.so`. |

---

## Quickstart (Python)

```python
import sqlite_hybrid_search

engine = sqlite_hybrid_search.Engine("memory.sqlite3", dim=3)

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
through the native `sqlite_hybrid_search._sqlite_hybrid_search_ext` types — see
`tests/test_memory_search.py`.

## Quickstart (CLI)

The `hybrid-search` console script chunks a folder of `.txt` files into an index in the
current directory. It uses a deterministic hashing stand-in for embeddings unless
you pass `--model` / `--model-dim`.

```console
$ hybrid-search ingest ./docs
Ingested 2 document(s), 2 chunk(s), into /path/to/cwd/.hybrid_search.sqlite3

$ hybrid-search query "where do I live" --decay 0.1
1. [notes.txt] (score=0.0328) I live in Munich. My office is near the river.
2. [work.txt] (score=0.0161) The quarterly report is due on Friday.

$ hybrid-search query "where do I live" --decay 0.1 --explain    # full score trail
```

---

## Empirical benchmarks (verified)

Reproduce with `python benchmarks/run_eval.py`; full method and raw data in
[`BENCHMARKS.md`](BENCHMARKS.md) and [`benchmarks/results.json`](benchmarks/results.json).
Machine: Apple Silicon (macOS arm64), single thread, embedding dim 64, 100 labelled queries.

> **Read the corpus honestly.** It is synthetic (a Zipf-distributed pseudo-word
> vocabulary), and the `dense` column uses a 64-dim hashed bag-of-words, not a
> trained model — treat it as a floor. `sparse` gets a clean per-query
> exact-match cue, so read its perfect score as an upper bound. Latency and
> memory transfer directly. A real BEIR run with a trained embedder is tracked
> follow-up work.

### Retrieval quality (Recall@10 / nDCG@10)

| Approach | 1k | 10k | 100k |
|---|---|---|---|
| Dense only | 1.000 / 0.987 | 0.990 / 0.947 | 0.883 / 0.752 |
| Sparse only (BM25) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| **Hybrid (RRF)** | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 0.993** |
| Hybrid + recency decay | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

Hybrid recovers **the recall and ranking dense loses at scale** (1.000 / 0.993
vs 0.883 / 0.752 at 100k) — RRF lets the BM25 side carry the query when the
vector side weakens.

### Latency (warm cache, single thread)

| Approach | 1k p50 / p99 | 100k p50 / p99 | 100k throughput |
|---|---|---|---|
| Dense | 0.092 / 0.103 ms | 0.185 / 0.334 ms | 5,209 q/s |
| Hybrid | 0.169 / 0.192 ms | 6.7 / 8.0 ms | 148 q/s |

Dense stays **sub-0.2 ms p50 at 100k**. Hybrid stays **single-digit ms** and
grows with corpus size (FTS5 posting-list merge). The agent-memory read
(`hybrid_decay`) is ~5× slower — a per-candidate `created_at` lookup that's
next on the optimisation list.

### Startup: instant, disk-backed index

The usearch graph is serialised to a `<db>.usearch` sidecar and memory-loaded
on the next open, instead of being rebuilt from SQLite:

| | 1k | 10k | 100k |
|---|---|---|---|
| Index load on open | 0.6 ms | 3.0 ms | **~40 ms** |
| _(previously: rebuild from SQLite)_ | 40 ms | 0.7 s | **14.6 s** |

SQLite stays authoritative — a missing, truncated, or out-of-sync sidecar is
rejected and the engine rebuilds transparently.

### Memory & storage footprint

| | 1k | 10k | 100k |
|---|---|---|---|
| SQLite on disk | 0.59 MB | 5.4 MB | 55 MB |
| `.usearch` sidecar | ~0.5 MB | ~4 MB | ~41 MB |
| Peak RSS (Python-driven) | 36 MB | 79 MB | 431 MB |

The native C++ cross-check (`benchmarks/run_benchmarks.cpp`) puts the
**engine-only peak RSS at ~37 MB for 20,000 documents** — most of the
Python-driven figure is the benchmark driver holding the corpus, not the engine.

---

## Comparison

| | **sqlite-hybrid-search** | sqlite-vec + glue | ChromaDB | Qdrant / Milvus / Weaviate |
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
hard dependency is SQLite (usearch is fetched by CMake). The C++ symbols live
in the `retrieval_engine` namespace (header path `retrieval_engine/`) — an
internal name kept stable across the Python-package rename.

```cmake
include(FetchContent)
FetchContent_Declare(sqlite_hybrid_search
    GIT_REPOSITORY https://github.com/ashray-00/sqlite-hybrid-search
    GIT_TAG main)
FetchContent_MakeAvailable(sqlite_hybrid_search)

target_link_libraries(your_target PRIVATE sqlite_hybrid_search::core)
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
python/      sqlite_hybrid_search package (friendly wrapper + `hybrid-search` CLI)
benchmarks/  reproducible recall / latency / memory harness
docs/        architecture decision records + development log
```

## Building & testing

```console
# Linux
cmake -B build && cmake --build build

# macOS (Homebrew SQLite / ONNX Runtime live under /opt/homebrew)
cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build build

ctest --test-dir build --output-on-failure      # C++ suite
.venv/bin/pytest                                 # Python + CLI suite
```

## Roadmap

- **Concurrent reads.** A single-writer / lock-free-reads model so one engine
  instance can serve many query threads (today: one instance per thread).
- **Batched recency lookup.** `search_memory` fetches each candidate's
  `created_at` with its own query; one batched lookup removes the
  `hybrid_decay` latency gap.
- **Real retrieval eval.** A BEIR run (SciFact / NFCorpus) with a trained ONNX
  embedder, alongside the mechanism benchmark.
- **Wider wheels.** Windows and Linux aarch64.

## Contributing

Issues and pull requests welcome. The project follows a strict TDD workflow
(failing test first, then implementation, then an independent review pass) and
enforces formatting with `.clang-format` (C++) and `ruff` (Python). Run both
test suites before opening a PR. Design rationale is recorded as ADRs in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## License

[MIT](LICENSE) © 2026 Ashray Adhikari
