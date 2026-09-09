# Project 1 — Embeddable Retrieval & Memory Engine

*A build plan. Scoped for a solo developer, ~6–8 weeks to a usable v1. Written September 2026; tool landscape verified against current (2026) sources.*

---

## 1. What it is, in one line

A single fast library you drop **inside** an app that gives a local LLM two things it lacks: knowledge of your private data (RAG) and memory that survives across sessions — with **no separate vector database to run and no cloud**.

## 2. The problem it solves (use case)

Today, adding retrieval or memory to an AI app normally means running a separate vector database (Qdrant, Chroma) plus a pile of Python glue to chunk documents, embed them, search, blend keyword + semantic results, and re-rank. That is heavy and assumes a running service — overkill for anything meant to run on one machine.

Your engine collapses all of that into one in-process library: hand it documents, it stores/indexes them, ask a question, it returns the most relevant pieces instantly — and remembers facts across sessions. Because the core is C++, it is fast and light enough to embed in a desktop app, a CLI, or a local agent where a Python service stack would not fit.

**Who uses it:** builders of local/desktop AI apps, offline/edge assistants, and privacy-sensitive deployments (a strong fit for the German/EU market, where "data never leaves the machine" is a real requirement). It doubles as your freelance calling card for "private, on-prem AI over our data" work.

## 3. Positioning & honest differentiation

The retrieval space has two layers. **Do not build the bottom layer.**

- **Index layer (reuse):** `usearch` (C++, SIMD, HNSW, int8/4-bit quantization, multi-platform) is the current embeddable workhorse; `hnswlib` and `sqlite-vec` are the alternatives. These are solved. Build **on** usearch.
- **Memory-framework layer (crowded, but not where you'd compete):** Mem0 (~48k stars), Zep, Letta, Cognee, LangMem. Almost all are **Python, cloud/service-oriented, and depend on an LLM to extract facts.**

**The open gap — and your niche:** a fast, **embeddable, in-process, local-first** retrieval+memory engine in C++ with clean Python bindings, that needs no service and no mandatory LLM-extraction step. The closest existing thing is a Rust crate (`semantic-memory`: SQLite-authoritative + usearch sidecar + BM25 + RRF). That it exists **validates the concept**; that it's Rust-only and library-level leaves room for a C++ engine with better ergonomics and agent-memory semantics.

**Your unfair advantage:** you have already built this exact retrieval pipeline twice — hybrid dense + BM25 + RRF + cross-encoder rerank — in your Moat and Atlas projects. This is productizing work you have done, with C++ as the differentiator.

**The risk to design against (be honest about it):** the raw "embeddable vector index" slot is taken by usearch/sqlite-vec. If a user can get 90% of your value from `sqlite-vec` + 30 lines of Python, they will. Your value must come from the **full pipeline + memory semantics + DX**, not from the index. Pressure-test this before committing: write down the 30-line sqlite-vec version and make sure your engine is clearly, obviously better to use.

## 4. Current tool landscape (2026) — what to use

| Layer | Recommended | Notes |
|---|---|---|
| Vector index (ANN) | **usearch** (unum-cloud) | C++, SIMD, HNSW, quantization, mmap; the embeddable standard |
| Durable store + metadata | **SQLite** (authoritative) | Rebuild the index from SQLite if it corrupts; the proven pattern |
| Keyword / sparse (BM25) | **SQLite FTS5** | Already in SQLite; no extra dependency |
| Fusion | **RRF** (reciprocal rank fusion) | You've implemented this before |
| Embedding model (default) | **nomic-embed-text** or **BGE-M3** | BGE-M3 gives dense **and** sparse in one model; nomic is light/CPU-friendly |
| Embedding model (edge) | **EmbeddingGemma-300M** | Purpose-built for on-device |
| Embedding model (quality) | **Qwen3-Embedding-0.6B** | MRL (truncatable 32–1024 dims), ONNX + GGUF, matching reranker |
| Reranker | **bge-reranker-v2-m3** or **Qwen3-Reranker-0.6B** | Cross-encoder final pass |
| Embedding runtime (local) | **llama.cpp** (GGUF) or **ONNX Runtime** / fastembed | Keeps the engine dependency-light; matches your C++ core |
| Chunking | **chonkie** (or your own) | Fast; or implement token/semantic chunking yourself in C++ |
| Python bindings | **pybind11** or **nanobind** | nanobind is lighter/faster to compile |
| Eval / testing | **BEIR** datasets, `ir_measures`, **ann-benchmarks**, RAGAS | For recall/nDCG and end-to-end RAG quality |

**Design choice:** default to running the embedding model **outside** the core (via llama.cpp/ONNX) and let the user pass vectors in, OR bundle a llama.cpp embedding path. Start with "user supplies embeddings"; add a built-in embedder in a later stage. This keeps v1 small.

## 5. Architecture (one paragraph)

SQLite is the source of truth for documents, chunks, metadata, and memory records. A **usearch** HNSW index is an acceleration sidecar (rebuildable from SQLite). Ingestion = chunk → embed → store in SQLite → add vector to usearch → index text in FTS5. Query = embed query → dense search (usearch) + sparse search (FTS5) → **RRF fusion** → optional cross-encoder rerank → return top-k with metadata. Memory adds a layer on top: records with timestamps, source, recency/decay, dedup/update, and namespaces (per user/session). A thin C++ API is exposed to Python via pybind11/nanobind, plus a small CLI.

## 6. Build plan — staged milestones

Each stage is a working milestone. For each: **Goal · Implement · Need · Test · Done.**

### Stage 0 — Skeleton & decisions (2–3 days)
- **Goal:** compiling C++ project with SQLite + usearch linked, and a decision log.
- **Implement:** CMake project; link SQLite and usearch; a `hello` that inserts one vector, searches it, returns it. Write the "30-line sqlite-vec competitor" and your differentiation notes in the README.
- **Need:** CMake, a C++17 toolchain, SQLite, usearch (header/lib), a tiny sample dataset.
- **Test:** unit test — insert N random vectors, nearest-neighbour query returns the known-closest; SQLite row count matches.
- **Done:** `ctest` green; you can articulate in one paragraph why this beats sqlite-vec-plus-glue.

### Stage 1 — Ingestion + dense retrieval (week 1)
- **Goal:** put documents in, get relevant chunks out, by meaning.
- **Implement:** chunker (token-window with overlap to start); `add_documents()`; store chunks+metadata in SQLite; embeddings supplied by caller for now; add vectors to usearch; `search(query_vec, k)`.
- **Need:** an embedding model available to the test harness (nomic-embed-text via llama.cpp or fastembed) to generate query/doc vectors; a small labelled corpus (a BEIR subset, e.g. SciFact or NFCorpus).
- **Test:** **recall@k and nDCG@k** on the BEIR subset vs a brute-force baseline (usearch should be ≥95% of brute-force recall); ingestion of 10k chunks completes and persists (reopen DB, data present).
- **Done:** dense retrieval returns sensible results and recall is measured, not asserted.

### Stage 2 — Hybrid retrieval (BM25 + RRF) + rerank (week 2)
- **Goal:** better relevance than dense-only, especially for exact terms/names.
- **Implement:** FTS5 sparse index; `rrf_fuse(dense_results, sparse_results)`; optional cross-encoder rerank stage (call bge-reranker via the runtime); a `search_explained()` that returns the score breakdown (dense score, sparse score, fused rank, rerank score).
- **Need:** the reranker model; queries with known-relevant docs from your BEIR subset.
- **Test:** nDCG@10 of hybrid **> ** dense-only on the same queries (this is the headline quality result); `search_explained()` output is correct on hand-checked examples; rerank improves top-5 ordering on a spot-check set.
- **Done:** hybrid + rerank measurably beats dense-only, with an explainable score trail.

### Stage 3 — Python bindings + CLI (week 3)
- **Goal:** make it usable by real people (this is what turns a repo into a tool).
- **Implement:** pybind11/nanobind bindings: `Engine.add()`, `.search()`, `.save()/.load()`; a CLI: `engine ingest ./docs`, `engine query "..."`. Package for `pip install` (wheels for Linux/macOS/Windows). Write a 5-line quickstart.
- **Need:** cibuildwheel or similar for wheels; a docs page with the quickstart.
- **Test:** from a clean venv, `pip install`, ingest a folder, query, get results — on all three OSes (CI matrix); the 5-line quickstart works verbatim.
- **Done:** a stranger can `pip install` and get value in under five minutes. **This is the minimum shippable/usable product.**

### Stage 4 — Memory semantics (week 4–5) — the differentiator
- **Goal:** the "memory" half that Mem0/Zep charge for, but local and in-process.
- **Implement:** memory records with `created_at`, `source`, `namespace` (per user/session); **recency-weighted scoring** (decay); **dedup/update** (near-duplicate detection so re-stating a fact updates rather than duplicates); `remember(text)` / `recall(query)` API; optional forgetting/eviction policy.
- **Need:** a small synthetic multi-session conversation dataset (or adapt a public long-term-memory benchmark) to test recall over "sessions."
- **Test:** across simulated sessions, `recall()` surfaces the right earlier fact; a restated-then-changed fact returns the **updated** value, not both; recency weighting demonstrably reorders results. Measure recall@k on the memory benchmark.
- **Done:** the engine remembers, updates, and forgets correctly across sessions — the capability that distinguishes it from a plain retriever.

### Stage 5 — Built-in embedder + polish (week 5–6, optional)
- **Goal:** zero-setup mode ("just give it text").
- **Implement:** bundle a llama.cpp/ONNX embedding path so callers can pass raw text; auto-download a default model (nomic or EmbeddingGemma); MRL dimension truncation option (Qwen3) to trade size for speed.
- **Need:** model download/caching logic; license check on bundled model.
- **Test:** text-in path produces vectors matching the external path within tolerance; cold-start (first run downloads model) works; memory/latency stays within target on a laptop.
- **Done:** `engine.add("some text")` works with no embedding setup.

### Stage 6 — Benchmarks + honest writeup (week 6–8, high value)
- **Goal:** credibility.
- **Implement:** a reproducible benchmark: recall/nDCG vs sqlite-vec and vs dense-only; ingestion throughput; query latency (p50/p95); memory footprint per 1M chunks; the sqlite-vec-vs-you DX comparison in prose.
- **Test:** numbers reproduce from a script; comparison is fair (same data, same embedding model).
- **Done:** README leads with a quality graph and an honest "when NOT to use this" section.

## 7. How success is measured (metrics to quote)

Retrieval quality: **recall@k, nDCG@k, MRR** on a BEIR subset. Index fidelity: usearch recall vs brute-force (≥95%). Speed: ingestion throughput (chunks/sec), query **latency p50/p95**, memory per 1M chunks. Memory correctness: recall over simulated sessions; correct update/dedup rate. DX: lines-of-code and time-to-first-result for a new user (your best marketing metric).

## 8. Scope guards (so it ships)

Start with **caller-supplied embeddings** (defer the built-in embedder). Start with **token-window chunking** (defer semantic chunking). Ship Stage 3 (pip-installable) before touching memory — a usable retriever beats an unfinished memory engine. Do **not** add a graph/knowledge-graph layer in v1 (that's where scope dies). English first.

## 9. Freelance flywheel

The open-source engine is proof you can build private, fast, on-prem retrieval/memory. The paid engagement it unlocks: "put AI over our internal documents without the data leaving our servers" — which is exactly the privacy-driven demand in your local (Munich/EU) market. Users of the OSS tool become inbound leads for custom deployment and integration work.

## 10. Suggested repo layout

```
/core        C++ engine (ingest, index, hybrid search, memory)
/bindings    pybind11/nanobind Python layer
/cli         command-line tool
/bench       BEIR harness, latency + recall scripts, graphs
/examples    5-line quickstart, a desktop-app demo, an agent-memory demo
/docs        quickstart + "when NOT to use this"
README.md    leads with the quality graph and the honest comparison
```

## 11. Key references

usearch (github.com/unum-cloud/usearch) · sqlite-vec (github.com/asg017/sqlite-vec) · BGE-M3 / bge-reranker-v2-m3 (BAAI) · Qwen3-Embedding / Qwen3-Reranker · EmbeddingGemma · nomic-embed-text · chonkie (chunking) · BEIR + ir_measures (eval) · ann-benchmarks · pybind11 / nanobind. Study for positioning: Mem0, Zep, Letta, and the `semantic-memory` Rust crate.