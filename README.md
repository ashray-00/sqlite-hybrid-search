# Embeddable Retrieval & Memory Engine

A local-first, in-process retrieval + memory engine in C++17, with Python
bindings via nanobind. See `BUILD_PLAN.md` for the full plan and
`docs/DECISIONS.md` for the build log.

## Quickstart

```python
import retrieval_engine

engine = retrieval_engine.Engine("my_index.sqlite3", dim=4)
engine.add(documents=[{"id": "doc-1", "text": "the quick brown fox"}],
           embeddings=[[1.0, 0.0, 0.0, 0.0]])
print(engine.search([1.0, 0.0, 0.0, 0.0], top_k=1))
```

Or from the command line, over a folder of `.txt` files (see the CLI's own
`--help` for details; it uses a placeholder hashing-trick embedding, not a
real model -- see `python/retrieval_engine/cli.py`'s docstring):

```console
$ pip install -e .
$ engine ingest ./docs
$ engine query "your search text"
```

**Status: Stage 3 (Python bindings + CLI).** Hybrid (dense + sparse, RRF-fused)
retrieval with an explainable score trail is implemented and tested in C++
(Stages 0-2), now exposed via nanobind bindings, a Python package, and this
CLI. Chunking is token-window-based (no real tokenizer yet); embeddings are
caller-supplied everywhere except the CLI's placeholder, no real embedding
model is bundled yet (Stage 5); memory semantics (recency, dedup, forgetting)
don't exist yet (Stage 4). Read this file as the pressure-test called for in
`BUILD_PLAN.md` section 3: before writing more of this, be honest about
whether it earns its place next to the obvious shortcut -- the comparison
below is about what's being built *toward*, not a claim about what's
finished.

## Why not just sqlite-vec?

[`sqlite-vec`](https://github.com/asg017/sqlite-vec) is a real, legitimate
answer to "I want an embeddable vector index with no service to run." It is
not a toy, and this project does not compete with it at the index layer --
see `BUILD_PLAN.md` section 3. Here is roughly what using it directly looks
like, in Python:

```python
import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32

DIM = 4

db = sqlite3.connect("notes.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

db.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks
    USING vec0(embedding float[{DIM}])
""")

def add_chunk(rowid: int, embedding: list[float]) -> None:
    db.execute(
        "INSERT INTO chunks(rowid, embedding) VALUES (?, ?)",
        [rowid, serialize_float32(embedding)],
    )
    db.commit()

def search(query_embedding: list[float], k: int = 5):
    return db.execute(
        """
        SELECT rowid, distance FROM chunks
        WHERE embedding MATCH ?
        ORDER BY distance LIMIT ?
        """,
        [serialize_float32(query_embedding), k],
    ).fetchall()

add_chunk(1, [0.10, 0.20, 0.30, 0.40])
add_chunk(2, [0.20, 0.10, 0.40, 0.30])

for rowid, distance in search([0.12, 0.19, 0.29, 0.41], k=1):
    print(rowid, distance)
```

That's roughly 30 lines, and it works: dense ANN search, persisted in one
SQLite file, no server. If that's all you need, use it -- it's a fine tool
and this project would rather admit that than pretend otherwise.

### What those 30 lines don't give you

- **Nowhere for the text or metadata to live.** `vec0` stores the vector.
  The document text, source, and timestamps need a second table you design,
  populate, and join back to `chunks.rowid` yourself, in every project that
  uses this pattern.
- **No chunking.** You bring already-chunked text; the chunking strategy
  (window size, overlap) is yours to write and re-write per project.
- **Dense-only retrieval.** No BM25/FTS5 integration, no fusion. Exact
  matches on a name, an ID, or a rare term -- the case dense embeddings are
  worst at -- get no help here (`BUILD_PLAN.md` section 2).
- **No reranking, no explainability.** No cross-encoder pass, no way to see
  *why* a result ranked where it did (dense score vs. sparse score vs. fused
  rank vs. rerank score).
- **No memory semantics.** No recency weighting, no dedup-or-update when a
  fact is restated, no namespaces, no forgetting policy. This is the part
  Mem0/Zep/Letta charge for, and `vec0` has no opinion on any of it.
- **No consistency discipline between the two tables.** The snippet above
  has no story for what happens if a crash lands between writing to `chunks`
  and writing to your metadata table. Getting the write order wrong is easy
  and easy to miss in one-off glue code -- this project's own Stage 0 review
  caught exactly that bug in `RetrievalEngine::add_vector()` on day one (see
  `docs/DECISIONS.md`, "Decision: write SQLite before usearch").
- **No packaging.** No `pip install`, no CLI, no quickstart. Every team that
  wants this pattern re-derives and re-tests it from scratch.

### The honest part

"Embeddable" is not the differentiator -- `sqlite-vec` is embeddable too, and
pretending otherwise would be the strawman `BUILD_PLAN.md` section 3
explicitly warns against. The actual gap is that nobody ships the *whole*
pipeline (chunk -> hybrid dense+sparse retrieval -> RRF fusion -> rerank ->
explain) plus memory semantics (dedup, recency decay, namespaces, forgetting)
as one tested, embeddable, no-LLM-required C++ library with clean bindings.
The closest thing that does is `semantic-memory`, a Rust crate with the same
SQLite-authoritative + vector-sidecar shape this project uses -- which
validates the idea, while being Rust-only and library-level (no bindings,
no CLI) is exactly the room a well-packaged C++ + Python version can fill.

The bet this project is making: most people reach for `sqlite-vec` (or roll
their own with `usearch` directly) and then quietly rebuild the six bullet
points above, worse, once per project, because nothing packages them
together. If that bet is wrong -- if the 30 lines above really do cover 90%
of what people need -- this project isn't worth finishing. Each stage in
`BUILD_PLAN.md` is designed to test that bet against something measurable
(recall/nDCG, hybrid-vs-dense-only quality, memory-recall correctness)
rather than assert it.
