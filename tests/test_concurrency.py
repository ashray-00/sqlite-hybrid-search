"""Concurrent use of one Engine instance from many Python threads (ADR-11).

The C++ core gives a single RetrievalEngine a single-writer / concurrent-reader
discipline (std::shared_mutex + a read-only SQLite connection pool + WAL), and
the nanobind layer already releases the GIL on every heavy method. These smoke
tests exercise that from Python: N threads hammering search_* against one shared
engine must return exactly what a single thread would, and readers running
while a writer ingests must never crash or observe a torn result.
"""

from concurrent.futures import ThreadPoolExecutor

import sqlite_hybrid_search


def _corpus(count):
    documents = []
    embeddings = []
    for i in range(count):
        documents.append(
            {
                "id": f"doc-{i}",
                "text": f"document number {i} about foxes and dogs and boxes",
                "metadata": "source:test",
            }
        )
        # Deterministic, well-separated unit vectors on a 4-d ring.
        angle = i % 4
        vec = [0.0, 0.0, 0.0, 0.0]
        vec[angle] = 1.0
        embeddings.append(vec)
    return documents, embeddings


def test_concurrent_hybrid_searches_match_single_threaded(tmp_path):
    db_path = str(tmp_path / "concurrency_reads.sqlite3")
    engine = sqlite_hybrid_search.Engine(db_path, 4)
    documents, embeddings = _corpus(200)
    engine.add(documents, embeddings)

    queries = [
        ("foxes", [1.0, 0.0, 0.0, 0.0]),
        ("dogs", [0.0, 1.0, 0.0, 0.0]),
        ("boxes", [0.0, 0.0, 1.0, 0.0]),
        ("document", [0.0, 0.0, 0.0, 1.0]),
    ]
    expected = [[hit["document_id"] for hit in engine.search_hybrid(text, vec, 10)] for text, vec in queries]

    def run(_):
        for idx, (text, vec) in enumerate(queries):
            got = [hit["document_id"] for hit in engine.search_hybrid(text, vec, 10)]
            assert got == expected[idx]
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(200)))
    assert all(results)


def test_readers_during_writes_stay_consistent(tmp_path):
    db_path = str(tmp_path / "concurrency_rw.sqlite3")
    engine = sqlite_hybrid_search.Engine(db_path, 4)

    batches = 40
    per_batch = 10

    def writer():
        for b in range(batches):
            start = b * per_batch
            docs = []
            embs = []
            for i in range(start, start + per_batch):
                docs.append({"id": f"doc-{i}", "text": f"row {i} fox dog box", "metadata": "m"})
                v = [0.0, 0.0, 0.0, 0.0]
                v[i % 4] = 1.0
                embs.append(v)
            engine.add(docs, embs)
        return "done"

    def reader():
        last = 0
        for _ in range(400):
            count = engine.chunk_count()
            assert count >= last  # monotonic non-decreasing
            last = count
            for hit in engine.search_hybrid("fox", [1.0, 0.0, 0.0, 0.0], 5):
                assert hit["document_id"].startswith("doc-")
        return "done"

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(writer)] + [pool.submit(reader) for _ in range(4)]
        for f in futures:
            assert f.result() == "done"

    assert engine.chunk_count() == batches * per_batch
