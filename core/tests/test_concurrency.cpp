// Concurrency contract: one RetrievalEngine instance safely serves many
// reader threads (search_*, embed, chunk_count) with at most one writer
// (add_*, load_embedding_model) -- no external locking. Run under
// ThreadSanitizer to verify race-freedom:
//   cmake -B build-tsan -DRETRIEVAL_ENGINE_SANITIZE=thread -DRETRIEVAL_ENGINE_WITH_ONNX=OFF
//   ctest --test-dir build-tsan -R Concurrency --repeat until-fail:20
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "retrieval_engine/retrieval_engine.hpp"

namespace {

constexpr std::size_t kDim = 8;

void RemoveArtifacts(const std::string& path) {
    std::error_code ec;
    for (const char* suffix : {"", ".usearch", ".usearch.tmp", "-wal", "-shm"}) {
        std::filesystem::remove(path + suffix, ec);
    }
}

// Sweeps the given base paths (db files, model fixtures, their sidecars)
// both on construction and on destruction. Declared before the engine under
// test so its cleanup runs *after* the engine -- and every connection it
// owns -- has been destroyed.
class ArtifactGuard {
public:
    explicit ArtifactGuard(std::vector<std::string> paths) : paths_(std::move(paths)) { Sweep(); }
    ArtifactGuard(const ArtifactGuard&) = delete;
    ArtifactGuard& operator=(const ArtifactGuard&) = delete;
    ~ArtifactGuard() { Sweep(); }

private:
    void Sweep() {
        for (const auto& path : paths_) RemoveArtifacts(path);
    }
    std::vector<std::string> paths_;
};

// `count` single-chunk docs; embedding is 0.1 everywhere except one
// dimension bumped by the doc index, so nearest-neighbour is unambiguous.
std::vector<retrieval_engine::DocumentInput> MakeDocuments(std::size_t count, std::size_t start = 0) {
    std::vector<retrieval_engine::DocumentInput> documents;
    for (std::size_t i = start; i < start + count; ++i) {
        retrieval_engine::DocumentChunkInput chunk;
        chunk.text = "document number " + std::to_string(i) + " topic " + std::to_string(i % 7);
        chunk.embedding = std::vector<float>(kDim, 0.1f);
        chunk.embedding[i % kDim] = 1.0f + static_cast<float>(i % 97);
        chunk.start_token = 0;
        chunk.end_token = 5;

        retrieval_engine::DocumentInput document;
        document.document_id = "doc-" + std::to_string(i);
        document.chunks.push_back(std::move(chunk));
        documents.push_back(std::move(document));
    }
    return documents;
}

std::vector<std::vector<float>> QueryVectors(std::size_t n) {
    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> q(kDim, 0.1f);
        q[i % kDim] = 1.0f + static_cast<float>((i * 13) % 97);
        queries.push_back(std::move(q));
    }
    return queries;
}

std::vector<std::string> Ids(const std::vector<retrieval_engine::ChunkSearchResult>& hits) {
    std::vector<std::string> ids;
    for (const auto& h : hits) ids.push_back(h.document_id);
    return ids;
}

// A few readers past the pool size: the read-connection pool and the
// usearch search-slot pool are each sized to max(1, hardware_concurrency()),
// so "+ 2" guarantees at least two readers hit BlockingPool's block-and-wake
// path (condvar wait + notify_one on checkout) on every host. Kept modest on
// purpose -- a large reader swarm just starves the writer under a
// reader-preference rwlock without testing anything new.
std::size_t ReaderCount() { return std::max<std::size_t>(2, std::thread::hardware_concurrency()) + 2; }

void WriteMockModel(const std::string& path, std::size_t dim) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1\n";
    out << "dim=" << dim << "\n";
}

}  // namespace

TEST(Concurrency, ConcurrentDenseAndHybridSearchesMatchSingleThreaded) {
    const std::string db = "concurrency_test_search.sqlite3";
    const ArtifactGuard guard({db});

    retrieval_engine::RetrievalEngine engine(db, kDim);
    engine.add_documents(MakeDocuments(240));

    const auto queries = QueryVectors(12);
    std::vector<std::vector<std::string>> expected_dense;
    std::vector<std::vector<std::string>> expected_hybrid;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        expected_dense.push_back(Ids(engine.search_dense(queries[i], 5)));
        expected_hybrid.push_back(Ids(engine.search_hybrid("document number " + std::to_string(i), queries[i], 5)));
    }

    std::atomic<bool> mismatch{false};
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < ReaderCount(); ++t) {
        threads.emplace_back([&] {
            for (int iter = 0; iter < 40 && !mismatch; ++iter) {
                for (std::size_t i = 0; i < queries.size(); ++i) {
                    if (Ids(engine.search_dense(queries[i], 5)) != expected_dense[i]) mismatch = true;
                    if (Ids(engine.search_hybrid("document number " + std::to_string(i), queries[i], 5)) !=
                        expected_hybrid[i])
                        mismatch = true;
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_FALSE(mismatch) << "a concurrent search returned a different result than the single-threaded baseline";
}

TEST(Concurrency, ReadersDuringWritesStayConsistent) {
    const std::string db = "concurrency_test_rw.sqlite3";
    const ArtifactGuard guard({db});

    retrieval_engine::RetrievalEngine engine(db, kDim);
    engine.add_documents(MakeDocuments(50));  // seed

    constexpr std::size_t kBatches = 40;
    constexpr std::size_t kBatchSize = 15;
    const std::size_t total = 50 + kBatches * kBatchSize;

    std::atomic<bool> done{false};
    std::atomic<bool> failure{false};
    std::atomic<std::size_t> last_count{50};

    std::vector<std::thread> readers;
    for (std::size_t t = 0; t < ReaderCount(); ++t) {
        readers.emplace_back([&] {
            const auto queries = QueryVectors(6);
            while (!done) {
                const std::size_t c = engine.chunk_count();
                if (c < last_count.load()) failure = true;  // must never go backwards
                last_count = c;
                for (const auto& q : queries) {
                    for (const auto& hit : engine.search_hybrid("document number", q, 5)) {
                        // every returned id must be one we actually ingested
                        if (hit.document_id.rfind("doc-", 0) != 0) failure = true;
                    }
                }
                // Yield between passes so the writer is not starved of the
                // exclusive lock under a reader-preference shared_mutex.
                std::this_thread::yield();
            }
        });
    }

    for (std::size_t b = 0; b < kBatches; ++b) {
        engine.add_documents(MakeDocuments(kBatchSize, 50 + b * kBatchSize));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    done = true;
    for (auto& th : readers) th.join();

    EXPECT_FALSE(failure);
    EXPECT_EQ(engine.chunk_count(), total);
}

TEST(Concurrency, ConcurrentEmbedAndSearchTextWithMockModel) {
    const std::string db = "concurrency_test_text.sqlite3";
    const std::string model = "concurrency_test_mock_model.txt";
    const ArtifactGuard guard({db, model});
    WriteMockModel(model, kDim);

    retrieval_engine::RetrievalEngine engine(db, kDim);
    engine.load_embedding_model(model);
    engine.add_documents(MakeDocuments(160));

    const std::vector<std::string> phrases = {"document number 3", "topic 4", "number 42 topic 0", "document"};
    std::vector<std::vector<float>> expected_embed;
    std::vector<std::vector<std::string>> expected_text;
    for (const auto& p : phrases) {
        expected_embed.push_back(engine.embed(p));
        expected_text.push_back(Ids(engine.search_text(p, 5)));
    }

    std::atomic<bool> mismatch{false};
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < ReaderCount(); ++t) {
        threads.emplace_back([&] {
            for (int iter = 0; iter < 30 && !mismatch; ++iter) {
                for (std::size_t i = 0; i < phrases.size(); ++i) {
                    if (engine.embed(phrases[i]) != expected_embed[i]) mismatch = true;
                    if (Ids(engine.search_text(phrases[i], 5)) != expected_text[i]) mismatch = true;
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_FALSE(mismatch);
}

TEST(Concurrency, LoadModelConcurrentWithReaders) {
    const std::string db = "concurrency_test_loadmodel.sqlite3";
    const std::string model = "concurrency_test_loadmodel_mock.txt";
    const ArtifactGuard guard({db, model});
    WriteMockModel(model, kDim);

    retrieval_engine::RetrievalEngine engine(db, kDim);
    engine.add_documents(MakeDocuments(120));

    std::atomic<bool> done{false};
    std::atomic<bool> reverted{false};
    std::atomic<bool> failure{false};

    std::vector<std::thread> readers;
    for (std::size_t t = 0; t < ReaderCount(); ++t) {
        readers.emplace_back([&] {
            bool seen_model = false;
            while (!done) {
                const bool has = engine.has_embedding_model();
                if (has) {
                    seen_model = true;
                    try {
                        (void)engine.search_text("document number 5", 3);
                    } catch (const std::exception&) {
                        failure = true;
                    }
                } else if (seen_model) {
                    reverted = true;  // must never go true -> false
                }
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    engine.load_embedding_model(model);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    done = true;
    for (auto& th : readers) th.join();

    EXPECT_FALSE(failure);
    EXPECT_FALSE(reverted);
    EXPECT_TRUE(engine.has_embedding_model());
}

// Informational: proves the read path actually parallelises rather than
// serialising on the connection pool. Timing-based, so DISABLED by default
// (CI-flaky); run locally in a Release build.
TEST(Concurrency, DISABLED_ReaderScalingIsParallel) {
    const std::string db = "concurrency_test_scaling.sqlite3";
    const ArtifactGuard guard({db});
    retrieval_engine::RetrievalEngine engine(db, kDim);
    engine.add_documents(MakeDocuments(2000));
    const auto queries = QueryVectors(16);

    auto run = [&](std::size_t n_threads, int iters_per_thread) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < n_threads; ++t) {
            threads.emplace_back([&] {
                for (int i = 0; i < iters_per_thread; ++i)
                    for (const auto& q : queries) (void)engine.search_hybrid("document number", q, 5);
            });
        }
        for (auto& th : threads) th.join();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    const std::size_t p = std::max<std::size_t>(2, std::thread::hardware_concurrency());
    const double serial = run(1, static_cast<int>(p) * 20);
    const double parallel = run(p, 20);
    EXPECT_LT(parallel, serial / 1.5) << "serial=" << serial << "s parallel=" << parallel << "s (p=" << p << ")";
}
