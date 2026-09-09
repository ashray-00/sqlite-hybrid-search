// Native retrieval micro-benchmark: latency + memory with zero Python in
// the loop, as a cross-check on benchmarks/run_eval.py's figures.
//
//   run_benchmarks --dim 64 --docs 20000 --queries 200 --output out.json
//
// Builds a random single-chunk corpus, ingests it, then times warm dense
// and hybrid queries (p50/p95/p99) and reports ingestion throughput, the
// on-disk SQLite size, and peak resident set size. Emits a small JSON
// object; run_eval.py merges it under "native_benchmark".

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "retrieval_engine/retrieval_engine.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::size_t dim = 64;
    std::size_t docs = 20000;
    std::size_t queries = 200;
    std::string output = "native_results.json";
};

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc - 1; ++i) {
        const std::string flag = argv[i];
        const std::string value = argv[i + 1];
        if (flag == "--dim") args.dim = std::stoul(value);
        else if (flag == "--docs") args.docs = std::stoul(value);
        else if (flag == "--queries") args.queries = std::stoul(value);
        else if (flag == "--output") args.output = value;
    }
    return args;
}

std::vector<float> RandomUnitVector(std::mt19937& rng, std::size_t dim) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> v(dim);
    double norm_sq = 0.0;
    for (float& x : v) {
        x = gauss(rng);
        norm_sq += static_cast<double>(x) * x;
    }
    const float inv_norm = static_cast<float>(1.0 / std::sqrt(std::max(norm_sq, 1e-12)));
    for (float& x : v) x *= inv_norm;
    return v;
}

double PeakRssMb() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    // ru_maxrss: bytes on macOS, kibibytes on Linux.
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

double Percentile(std::vector<double>& sorted_ms, double pct) {
    if (sorted_ms.empty()) return 0.0;
    if (sorted_ms.size() == 1) return sorted_ms.front();
    const double rank = pct / 100.0 * static_cast<double>(sorted_ms.size() - 1);
    const auto low = static_cast<std::size_t>(rank);
    const std::size_t high = std::min(low + 1, sorted_ms.size() - 1);
    return sorted_ms[low] + (sorted_ms[high] - sorted_ms[low]) * (rank - static_cast<double>(low));
}

struct LatencyStats {
    double p50 = 0.0, p95 = 0.0, p99 = 0.0, mean = 0.0, qps = 0.0;
};

template <typename QueryFn>
LatencyStats MeasureWarm(const std::vector<std::vector<float>>& query_vecs, QueryFn&& run) {
    for (const auto& q : query_vecs) run(q);  // warm-up

    std::vector<double> samples_ms;
    samples_ms.reserve(query_vecs.size() * 5);
    for (int pass = 0; pass < 5; ++pass) {
        for (const auto& q : query_vecs) {
            const auto start = Clock::now();
            run(q);
            samples_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        }
    }
    std::sort(samples_ms.begin(), samples_ms.end());

    double total_ms = 0.0;
    for (double s : samples_ms) total_ms += s;
    LatencyStats stats;
    stats.p50 = Percentile(samples_ms, 50);
    stats.p95 = Percentile(samples_ms, 95);
    stats.p99 = Percentile(samples_ms, 99);
    stats.mean = total_ms / static_cast<double>(samples_ms.size());
    stats.qps = total_ms > 0.0 ? static_cast<double>(samples_ms.size()) / (total_ms / 1000.0) : 0.0;
    return stats;
}

void WriteJson(const std::string& path, const Args& args, double ingest_docs_per_sec, double db_size_mb,
               double peak_rss_mb, const LatencyStats& dense, const LatencyStats& hybrid) {
    std::ofstream out(path);
    out << "{\n"
        << "  \"docs\": " << args.docs << ",\n"
        << "  \"queries\": " << args.queries << ",\n"
        << "  \"dim\": " << args.dim << ",\n"
        << "  \"index_docs_per_sec\": " << ingest_docs_per_sec << ",\n"
        << "  \"db_size_mb\": " << db_size_mb << ",\n"
        << "  \"peak_memory_mb\": " << peak_rss_mb << ",\n";
    const auto emit = [&out](const char* name, const LatencyStats& s, bool trailing_comma) {
        out << "  \"" << name << "\": {"
            << "\"latency_p50_ms\": " << s.p50 << ", "
            << "\"latency_p95_ms\": " << s.p95 << ", "
            << "\"latency_p99_ms\": " << s.p99 << ", "
            << "\"latency_mean_ms\": " << s.mean << ", "
            << "\"throughput_qps\": " << s.qps << "}" << (trailing_comma ? ",\n" : "\n");
    };
    emit("dense", dense, true);
    emit("hybrid", hybrid, false);
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    std::mt19937 rng(2026);
    const std::filesystem::path db_path =
        std::filesystem::temp_directory_path() / "retrieval_native_bench.sqlite3";
    std::filesystem::remove(db_path);

    try {
        retrieval_engine::RetrievalEngine engine(db_path.string(), args.dim);

        std::vector<retrieval_engine::DocumentInput> documents;
        documents.reserve(args.docs);
        for (std::size_t i = 0; i < args.docs; ++i) {
            retrieval_engine::DocumentChunkInput chunk;
            chunk.text = "doc " + std::to_string(i) + " token" + std::to_string(i % 997) +
                         " token" + std::to_string(i % 131);
            chunk.embedding = RandomUnitVector(rng, args.dim);
            chunk.start_token = 0;
            chunk.end_token = 4;

            retrieval_engine::DocumentInput doc;
            doc.document_id = "doc" + std::to_string(i);
            doc.chunks.push_back(std::move(chunk));
            documents.push_back(std::move(doc));
        }

        const auto ingest_start = Clock::now();
        engine.add_documents(documents);
        const double ingest_s =
            std::chrono::duration<double>(Clock::now() - ingest_start).count();
        const double ingest_docs_per_sec =
            ingest_s > 0.0 ? static_cast<double>(args.docs) / ingest_s : 0.0;

        std::vector<std::vector<float>> query_vecs;
        query_vecs.reserve(args.queries);
        for (std::size_t i = 0; i < args.queries; ++i) query_vecs.push_back(RandomUnitVector(rng, args.dim));

        const LatencyStats dense = MeasureWarm(query_vecs, [&](const std::vector<float>& q) {
            engine.search_dense(q, 10);
        });
        const LatencyStats hybrid = MeasureWarm(query_vecs, [&](const std::vector<float>& q) {
            engine.search_hybrid("token1 token2", q, 10);
        });

        const double db_size_mb =
            static_cast<double>(std::filesystem::file_size(db_path)) / 1'000'000.0;

        WriteJson(args.output, args, ingest_docs_per_sec, db_size_mb, PeakRssMb(), dense, hybrid);
        std::filesystem::remove(db_path);
        std::cout << "native benchmark wrote " << args.output << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "native benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
