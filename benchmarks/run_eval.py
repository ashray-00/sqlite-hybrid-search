"""Benchmark runner: evaluate every retrieval approach across dataset sizes
and write structured metrics to benchmarks/results.json.

    python benchmarks/run_eval.py [--sizes N ...] [--queries N]
                                  [--dim N] [--decay F] [--output PATH]
                                  [--native-bin PATH | --no-native]

Quality (Recall@10 / nDCG@10 / MRR@10), warm + cold latency (p50/p95/p99),
and memory / storage footprint come from benchmarks/harness.py. If the
compiled C++ micro-benchmark (benchmarks/run_benchmarks.cpp) is present its
pure-native latency/RSS numbers are merged in under "native_benchmark" for
cross-checking the Python-driven figures.

The prose report (BENCHMARKS.md) is written by hand from this output.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from harness import APPROACHES, benchmark_size

# Dataset sizes the shipped BENCHMARKS.md report is generated at.
REPORT_DATASET_SIZES = (1_000, 10_000, 100_000)

_DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
_PRIMARY_APPROACH = "hybrid"
_NATIVE_BIN_CANDIDATES = (
    "build/benchmarks/run_benchmarks",
    "build/bin/run_benchmarks",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(REPORT_DATASET_SIZES))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--decay", type=float, default=0.1, help="recency decay lambda for hybrid_decay")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--native-bin", default=None, help="path to the compiled run_benchmarks binary")
    parser.add_argument("--no-native", action="store_true", help="skip the C++ micro-benchmark even if built")
    return parser.parse_args(argv)


def _find_native_bin(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in _NATIVE_BIN_CANDIDATES:
        path = os.path.join(repo_root, candidate)
        if os.path.exists(path):
            return path
    return None


def _run_native(native_bin: str, dim: int, tmp_dir: str) -> dict | None:
    out_path = os.path.join(tmp_dir, "native_results.json")
    proc = subprocess.run(
        [native_bin, "--dim", str(dim), "--docs", "20000", "--queries", "200", "--output", out_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not os.path.exists(out_path):
        return {"error": proc.stderr.strip() or "native benchmark did not produce output"}
    with open(out_path, encoding="utf-8") as handle:
        return json.load(handle)


def _summary(results: list[dict], sizes: list[int]) -> dict:
    """Top-level convenience copy: the primary approach at the largest size."""
    largest = max(sizes)
    for record in results:
        if record["approach"] == _PRIMARY_APPROACH and record["dataset_size"] == largest:
            return {
                key: record[key]
                for key in (
                    "recall_at_10",
                    "ndcg_at_10",
                    "latency_p50_ms",
                    "latency_p95_ms",
                    "latency_p99_ms",
                    "peak_memory_mb",
                )
            }
    return {}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sizes = sorted(set(args.sizes))

    document: dict = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "engine_dim": args.dim,
            "queries_per_size": args.queries,
            "decay_lambda": args.decay,
            "k": 10,
        },
        "dataset_sizes": sizes,
        "approaches": list(APPROACHES),
        "results": [],
    }

    with tempfile.TemporaryDirectory(prefix="retrieval_bench_") as tmp_dir:
        for size in sizes:
            started = time.perf_counter()
            print(f"[run_eval] dataset_size={size} ...", flush=True)
            document["results"].extend(
                benchmark_size(size, args.queries, args.dim, args.decay, tmp_dir)
            )
            print(f"[run_eval] dataset_size={size} done in {time.perf_counter() - started:.1f}s", flush=True)

        native_bin = None if args.no_native else _find_native_bin(args.native_bin)
        if native_bin:
            print(f"[run_eval] native micro-benchmark: {native_bin}", flush=True)
            document["native_benchmark"] = _run_native(native_bin, args.dim, tmp_dir)

    document.update(_summary(document["results"], sizes))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[run_eval] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
