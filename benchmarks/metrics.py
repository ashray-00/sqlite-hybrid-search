"""Pure ranking-quality metrics for the benchmark harness.

Binary relevance, no engine or I/O dependency: a ranked list of ids in, a
float in [0, 1] out. Kept separate from the harness so the math can be
unit-tested in isolation (benchmarks/test_eval.py) and reused by any
caller.
"""

from __future__ import annotations

from math import log2
from typing import Iterable, Sequence


def _as_set(relevant_ids: Iterable[str]) -> set[str]:
    return relevant_ids if isinstance(relevant_ids, (set, frozenset)) else set(relevant_ids)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the relevant ids that appear in the top-`k` retrieved.

    Returns 0.0 when nothing is relevant (an undefined ratio otherwise).
    """
    relevant = _as_set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for cid in list(retrieved_ids)[:k] if cid in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Normalised discounted cumulative gain at `k`, binary gains.

    DCG sums 1/log2(rank+1) over the relevant hits in the top-`k`; IDCG is
    the same sum for the best possible ordering (every relevant item first).
    Returns 0.0 when nothing is relevant.
    """
    relevant = _as_set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, cid in enumerate(list(retrieved_ids)[:k], start=1)
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Reciprocal rank of the first relevant hit in the top-`k` (0.0 if none)."""
    relevant = _as_set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, cid in enumerate(list(retrieved_ids)[:k], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0
