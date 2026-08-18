"""Measure end-to-end retrieval latency (embed + Qdrant search) against the
50ms budget defined in app/config.py.

Usage:
    python -m app.benchmark [n_queries]
"""
import statistics
import sys
import time

from app.config import settings
from app.retrieval.embedder import get_embedder
from app.retrieval.qdrant_client import get_qdrant


QUERIES = [
    "What is the capital of India?",
    "Who is the Prime Minister of India?",
    "What is the population of India?",
    "When did India get independence?",
    "What are the official languages of India?",
    "Which is the largest state in India?",
    "What is the currency of India?",
    "Who wrote the Indian national anthem?",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Warming up (model load + first inference)...")
    embedder = get_embedder()
    qdrant = get_qdrant()
    
    # Warmup embed
    _ = embedder.embed_single("warmup query")
    
    # Warmup search (if collections have data)
    try:
        warmup_vector = _[0].tolist() if hasattr(_[0], "tolist") else _[0]
        qdrant.search(warmup_vector, "hi", limit=5)
    except Exception:
        pass

    total_ms, embed_ms, search_ms = [], [], []
    
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        
        # Embed
        start = time.perf_counter()
        query_vector = embedder.embed_single(query)
        embed_time = (time.perf_counter() - start) * 1000
        
        # Search (try Hindi first since it has data)
        start = time.perf_counter()
        try:
            results = qdrant.search(query_vector.tolist(), "hi", limit=5)
        except Exception as e:
            results = []
        search_time = (time.perf_counter() - start) * 1000
        
        total_ms.append(embed_time + search_time)
        embed_ms.append(embed_time)
        search_ms.append(search_time)

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    budget_ms = getattr(settings, 'retrieval_latency_budget_ms', 50)
    print(f"\nLatency budget: {budget_ms}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= budget_ms:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- see README 'Tuning latency' section")
        sys.exit(1)


if __name__ == "__main__":
    main()