"""Benchmark evaluation: runs sample queries through the IIL pipeline
and measures accuracy against ground truth answers.

Metrics:
  - Citation match rate: % of queries where expected citations appear in response
  - Semantic similarity: cosine similarity between RAG answer and ground truth
  - Grounding rate: % of queries that passed the grounding check
  - Per-category breakdown

Usage:
  python evals/benchmark.py [--limit N] [--category CATEGORY]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_benchmark_queries(
    queries_path: Path | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    queries_path = queries_path or (ROOT / "evals" / "data" / "sample_queries.json")
    data = json.loads(queries_path.read_text())
    queries = data["queries"]

    if category:
        queries = [q for q in queries if q.get("category") == category]
    if limit:
        queries = queries[:limit]
    return queries


def compute_citation_match(
    expected_citations: list[str],
    actual_citations: list[str],
) -> float:
    """Return fraction of expected citations found in actual citations."""
    if not expected_citations:
        return 1.0
    actual_lower = " ".join(actual_citations).lower()
    matched = sum(
        1 for ec in expected_citations
        if any(part.lower() in actual_lower for part in ec.split(",")[:2])
    )
    return matched / len(expected_citations)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using OpenAI embeddings."""
    try:
        from openai import OpenAI
        from config import settings
        import numpy as np

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=[text1[:8000], text2[:8000]],
        )
        e1 = response.data[0].embedding
        e2 = response.data[1].embedding

        e1_arr = np.array(e1)
        e2_arr = np.array(e2)
        return float(
            np.dot(e1_arr, e2_arr)
            / (np.linalg.norm(e1_arr) * np.linalg.norm(e2_arr))
        )
    except Exception as e:
        print(f"  [Warning] Semantic similarity failed: {e}")
        return 0.0


def run_benchmark(
    category: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full benchmark suite and return results."""
    from pipelines.indexer import load_existing_index
    from rag.query_engine import IILQueryEngine

    print("\n" + "=" * 60)
    print("  IIL BENCHMARK EVALUATION")
    print("=" * 60)

    queries = load_benchmark_queries(category=category, limit=limit)
    print(f"\nRunning {len(queries)} queries" + (f" (category: {category})" if category else ""))

    index = load_existing_index()
    engine = IILQueryEngine(index)

    results = []
    category_scores: dict[str, list[float]] = {}

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {q['id']}: {q['query'][:70]}...")
        t0 = time.time()

        try:
            response = engine.query(q["query"], verbose=verbose)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "error": str(e),
                "citation_match": 0.0,
                "semantic_similarity": 0.0,
                "grounded": False,
                "processing_time_ms": 0,
            })
            continue

        citation_match = compute_citation_match(
            q["expected_citations"],
            response.citations,
        )
        semantic_sim = compute_semantic_similarity(
            response.raw_answer,
            q["ground_truth"],
        )

        result = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q.get("difficulty", "medium"),
            "query": q["query"],
            "ground_truth": q["ground_truth"],
            "rag_answer": response.raw_answer[:500],
            "citations": response.citations,
            "citation_match": round(citation_match, 3),
            "semantic_similarity": round(semantic_sim, 3),
            "grounded": response.grounded,
            "processing_time_ms": round(response.processing_time_ms),
        }
        results.append(result)

        cat = q["category"]
        category_scores.setdefault(cat, []).append(semantic_sim)

        print(f"  Citation match: {citation_match:.0%} | Semantic sim: {semantic_sim:.2f} | Grounded: {response.grounded}")

    # Aggregate metrics
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("\n[Benchmark] No valid results to report.")
        return {"results": results}

    avg_citation = sum(r["citation_match"] for r in valid_results) / len(valid_results)
    avg_semantic = sum(r["semantic_similarity"] for r in valid_results) / len(valid_results)
    grounding_rate = sum(1 for r in valid_results if r["grounded"]) / len(valid_results)
    high_accuracy = sum(1 for r in valid_results if r["semantic_similarity"] >= 0.85) / len(valid_results)

    summary = {
        "total_queries": len(queries),
        "valid_results": len(valid_results),
        "avg_citation_match": round(avg_citation, 3),
        "avg_semantic_similarity": round(avg_semantic, 3),
        "grounding_rate": round(grounding_rate, 3),
        "high_accuracy_rate": round(high_accuracy, 3),
        "avg_processing_time_ms": round(
            sum(r["processing_time_ms"] for r in valid_results) / len(valid_results)
        ),
        "per_category": {
            cat: round(sum(scores) / len(scores), 3)
            for cat, scores in category_scores.items()
        },
    }

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Queries evaluated:     {summary['total_queries']}")
    print(f"  Citation match rate:   {summary['avg_citation_match']:.1%}")
    print(f"  Semantic similarity:   {summary['avg_semantic_similarity']:.2f}")
    print(f"  Grounding rate:        {summary['grounding_rate']:.1%}")
    print(f"  High-accuracy queries: {summary['high_accuracy_rate']:.1%}")
    print(f"  Avg response time:     {summary['avg_processing_time_ms']}ms")
    print("\n  Per-category semantic similarity:")
    for cat, score in sorted(summary["per_category"].items()):
        print(f"    {cat:<25} {score:.2f}")
    print("=" * 60)

    # Save results
    out_path = ROOT / "evals" / "data" / "benchmark_results.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )
    print(f"\n  Results saved to: {out_path}")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IIL benchmark evaluation")
    parser.add_argument("--category", help="Filter by category", default=None)
    parser.add_argument("--limit", type=int, help="Limit number of queries", default=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose pipeline output")
    args = parser.parse_args()

    run_benchmark(category=args.category, limit=args.limit, verbose=args.verbose)
