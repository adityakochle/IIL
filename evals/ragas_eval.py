"""Ragas evaluation: measures Faithfulness, Answer Relevancy, and Context Precision.

Uses the Ragas framework to evaluate the IIL pipeline against the sample query dataset.

Metrics (as defined by Ragas):
  - Faithfulness (0-1): Are all claims in the answer grounded in the retrieved context?
  - Answer Relevancy (0-1): Does the answer directly address the question?
  - Context Precision (0-1): Is the retrieved context signal-to-noise ratio high?

Usage:
  python evals/ragas_eval.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def build_ragas_dataset(limit: int | None = None) -> list[dict]:
    """
    Run the IIL pipeline on benchmark queries and collect data for Ragas evaluation.
    Returns a list of dicts with question, answer, contexts, and ground_truth.
    """
    from pipelines.indexer import load_existing_index
    from rag.query_engine import IILQueryEngine
    from evals.benchmark import load_benchmark_queries

    queries = load_benchmark_queries(limit=limit)

    index = load_existing_index()
    engine = IILQueryEngine(index)

    dataset_rows = []
    print(f"\n[Ragas] Generating {len(queries)} pipeline responses...")

    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q['id']}...")
        try:
            response = engine.query(q["query"])

            # Ragas needs the raw context strings
            contexts = [c.parent_text for c in engine.retriever.retrieve(q["query"])]

            dataset_rows.append({
                "question": q["query"],
                "answer": response.raw_answer,
                "contexts": contexts[:5],  # Ragas recommends ≤5 contexts
                "ground_truth": q["ground_truth"],
                "query_id": q["id"],
                "category": q["category"],
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Save the raw dataset for inspection / re-use
    out_path = ROOT / "evals" / "data" / "ragas_dataset.json"
    out_path.write_text(json.dumps(dataset_rows, indent=2))
    print(f"[Ragas] Dataset saved to {out_path}")

    return dataset_rows


def run_ragas_evaluation(limit: int | None = None) -> dict:
    """Run Ragas metrics on the IIL pipeline output."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
        from ragas.metrics._context_recall import context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from config import settings

        _RAGAS_AVAILABLE = True
    except ImportError as e:
        print(f"[Ragas] Import error: {e}")
        print("[Ragas] Install with: pip install ragas datasets langchain-openai")
        return {}

    dataset_rows = build_ragas_dataset(limit=limit)
    if not dataset_rows:
        print("[Ragas] No data to evaluate.")
        return {}

    print(f"\n[Ragas] Running evaluation on {len(dataset_rows)} examples...")

    dataset = Dataset.from_list(dataset_rows)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key),
        embeddings=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        ),
    )

    scores = {
        "faithfulness": round(results["faithfulness"], 3),
        "answer_relevancy": round(results["answer_relevancy"], 3),
        "context_precision": round(results["context_precision"], 3),
        "num_examples": len(dataset_rows),
    }

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Faithfulness:      {scores['faithfulness']:.1%}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.1%}")
    print(f"  Context Precision: {scores['context_precision']:.1%}")
    print(f"  Examples:          {scores['num_examples']}")
    print("=" * 60)

    out_path = ROOT / "evals" / "data" / "ragas_results.json"
    out_path.write_text(json.dumps(scores, indent=2))
    print(f"\n  Results saved to: {out_path}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on IIL pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of evaluation examples", default=None)
    args = parser.parse_args()

    run_ragas_evaluation(limit=args.limit)
