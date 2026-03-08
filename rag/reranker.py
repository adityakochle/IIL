"""Reranker using Cohere rerank-english-v3.0.

Falls back to a cross-encoder (sentence-transformers) if no Cohere API key is set.
"""
from __future__ import annotations

from config import settings
from rag.retriever import RetrievedContext


def rerank(
    query: str,
    contexts: list[RetrievedContext],
    top_n: int | None = None,
) -> list[RetrievedContext]:
    """
    Rerank retrieved contexts using Cohere or a local cross-encoder fallback.

    Returns top_n contexts sorted by relevance.
    """
    top_n = top_n or settings.top_n_rerank
    if not contexts:
        return []

    if settings.cohere_api_key:
        return _cohere_rerank(query, contexts, top_n)
    return _crossencoder_rerank(query, contexts, top_n)


def _cohere_rerank(
    query: str,
    contexts: list[RetrievedContext],
    top_n: int,
) -> list[RetrievedContext]:
    """Rerank using Cohere API."""
    try:
        import cohere

        co = cohere.Client(api_key=settings.cohere_api_key)
        documents = [ctx.parent_text[:4096] for ctx in contexts]  # Cohere limit

        results = co.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked: list[RetrievedContext] = []
        for r in results.results:
            ctx = contexts[r.index]
            ctx.score = r.relevance_score
            reranked.append(ctx)

        print(f"[Reranker] Cohere rerank: {len(contexts)} → {len(reranked)} contexts")
        return reranked

    except Exception as e:
        print(f"[Reranker] Cohere failed ({e}); falling back to cross-encoder")
        return _crossencoder_rerank(query, contexts, top_n)


def _crossencoder_rerank(
    query: str,
    contexts: list[RetrievedContext],
    top_n: int,
) -> list[RetrievedContext]:
    """Rerank using a local sentence-transformers cross-encoder."""
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, ctx.parent_text[:512]) for ctx in contexts]
        scores = model.predict(pairs)

        import math
        for ctx, score in zip(contexts, scores):
            ctx.score = 1.0 / (1.0 + math.exp(-float(score)))  # sigmoid → [0, 1]

        reranked = sorted(contexts, key=lambda c: c.score, reverse=True)[:top_n]
        print(f"[Reranker] Cross-encoder rerank: {len(contexts)} → {len(reranked)} contexts")
        return reranked

    except Exception as e:
        print(f"[Reranker] Cross-encoder failed ({e}); returning top-{top_n} by original score")
        return sorted(contexts, key=lambda c: c.score, reverse=True)[:top_n]
