"""Strict Grounding Pipeline.

Prevents hallucinations by verifying that retrieved contexts have sufficient
relevance before passing them to the LLM. If contexts don't meet the threshold,
returns a "Verified Source Not Found" response rather than allowing the LLM
to speculate.
"""
from dataclasses import dataclass, field

from config import settings
from rag.retriever import RetrievedContext

NOT_FOUND_RESPONSE = (
    "⚠️ VERIFIED SOURCE NOT FOUND\n\n"
    "The IIL system could not locate a verified policy document that directly "
    "addresses your specific query. This may be because:\n"
    "- The policy you're asking about falls outside the current indexed document set\n"
    "- The query requires cross-department information not yet indexed\n"
    "- The relevant policy was updated within the last 24 hours (outside refresh cycle)\n\n"
    "Recommended action: Please contact your academic advisor directly or visit "
    "the SFSU Registrar's Office for authoritative guidance on this specific question."
)


@dataclass
class GroundingResult:
    verified: bool
    contexts: list[RetrievedContext] = field(default_factory=list)
    max_score: float = 0.0
    warning: str | None = None


def verify_grounding(
    contexts: list[RetrievedContext],
    threshold: float | None = None,
) -> GroundingResult:
    """
    Verify that at least one retrieved context meets the relevance threshold.

    Args:
        contexts: Reranked contexts from the retriever
        threshold: Minimum score to pass grounding (defaults to settings value)

    Returns:
        GroundingResult with verified=True if any context passes the threshold
    """
    threshold = threshold if threshold is not None else settings.grounding_threshold

    if not contexts:
        return GroundingResult(
            verified=False,
            contexts=[],
            max_score=0.0,
            warning="No contexts retrieved.",
        )

    max_score = max(ctx.retrieval_score for ctx in contexts)

    if max_score < threshold:
        return GroundingResult(
            verified=False,
            contexts=contexts,
            max_score=max_score,
            warning=(
                f"Maximum relevance score {max_score:.3f} is below "
                f"grounding threshold {threshold:.3f}."
            ),
        )

    # Filter to contexts that pass the threshold
    verified_contexts = [ctx for ctx in contexts if ctx.retrieval_score >= threshold]

    return GroundingResult(
        verified=True,
        contexts=verified_contexts,
        max_score=max_score,
    )
