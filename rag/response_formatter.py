"""Response formatter.

Structures the raw LLM output into the demo format shown in the README:
  - TRANSFER ANALYSIS
  - MAJOR SWITCH CONSIDERATION
  - ALTERNATIVE PATH
  - CITATIONS
"""
import re
from dataclasses import dataclass, field

from rag.retriever import RetrievedContext


@dataclass
class FormattedResponse:
    raw_answer: str
    citations: list[str]
    sources: list[str]
    processing_time_ms: float = 0.0
    grounded: bool = True
    num_contexts_used: int = 0

    def to_display_text(self) -> str:
        """Render the response as rich formatted text for the CLI."""
        lines = [self.raw_answer.strip(), ""]

        if self.citations:
            lines.append("📚 CITATIONS")
            for c in self.citations:
                lines.append(f"  • {c}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "answer": self.raw_answer,
            "citations": self.citations,
            "sources": self.sources,
            "grounded": self.grounded,
            "num_contexts_used": self.num_contexts_used,
            "processing_time_ms": self.processing_time_ms,
        }


def build_citations(contexts: list[RetrievedContext]) -> list[str]:
    """Deduplicate and format citation strings from context metadata."""
    seen: set[str] = set()
    citations: list[str] = []
    for ctx in contexts:
        citation = ctx.source
        if citation and citation not in seen:
            seen.add(citation)
            if ctx.section:
                citation = f"{citation}: {ctx.section}"
            citations.append(citation)
    return citations


def build_sources(contexts: list[RetrievedContext]) -> list[str]:
    """Return unique source document names."""
    seen: set[str] = set()
    sources: list[str] = []
    for ctx in contexts:
        src = ctx.metadata.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def format_response(
    raw_answer: str,
    contexts: list[RetrievedContext],
    processing_time_ms: float = 0.0,
    grounded: bool = True,
) -> FormattedResponse:
    """Wrap the LLM answer with citation metadata."""
    citations = build_citations(contexts)
    sources = build_sources(contexts)

    return FormattedResponse(
        raw_answer=raw_answer.strip(),
        citations=citations,
        sources=sources,
        processing_time_ms=processing_time_ms,
        grounded=grounded,
        num_contexts_used=len(contexts),
    )
