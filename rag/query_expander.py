"""Query expansion via multi-query generation with GPT-4o-mini.

Generates 3 semantically diverse sub-queries to improve retrieval recall
across paraphrased and policy-specific language.
"""
import os
import re

from openai import OpenAI

from config import settings

_EXPANSION_PROMPT = """\
You are an academic advisor assistant helping retrieve university policy information.
Given a student's question, generate exactly 3 different search queries that will help
retrieve the most relevant policy documents from a vector database.

Rules:
- Each sub-query should focus on a different aspect of the question
- Use general academic policy terminology (e.g., "unit requirements", "GPA threshold", "graduation requirements", "financial aid eligibility", "academic standing", "course prerequisites")
- Do NOT assume any specific major — keep queries applicable across all degree programs
- Keep each sub-query to 1-2 sentences
- Output ONLY the 3 queries, one per line, numbered 1. 2. 3.

Student question: {query}

Sub-queries:"""


def expand_query(query: str) -> list[str]:
    """
    Generate 3 expanded sub-queries for a student question.
    Falls back to [query] on any API error.
    """
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.chat.completions.create(
            model=settings.metadata_model,
            messages=[
                {"role": "user", "content": _EXPANSION_PROMPT.format(query=query)}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        sub_queries = _parse_numbered_list(raw)
        if not sub_queries:
            return [query]
        return sub_queries
    except Exception as e:
        print(f"[QueryExpander] Warning: expansion failed ({e}); using original query")
        return [query]


def _parse_numbered_list(text: str) -> list[str]:
    """Extract numbered items from GPT output."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.*)", line)
        if m:
            q = m.group(1).strip()
            if q:
                lines.append(q)
    return lines
