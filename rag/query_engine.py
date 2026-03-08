"""Institutional Intelligence Layer — Main Query Orchestrator.

Pipeline (anonymous):
  query → expand_query → retrieve (multi-query) → rerank → ground_check
        → build_prompt → GPT-4o → format_response

Pipeline (with student_id):
  student_id → StudentStore.get() → format_for_prompt()
  query → expand_query → retrieve → rerank → ground_check
        → build_prompt (policy context + student record) → GPT-4o → format_response
"""
import os
import time

from llama_index.core import VectorStoreIndex
from openai import OpenAI

from config import settings
from rag.grounding import NOT_FOUND_RESPONSE, GroundingResult, verify_grounding
from rag.query_expander import expand_query
from rag.response_formatter import FormattedResponse, format_response
from rag.retriever import ParentDocumentRetriever, RetrievedContext
from rag.reranker import rerank

_SYSTEM_PROMPT = """\
You are the SFSU Institutional Intelligence Layer — an expert academic advisor AI.
Your role is to answer student questions about SFSU academic policies with precision
and clarity, grounded strictly in the provided policy documents.

CRITICAL RULES:
1. Base EVERY claim on the provided context. Do not fabricate policies.
2. If the context doesn't explicitly address a sub-question, say so clearly.
3. Cite specific documents and page numbers when making policy claims.
4. Structure your response clearly with labeled sections.
5. If timelines are estimated, label them as estimates and explain the reasoning.
6. Highlight important warnings (⚠️) for prerequisite chains, GPA thresholds, or deadlines.

Response format:
- Use clear section headers (e.g., 📊 TRANSFER ANALYSIS, ⚠️ CONSIDERATIONS, ✅ RECOMMENDATIONS)
- Be concise but complete
- Lead with the most critical information
- End with specific next steps the student should take
"""

_SYSTEM_PROMPT_PERSONALIZED = """\
You are the SFSU Institutional Intelligence Layer — an expert academic advisor AI.
You have been given a specific student's verified transcript record alongside policy documents.

CRITICAL RULES:
1. Ground ALL policy claims in the provided policy context — never fabricate.
2. Cross-reference the student's completed courses, GPA, and units with policy requirements.
3. Give advice SPECIFIC to this student — not generic answers. Name the exact courses
   they still need, flag their specific GPA against thresholds, calculate their remaining units.
4. Cite both the policy source AND the student's record when making claims.
5. If the student's situation is at risk (probation, SAP warning, unit ceiling), call it out explicitly.
6. Highlight ⚠️ for any warnings and ✅ for requirements already satisfied by this student.

Response format:
- Start with a brief "Student Snapshot" confirming key details (GPA, standing, units)
- Then address the question with personalized analysis
- End with a clear, numbered action plan for THIS student
"""

_USER_TEMPLATE = """\
STUDENT QUESTION:
{query}

VERIFIED POLICY CONTEXT:
{context_block}

Please provide a structured, citation-backed answer addressing the student's question.
"""

_USER_TEMPLATE_PERSONALIZED = """\
STUDENT QUESTION:
{query}

{student_block}

VERIFIED POLICY CONTEXT:
{context_block}

Using the student's actual record above, provide a personalized, specific answer to their
question. Reference their exact courses completed, GPA, and units where relevant.
"""


class IILQueryEngine:
    """The Institutional Intelligence Layer query engine."""

    def __init__(self, index: VectorStoreIndex):
        self._index = index  # kept for DegreeAuditor
        self.retriever = ParentDocumentRetriever(index)
        self._llm_client = OpenAI(api_key=settings.openai_api_key)
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

        # Lazy-loaded; avoid import cycles
        self._student_store = None
        self._degree_auditor = None

    def _get_student_store(self):
        if self._student_store is None:
            from pipelines.student_store import StudentStore
            self._student_store = StudentStore()
        return self._student_store

    def _get_degree_auditor(self):
        if self._degree_auditor is None:
            from pipelines.degree_auditor import DegreeAuditor
            self._degree_auditor = DegreeAuditor(self._index)
        return self._degree_auditor

    def query(
        self,
        user_query: str,
        student_id: str | None = None,
        verbose: bool = False,
    ) -> FormattedResponse:
        """
        Run the full RAG pipeline for a student query.

        Args:
            user_query: The student's natural language question
            student_id: Optional student ID to pull a verified transcript record.
                        When provided, the response is personalized to that student.
            verbose: Print step-by-step pipeline info

        Returns:
            FormattedResponse with answer, citations, and metadata
        """
        t_start = time.time()

        # Step 0: Resolve student record + live degree audit (if student_id provided)
        student_block: str | None = None
        if student_id:
            store = self._get_student_store()
            record = store.get(student_id)
            if record is None:
                return format_response(
                    raw_answer=(
                        f"⚠️ Student ID '{student_id}' was not found in the student records system.\n"
                        "Please verify the ID and try again, or contact the Registrar's Office."
                    ),
                    contexts=[],
                    processing_time_ms=0,
                    grounded=False,
                )
            if verbose:
                print(f"[Pipeline] Loaded student record: {record.name} ({student_id})")

            # Run degree audit: outstanding requirements come from policy docs, not the JSON
            if verbose:
                print(f"[Pipeline] Running degree audit against policy documents...")
            auditor = self._get_degree_auditor()
            audit = auditor.audit(record, verbose=verbose)
            if verbose:
                outstanding = len(audit.outstanding)
                satisfied = len(audit.satisfied)
                print(f"[Pipeline] Audit complete: {satisfied} satisfied, {outstanding} outstanding")

            # Combine transcript facts + policy-derived audit into one block
            student_block = (
                store.format_for_prompt(record)
                + "\n\n"
                + audit.format_for_prompt()
            )

        # Step 1: Query Expansion
        sub_queries = expand_query(user_query)
        if verbose:
            print(f"\n[Pipeline] Expanded to {len(sub_queries)} sub-queries:")
            for i, q in enumerate(sub_queries, 1):
                print(f"  {i}. {q}")

        # Step 2: Multi-query Retrieval (parent-document)
        contexts: list[RetrievedContext] = self.retriever.retrieve_multi(
            [user_query] + sub_queries
        )
        if verbose:
            print(f"[Pipeline] Retrieved {len(contexts)} unique parent contexts")

        # Step 3: Rerank
        contexts = rerank(user_query, contexts)
        if verbose:
            print(f"[Pipeline] Reranked to top {len(contexts)} contexts")

        # Step 4: Grounding Check
        grounding: GroundingResult = verify_grounding(contexts)
        if not grounding.verified:
            if verbose:
                print(f"[Pipeline] ⚠️ Grounding failed: {grounding.warning}")
            elapsed = (time.time() - t_start) * 1000
            return format_response(
                raw_answer=NOT_FOUND_RESPONSE,
                contexts=[],
                processing_time_ms=elapsed,
                grounded=False,
            )

        if verbose:
            print(f"[Pipeline] ✅ Grounding passed (max_score={grounding.max_score:.3f})")

        # Step 5: Build prompt and call GPT-4o
        context_block = self._build_context_block(grounding.contexts)
        answer = self._call_llm(
            query=user_query,
            context_block=context_block,
            student_block=student_block,
            verbose=verbose,
        )

        elapsed = (time.time() - t_start) * 1000
        if verbose:
            print(f"[Pipeline] Total processing time: {elapsed:.0f}ms")

        return format_response(
            raw_answer=answer,
            contexts=grounding.contexts,
            processing_time_ms=elapsed,
            grounded=True,
        )

    def _build_context_block(self, contexts: list[RetrievedContext]) -> str:
        """Format retrieved policy contexts for the LLM prompt."""
        blocks = []
        for i, ctx in enumerate(contexts, 1):
            header = f"[Source {i}: {ctx.source}]"
            if ctx.section:
                header += f" — {ctx.section}"
            blocks.append(f"{header}\n{ctx.parent_text}")
        return "\n\n---\n\n".join(blocks)

    def _call_llm(
        self,
        query: str,
        context_block: str,
        student_block: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Call GPT-4o with grounded policy context and optional student record."""
        if verbose:
            print(f"[Pipeline] Calling {settings.reasoning_model}"
                  f" ({'personalized' if student_block else 'anonymous'} mode)...")

        if student_block:
            system = _SYSTEM_PROMPT_PERSONALIZED
            user_content = _USER_TEMPLATE_PERSONALIZED.format(
                query=query,
                student_block=student_block,
                context_block=context_block,
            )
        else:
            system = _SYSTEM_PROMPT
            user_content = _USER_TEMPLATE.format(
                query=query,
                context_block=context_block,
            )

        response = self._llm_client.chat.completions.create(
            model=settings.reasoning_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=1800,
        )
        return response.choices[0].message.content or ""
