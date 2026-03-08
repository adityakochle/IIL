"""Degree Auditor — RAG-based gap analysis.

Derives a student's outstanding degree requirements by:
  1. Retrieving the relevant major requirements from policy documents (Qdrant)
  2. Sending the retrieved requirements alongside the student's completed
     courses to GPT-4o-mini, which performs the gap analysis
  3. Returning a structured DegreeAudit result

The student record NEVER stores requirements — this module is the
single source of truth for "what is still needed", and it reads that
from the live policy documents every time.
"""
import json
import os

from openai import OpenAI

from config import settings
from pipelines.student_store import StudentRecord, StudentStore

_AUDIT_SYSTEM_PROMPT = """\
You are an SFSU degree audit engine. You will be given:
  1. The official degree requirements for a student's major (retrieved from policy documents)
  2. The student's completed courses and courses currently in progress

Your job is to perform an accurate gap analysis and return a JSON object with exactly
these four keys:

{
  "satisfied": [
    {"requirement": "...", "satisfied_by": "COURSE_CODE — Course Title (Grade)"}
  ],
  "in_progress": [
    {"requirement": "...", "being_satisfied_by": "COURSE_CODE — Course Title (current term)"}
  ],
  "outstanding": [
    {"requirement": "...", "note": "brief note, e.g. prerequisite chain or unit count needed"}
  ],
  "warnings": [
    "Free-text warning strings, e.g. GPA below threshold, D grade won't count, etc."
  ]
}

RULES:
- Base the requirement list STRICTLY on the provided policy context. Do not invent requirements.
- A course with grade D or below does NOT satisfy a requirement that needs C or better — flag it.
- A W (Withdrawal) does NOT satisfy any requirement — flag it if it was a required course.
- Courses in_progress are tentative — note them separately, not as satisfied.
- If a transfer course is listed with an SFSU equivalent, treat the equivalent as completed.
- Be specific: name exact course codes in every entry.
- Output ONLY valid JSON. No markdown, no explanation outside the JSON.
"""

_AUDIT_USER_TEMPLATE = """\
MAJOR: {major}

OFFICIAL DEGREE REQUIREMENTS (from policy documents):
{policy_context}

STUDENT TRANSCRIPT:

Completed courses (transfer):
{transfer_courses}

Completed courses (at SFSU):
{sfsu_courses}

Courses currently in progress:
{in_progress_courses}

Perform the degree gap analysis and return the JSON audit result.
"""


class DegreeAudit:
    """Structured result of a degree gap analysis."""

    def __init__(self, raw: dict, policy_sources: list[str]):
        self.satisfied: list[dict] = raw.get("satisfied", [])
        self.in_progress: list[dict] = raw.get("in_progress", [])
        self.outstanding: list[dict] = raw.get("outstanding", [])
        self.warnings: list[str] = raw.get("warnings", [])
        self.policy_sources = policy_sources  # which policy docs were used

    def format_for_prompt(self) -> str:
        """Render the audit as a readable block for injection into the main LLM prompt."""
        lines = ["=== DEGREE AUDIT (derived from policy documents) ==="]

        if self.satisfied:
            lines.append(f"\n✅ SATISFIED ({len(self.satisfied)} requirements):")
            for item in self.satisfied:
                lines.append(f"  • {item['requirement']} → {item.get('satisfied_by', '')}")

        if self.in_progress:
            lines.append(f"\n🔄 IN PROGRESS ({len(self.in_progress)} requirements):")
            for item in self.in_progress:
                lines.append(f"  • {item['requirement']} → {item.get('being_satisfied_by', '')}")

        if self.outstanding:
            lines.append(f"\n❌ OUTSTANDING ({len(self.outstanding)} requirements):")
            for item in self.outstanding:
                note = f" [{item['note']}]" if item.get("note") else ""
                lines.append(f"  • {item['requirement']}{note}")

        if self.warnings:
            lines.append("\n⚠️  WARNINGS:")
            for w in self.warnings:
                lines.append(f"  • {w}")

        lines.append(f"\nPolicy sources used: {', '.join(self.policy_sources)}")
        lines.append("=" * 51)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "satisfied": self.satisfied,
            "in_progress": self.in_progress,
            "outstanding": self.outstanding,
            "warnings": self.warnings,
            "policy_sources": self.policy_sources,
        }


class DegreeAuditor:
    """
    Runs a RAG-based degree gap analysis for a student.

    Retrieves major requirements from Qdrant, then uses GPT-4o-mini to
    compare them against the student's transcript.
    """

    def __init__(self, index):
        """
        Args:
            index: A loaded VectorStoreIndex (already backed by Qdrant).
        """
        self._index = index
        self._llm = OpenAI(api_key=settings.openai_api_key)
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    def audit(self, record: StudentRecord, verbose: bool = False) -> DegreeAudit:
        """
        Derive outstanding requirements for a student by querying policy docs.

        Args:
            record: A StudentRecord from StudentStore
            verbose: Print retrieval details

        Returns:
            DegreeAudit with satisfied / in_progress / outstanding / warnings
        """
        policy_context, sources = self._retrieve_requirements(record.major, verbose)
        raw_audit = self._run_gap_analysis(record, policy_context, verbose)
        return DegreeAudit(raw=raw_audit, policy_sources=sources)

    def _retrieve_requirements(
        self, major: str, verbose: bool
    ) -> tuple[str, list[str]]:
        """Pull the degree requirements section from Qdrant for this major."""
        from rag.retriever import ParentDocumentRetriever

        retriever = ParentDocumentRetriever(self._index)

        # Targeted queries to pull requirement tables and prereq chains
        queries = [
            f"{major} major required courses prerequisites units",
            f"{major} degree requirements upper division core",
            f"graduation requirements GPA units residency SFSU",
        ]

        contexts = retriever.retrieve_multi(queries)

        if verbose:
            print(f"[DegreeAuditor] Retrieved {len(contexts)} contexts for '{major}' requirements")

        blocks = []
        sources = []
        for ctx in contexts[:6]:  # cap at 6 parent chunks
            blocks.append(f"[{ctx.source}]\n{ctx.parent_text}")
            if ctx.source not in sources:
                sources.append(ctx.source)

        return "\n\n---\n\n".join(blocks), sources

    def _format_course_list(self, courses: list[dict], include_grade: bool = True) -> str:
        if not courses:
            return "  (none)"
        lines = []
        for c in courses:
            equiv = c.get("sfsu_equivalent") or c.get("title") or ""
            grade = f" [Grade: {c['grade']}]" if include_grade and c.get("grade") else ""
            term = f" ({c['term']})" if c.get("term") else ""
            course_code = c.get("course", "")
            lines.append(f"  • {course_code}" + (f" → {equiv}" if equiv else "") + grade + term)
        return "\n".join(lines)

    def _run_gap_analysis(
        self, record: StudentRecord, policy_context: str, verbose: bool
    ) -> dict:
        """Call GPT-4o-mini to perform the structured gap analysis."""
        if verbose:
            print(f"[DegreeAuditor] Running gap analysis for {record.name} ({record.major})...")

        transfer_str = self._format_course_list(
            record.raw.get("transfer_courses_completed", [])
        )
        sfsu_str = self._format_course_list(
            record.raw.get("sfsu_courses_completed", [])
        )
        in_progress_str = self._format_course_list(
            record.courses_in_progress, include_grade=False
        )

        user_content = _AUDIT_USER_TEMPLATE.format(
            major=record.major,
            policy_context=policy_context,
            transfer_courses=transfer_str,
            sfsu_courses=sfsu_str,
            in_progress_courses=in_progress_str,
        )

        response = self._llm.chat.completions.create(
            model=settings.metadata_model,  # GPT-4o-mini — cheaper, fast enough
            messages=[
                {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,  # deterministic for audits
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"[DegreeAuditor] JSON parse error: {e}")
            return {"satisfied": [], "in_progress": [], "outstanding": [], "warnings": [str(e)]}
