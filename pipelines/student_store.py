"""Student Record Store.

Provides a fast in-memory lookup of individual student records by student_id.
In production, replace the JSON backend with a FERPA-compliant SIS API
(e.g., PeopleSoft Campus Solutions, Workday Student, or a secured PostgreSQL DB).

Usage:
    store = StudentStore()
    record = store.get("STU-2024-0001")
    summary = store.format_for_prompt(record)
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings


@dataclass
class StudentRecord:
    student_id: str
    raw: dict  # full record dict

    # Convenience accessors
    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown")

    @property
    def major(self) -> str:
        return self.raw.get("declared_major", "Undeclared")

    @property
    def cumulative_gpa(self) -> float:
        return float(self.raw.get("cumulative_gpa", 0.0))

    @property
    def major_gpa(self) -> float:
        return float(self.raw.get("major_gpa", 0.0))

    @property
    def academic_standing(self) -> str:
        return self.raw.get("academic_standing", "Unknown")

    @property
    def units_transferred(self) -> int:
        return int(self.raw.get("units_transferred", 0))

    @property
    def units_at_sfsu(self) -> int:
        return int(self.raw.get("units_completed_at_sfsu", 0))

    @property
    def total_units(self) -> int:
        return self.units_transferred + self.units_at_sfsu

    @property
    def courses_completed(self) -> list[dict]:
        transfer = self.raw.get("transfer_courses_completed", [])
        sfsu = self.raw.get("sfsu_courses_completed", [])
        return transfer + sfsu

    @property
    def courses_in_progress(self) -> list[dict]:
        return self.raw.get("courses_in_progress", [])

    @property
    def financial_aid(self) -> dict:
        return self.raw.get("financial_aid", {})

    @property
    def advisor_notes(self) -> str:
        return self.raw.get("advisor_notes", "")


class StudentStore:
    """
    In-memory student record store backed by a JSON file.
    Loaded once at startup; thread-safe for read operations.
    """

    _DEFAULT_PATH = settings.data_dir / "synthetic" / "student_records.json"

    def __init__(self, records_path: Path | None = None):
        path = records_path or self._DEFAULT_PATH
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._records: dict[str, dict] = raw.get("records", {})

    def get(self, student_id: str) -> StudentRecord | None:
        """Return a StudentRecord for the given ID, or None if not found."""
        raw = self._records.get(student_id)
        if raw is None:
            return None
        return StudentRecord(student_id=student_id, raw=raw)

    def list_ids(self) -> list[str]:
        return list(self._records.keys())

    def format_for_prompt(self, record: StudentRecord) -> str:
        """
        Render a student's transcript facts as a structured block for LLM prompt injection.

        NOTE: Outstanding requirements are intentionally excluded here.
        They are derived at query time by DegreeAuditor against live policy documents
        and injected separately. This block contains only student-owned facts.
        """
        completed = record.courses_completed
        in_progress = record.courses_in_progress
        aid = record.financial_aid

        completed_str = "\n".join(
            f"    • {c.get('course', '')} — {c.get('sfsu_equivalent', c.get('title', ''))} "
            f"[Grade: {c.get('grade', '?')}, {c.get('units', '?')} units]"
            for c in completed
        ) or "    None on record"

        in_progress_str = "\n".join(
            f"    • {c.get('course', '')} — {c.get('title', '')} ({c.get('term', '')})"
            for c in in_progress
        ) or "    None"

        aid_str = (
            f"Status: {aid.get('status', 'Unknown')} | "
            f"Types: {', '.join(aid.get('type', []))} | "
            f"Remaining eligibility: {aid.get('remaining_aid_years', '?')} years"
        )
        if aid.get("sap_note"):
            aid_str += f"\n    ⚠️  SAP Note: {aid['sap_note']}"

        return f"""
=== STUDENT RECORD (VERIFIED TRANSCRIPT) ===
Student ID:        {record.student_id}
Name:              {record.name}
Major:             {record.major}
College:           {record.raw.get('college', '')}
Catalog Year:      {record.raw.get('catalog_year', '')}
Admit Term:        {record.raw.get('admit_term', '')}
Admission Type:    {record.raw.get('admission_type', '')}
Academic Standing: {record.academic_standing}
Enrollment:        {record.raw.get('enrollment_status', '')}
IGETC Certified:   {record.raw.get('igetc_certified', False)}

GPA SUMMARY
  Cumulative GPA:    {record.cumulative_gpa}
  Major GPA:         {record.major_gpa}

UNITS SUMMARY
  Transferred:       {record.units_transferred}
  Completed at SFSU: {record.units_at_sfsu}
  Total:             {record.total_units}

FINANCIAL AID
  {aid_str}

COMPLETED COURSES (transcript)
{completed_str}

COURSES IN PROGRESS (current term)
{in_progress_str}

ADVISOR NOTES
  {record.advisor_notes or 'None'}

[Outstanding requirements are derived from policy documents — see DEGREE AUDIT below]
============================================
""".strip()
