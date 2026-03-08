"""Document loaders with rich metadata extraction."""
import json
import re
from pathlib import Path
from typing import Any

import pypdf
import yaml
from llama_index.core import Document

from config import settings


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown files."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    fm = yaml.safe_load(text[3:end]) or {}
    body = text[end + 3:].lstrip("\n")
    return fm, body


def _extract_page_refs(text: str) -> list[int]:
    """Extract page numbers referenced in the text via [Page N] markers."""
    return [int(m) for m in re.findall(r"\[Page (\d+)\]", text)]


def load_markdown_documents(directory: Path | None = None) -> list[Document]:
    """
    Load all markdown policy documents from the synthetic data directory.
    Attaches source, page, section, and doc_id metadata to each document.
    """
    directory = directory or settings.synthetic_dir
    docs: list[Document] = []

    for md_file in sorted(directory.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        frontmatter, body = _extract_frontmatter(raw)
        page_refs = _extract_page_refs(body)

        metadata: dict[str, Any] = {
            "file_name": md_file.name,
            "doc_id": frontmatter.get("doc_id", md_file.stem),
            "source": frontmatter.get("source", md_file.stem),
            "section": frontmatter.get("section", ""),
            "page_start": frontmatter.get("page_start", page_refs[0] if page_refs else 1),
            "page_end": frontmatter.get("page_end", page_refs[-1] if page_refs else 1),
            "last_updated": frontmatter.get("last_updated", ""),
            "doc_type": "policy_document",
        }

        docs.append(
            Document(
                text=body,
                metadata=metadata,
                doc_id=metadata["doc_id"],
            )
        )
        print(f"  Loaded: {md_file.name} ({len(body):,} chars, pages {metadata['page_start']}–{metadata['page_end']})")

    return docs


def load_pdf_documents(directory: Path | None = None) -> list[Document]:
    """
    Load all PDF policy documents from the synthetic data directory.
    Inserts [Page N] markers at each page boundary so citations work
    the same way as markdown documents.
    """
    directory = directory or settings.synthetic_dir
    docs: list[Document] = []

    for pdf_file in sorted(directory.glob("*.pdf")):
        reader = pypdf.PdfReader(str(pdf_file))
        pages: list[str] = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append(f"[Page {i}]\n{text.strip()}")
        full_text = "\n\n".join(pages)

        num_pages = len(reader.pages)
        source_name = pdf_file.stem.replace("_", " ").title()

        metadata: dict[str, Any] = {
            "file_name": pdf_file.name,
            "doc_id": pdf_file.stem,
            "source": source_name,
            "section": "",
            "page_start": 1,
            "page_end": num_pages,
            "last_updated": "",
            "doc_type": "policy_document",
        }

        docs.append(
            Document(
                text=full_text,
                metadata=metadata,
                doc_id=pdf_file.stem,
            )
        )
        print(f"  Loaded: {pdf_file.name} ({len(full_text):,} chars, {num_pages} pages)")

    return docs


def load_enrollment_data(json_path: Path | None = None) -> list[Document]:
    """
    Load the enrollment patterns JSON and convert each scenario to a Document
    for inclusion in the vector store (useful for data-driven queries).
    """
    json_path = json_path or (settings.synthetic_dir / "enrollment_patterns.json")
    raw = json.loads(json_path.read_text())

    docs: list[Document] = []

    # Aggregate stats as one document
    stats_text = (
        "SFSU Enrollment Aggregate Statistics (2009-2024, anonymized):\n"
        + json.dumps(raw["aggregate_statistics"], indent=2)
    )
    docs.append(
        Document(
            text=stats_text,
            metadata={
                "doc_id": "enrollment_aggregate",
                "source": "SFSU Institutional Research 2024",
                "section": "Aggregate Enrollment Statistics",
                "page_start": 1,
                "page_end": 1,
                "doc_type": "enrollment_data",
                "file_name": "enrollment_patterns.json",
                "last_updated": raw.get("last_updated", ""),
            },
            doc_id="enrollment_aggregate",
        )
    )

    # Each scenario as a separate document
    for scenario in raw.get("scenarios", []):
        text = (
            f"Enrollment Scenario {scenario['id']} (cohort {scenario['cohort_year']}):\n"
            + json.dumps(scenario, indent=2)
        )
        docs.append(
            Document(
                text=text,
                metadata={
                    "doc_id": f"enrollment_{scenario['id']}",
                    "source": "SFSU Institutional Research 2024",
                    "section": f"Scenario {scenario['id']}",
                    "page_start": 1,
                    "page_end": 1,
                    "doc_type": "enrollment_scenario",
                    "file_name": "enrollment_patterns.json",
                    "last_updated": raw.get("last_updated", ""),
                },
                doc_id=f"enrollment_{scenario['id']}",
            )
        )

    print(f"  Loaded enrollment data: {len(docs)} documents from {json_path.name}")
    return docs


def load_all_documents() -> list[Document]:
    """Load all documents (policy markdown + PDF + enrollment JSON)."""
    print("\n[Loader] Loading markdown policy documents...")
    policy_docs = load_markdown_documents()
    print("[Loader] Loading PDF policy documents...")
    pdf_docs = load_pdf_documents()
    print("[Loader] Loading enrollment data...")
    enrollment_docs = load_enrollment_data()
    all_docs = policy_docs + pdf_docs + enrollment_docs
    print(f"[Loader] Total documents loaded: {len(all_docs)}\n")
    return all_docs
