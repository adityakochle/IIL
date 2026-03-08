"""Parent-child chunking strategy.

Architecture:
  - Parent chunks: large (2048 tokens) — fed to LLM for full context
  - Child chunks: small (512 tokens) — embedded and stored in Qdrant for retrieval
  - Each child node carries a `parent_id` pointing to its parent text
"""
import hashlib
import json
import uuid
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from config import settings


def _node_id(text: str, doc_id: str, index: int) -> str:
    """Generate a deterministic UUID v5 from doc_id + index + text prefix."""
    name = f"{doc_id}:{index}:{text[:100]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def _parent_id(doc_id: str, index: int) -> str:
    return f"{doc_id}_parent_{index}"


def create_parent_child_nodes(
    documents: list[Document],
    parents_store_path: Path | None = None,
) -> tuple[list[TextNode], dict[str, str]]:
    """
    Chunk documents into parent and child nodes.

    Returns:
        child_nodes: list of TextNode (to be embedded and stored in Qdrant)
        parent_store: dict mapping parent_id → parent text (stored as JSON sidecar)
    """
    parents_store_path = parents_store_path or settings.parents_store

    parent_splitter = SentenceSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
    )
    child_splitter = SentenceSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )

    child_nodes: list[TextNode] = []
    parent_store: dict[str, str] = {}

    for doc in documents:
        parent_chunks = parent_splitter.split_text(doc.text)

        for p_idx, parent_text in enumerate(parent_chunks):
            pid = _parent_id(doc.doc_id or doc.metadata.get("doc_id", "doc"), p_idx)
            parent_store[pid] = parent_text

            child_chunks = child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(child_chunks):
                node_id = _node_id(child_text, pid, c_idx)
                node = TextNode(
                    text=child_text,
                    id_=node_id,
                    metadata={
                        **doc.metadata,
                        "parent_id": pid,
                        "chunk_index": c_idx,
                        "parent_index": p_idx,
                    },
                )
                child_nodes.append(node)

    # Persist parent store so retriever can look up full context later
    parents_store_path.parent.mkdir(parents=True, exist_ok=True)
    parents_store_path.write_text(json.dumps(parent_store, indent=2), encoding="utf-8")

    print(
        f"[Chunker] Created {len(child_nodes)} child nodes "
        f"from {len(parent_store)} parents across {len(documents)} documents"
    )
    return child_nodes, parent_store
