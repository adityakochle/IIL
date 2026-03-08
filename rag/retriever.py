"""Parent-document retriever.

Retrieves top-K child chunks from Qdrant, then maps each child back to its
parent chunk (the full policy section), returning deduplicated parent contexts.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from config import settings


@dataclass
class RetrievedContext:
    parent_id: str
    parent_text: str
    score: float
    retrieval_score: float = 0.0  # original Qdrant cosine similarity — used for grounding
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        src = self.metadata.get("source", "Unknown Source")
        page_start = self.metadata.get("page_start")
        page_end = self.metadata.get("page_end")
        if page_start and page_end and page_start != page_end:
            return f"{src}, pg {page_start}–{page_end}"
        elif page_start:
            return f"{src}, pg {page_start}"
        return src

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")


class ParentDocumentRetriever:
    """
    Retrieves child nodes from Qdrant, maps back to parent chunks for full
    policy-section context injection into the LLM prompt.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        parent_store_path: Path | None = None,
    ):
        self.index = index
        parent_store_path = parent_store_path or settings.parents_store
        self._parent_store = self._load_parent_store(parent_store_path)
        self._retriever = index.as_retriever(
            similarity_top_k=settings.top_k_retrieval
        )

    def _load_parent_store(self, path: Path) -> dict[str, str]:
        if not path.exists():
            print(f"[Retriever] Warning: parent store not found at {path}")
            return {}
        return json.loads(path.read_text())

    def retrieve(self, query: str) -> list[RetrievedContext]:
        """Retrieve and map child nodes to parent contexts."""
        child_nodes: list[NodeWithScore] = self._retriever.retrieve(query)
        return self._map_to_parents(child_nodes)

    def retrieve_multi(self, queries: list[str]) -> list[RetrievedContext]:
        """Retrieve for multiple sub-queries and deduplicate by parent_id."""
        seen_parents: dict[str, RetrievedContext] = {}

        for q in queries:
            contexts = self.retrieve(q)
            for ctx in contexts:
                # Keep the highest-scoring occurrence of each parent
                if ctx.parent_id not in seen_parents or ctx.score > seen_parents[ctx.parent_id].score:
                    seen_parents[ctx.parent_id] = ctx

        # Sort by relevance score descending
        return sorted(seen_parents.values(), key=lambda c: c.score, reverse=True)

    def _map_to_parents(self, child_nodes: list[NodeWithScore]) -> list[RetrievedContext]:
        """Map child nodes to their parent chunks."""
        contexts: list[RetrievedContext] = []
        seen: set[str] = set()

        for node_with_score in child_nodes:
            node = node_with_score.node
            parent_id = node.metadata.get("parent_id")
            score = node_with_score.score or 0.0

            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)

            parent_text = self._parent_store.get(parent_id)
            if not parent_text:
                # Fall back to child text if parent not in store
                parent_text = node.text

            contexts.append(
                RetrievedContext(
                    parent_id=parent_id,
                    parent_text=parent_text,
                    score=score,
                    retrieval_score=score,
                    metadata=dict(node.metadata),
                )
            )

        return contexts
