"""Qdrant indexer: embeds child nodes and upserts into Qdrant Cloud collection."""
import os

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import settings
from pipelines.embedder import get_embedding_model


def _get_qdrant_client() -> QdrantClient:
    """Return a connected Qdrant client (Cloud or local)."""
    if settings.qdrant_api_key:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
    # Local Docker fallback
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"[Indexer] Created collection '{settings.qdrant_collection}'")
    else:
        print(f"[Indexer] Collection '{settings.qdrant_collection}' already exists")


def index_nodes(nodes: list[TextNode]) -> VectorStoreIndex:
    """
    Embed child nodes and upsert into Qdrant.

    Returns:
        VectorStoreIndex backed by Qdrant for later retrieval.
    """
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    client = _get_qdrant_client()
    ensure_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embedding_model()

    print(f"[Indexer] Embedding and indexing {len(nodes)} child nodes...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"[Indexer] Successfully indexed {len(nodes)} nodes into Qdrant")
    return index


def load_existing_index() -> VectorStoreIndex:
    """Load an existing Qdrant-backed index (no re-embedding)."""
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    client = _get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embedding_model()
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
