"""Embedding utility using OpenAI text-embedding-3-small."""
import os

from llama_index.embeddings.openai import OpenAIEmbedding

from config import settings


def get_embedding_model() -> OpenAIEmbedding:
    """Return a configured OpenAI embedding model instance."""
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    return OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        embed_batch_size=100,
    )
