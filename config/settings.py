"""Central configuration for the Institutional Intelligence Layer."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API Keys ---
    openai_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    cohere_api_key: str = ""

    # --- Qdrant ---
    qdrant_collection: str = "sfsu_policies"
    qdrant_vector_size: int = 1536  # text-embedding-3-small

    # --- Models ---
    reasoning_model: str = "gpt-4o"
    metadata_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    cohere_rerank_model: str = "rerank-english-v3.0"

    # --- Chunking ---
    child_chunk_size: int = 512
    child_chunk_overlap: int = 64
    parent_chunk_size: int = 2048
    parent_chunk_overlap: int = 128

    # --- Retrieval ---
    top_k_retrieval: int = 10     # child chunks from Qdrant
    top_n_rerank: int = 5         # after reranking
    grounding_threshold: float = 0.35  # minimum relevance score

    # --- Paths ---
    data_dir: Path = ROOT_DIR / "data"
    synthetic_dir: Path = ROOT_DIR / "data" / "synthetic"
    parents_store: Path = ROOT_DIR / "data" / "parents.json"


settings = Settings()
