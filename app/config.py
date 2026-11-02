"""Application configuration using pydantic-settings.

Loads settings from environment variables and .env file.
Supports case-insensitive attribute access and aliases for robust compatibility.
"""

from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Attributes:
        openai_api_key: OpenAI API key for LLM and embedding calls.
        google_api_key: Google API key for Gemini models.
        llm_provider: LLM provider to use ('openai' or 'google').
        llm_model: Model name for chat/completion requests.
        embedding_model: Model name for generating embeddings.
        chroma_persist_dir: Directory path for ChromaDB persistence.
        upload_dir: Directory path for uploaded documents.
        collection_name: ChromaDB collection name for research documents.
        chunk_size: Number of characters per text chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        top_k_results: Number of top results to retrieve from vector search.
        eval_llm_model: Model name used for evaluation scoring.
        api_host: Host address for the FastAPI server.
        api_port: Port number for the FastAPI server.
        streamlit_port: Port number for the Streamlit UI.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = "your-openai-key-here"
    google_api_key: str = "your-google-key-here"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"
    collection_name: str = "research_docs"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    eval_llm_model: str = "gpt-4o-mini"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    def __getattr__(self, name: str) -> Any:
        """Allow case-insensitive and aliased attribute access (e.g. settings.LLM_PROVIDER)."""
        lower = name.lower()
        if lower == "chroma_collection_name":
            lower = "collection_name"

        if lower in self.__dict__:
            return self.__dict__[lower]

        if lower in self.model_fields:
            return getattr(self, lower)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Returns:
        Settings: Application settings populated from env vars / .env file.
    """
    return Settings()
