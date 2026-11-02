"""Unit tests for configuration loading and property aliases."""

import pytest
from app.config import Settings, get_settings


def test_settings_default_values():
    """Verify default configuration attributes."""
    settings = Settings()
    assert settings.api_port == 8000
    assert settings.llm_provider in ("openai", "google")
    assert settings.collection_name == "research_docs"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200


def test_settings_case_insensitive_access():
    """Verify settings support uppercase attribute access (e.g. LLM_PROVIDER)."""
    settings = Settings()
    assert settings.LLM_PROVIDER == settings.llm_provider
    assert settings.CHUNK_SIZE == settings.chunk_size
    assert settings.CHUNK_OVERLAP == settings.chunk_overlap
    assert settings.CHROMA_PERSIST_DIR == settings.chroma_persist_dir
    assert settings.CHROMA_COLLECTION_NAME == settings.collection_name


def test_get_settings_factory():
    """Verify get_settings returns a valid Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
