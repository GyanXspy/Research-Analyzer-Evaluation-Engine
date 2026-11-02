"""Unit tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import pytest

from main import app
from app.api.routes import get_rag_service, get_eval_engine, get_vector_store
from app.models.schemas import EvaluationResult, EvaluationScore, ChatResponse, DocumentMetadata


@pytest.fixture
def client():
    # Setup dependency overrides for testing without external services
    mock_vs = MagicMock()
    mock_vs.get_document_count = AsyncMock(return_value=12)
    mock_vs.delete_collection = AsyncMock()

    mock_rag = MagicMock()
    mock_rag.query = AsyncMock(
        return_value={
            "answer": "Transformers use self-attention to process tokens in parallel.",
            "context": ["Transformers rely on self-attention mechanisms."],
            "sources": [{"filename": "paper.pdf", "page_number": 1, "chunk_id": "0"}],
        }
    )

    mock_engine = MagicMock()
    mock_engine.evaluate = AsyncMock(
        return_value=EvaluationResult(
            scores=[
                EvaluationScore(
                    metric_name="faithfulness",
                    score=1.0,
                    reasoning="All claims grounded in context",
                )
            ],
            overall_score=1.0,
            verdict="PASS",
        )
    )

    app.dependency_overrides[get_vector_store] = lambda: mock_vs
    app.dependency_overrides[get_rag_service] = lambda: mock_rag
    app.dependency_overrides[get_eval_engine] = lambda: mock_engine

    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["documents_count"] == 12
    assert "llm_provider" in data


def test_chat_endpoint_without_evaluation(client):
    response = client.post(
        "/api/chat",
        json={"query": "What is transformer?", "include_evaluation": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) >= 1
    assert data["evaluation"] is None


def test_chat_endpoint_with_evaluation(client):
    response = client.post(
        "/api/chat",
        json={"question": "What is transformer?", "include_evaluation": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["evaluation"] is not None
    assert data["evaluation"]["verdict"] == "PASS"


def test_standalone_evaluate_endpoint(client):
    response = client.post(
        "/api/evaluate",
        json={
            "question": "What is attention?",
            "answer": "It is a mechanism",
            "context": ["Attention mechanism context"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["overall_score"] == 1.0
    assert data["verdict"] == "PASS"


def test_delete_documents_endpoint(client):
    response = client.delete("/api/documents")
    assert response.status_code == 200
    assert "message" in response.json()
