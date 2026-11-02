"""Unit tests for Pydantic schemas, validation, and aliases."""

from datetime import datetime, timezone
import pytest
from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentMetadata,
    DocumentUploadResponse,
    EvaluationRequest,
    EvaluationResult,
    EvaluationScore,
    HealthResponse,
)


def test_document_metadata_schema():
    meta = DocumentMetadata(
        filename="paper.pdf",
        page_number=3,
        chunk_id="chunk-12",
        upload_timestamp=datetime.now(timezone.utc),
    )
    assert meta.filename == "paper.pdf"
    assert meta.page_number == 3
    assert meta.chunk_id == "chunk-12"


def test_chat_request_query_and_question_alias():
    req1 = ChatRequest(query="What is transformer?")
    assert req1.query == "What is transformer?"

    req2 = ChatRequest(question="What is attention?")
    assert req2.query == "What is attention?"


def test_evaluation_score_and_result():
    s1 = EvaluationScore(
        metric_name="faithfulness",
        score=0.95,
        reasoning="All claims supported",
        details={"claims_count": 3},
    )
    assert s1.name == "faithfulness"
    assert s1.score == 0.95

    res = EvaluationResult(
        scores=[s1],
        overall_score=0.95,
        verdict="PASS",
    )
    assert res.verdict == "PASS"
    assert len(res.metrics) == 1
    assert res.metrics[0]["name"] == "faithfulness"


def test_document_upload_response_aliases():
    resp = DocumentUploadResponse(
        filename="test.pdf",
        num_chunks=42,
        status="success",
    )
    assert resp.num_chunks == 42
    assert resp.chunks_created == 42


def test_health_response_aliases():
    health = HealthResponse(
        status="healthy",
        version="1.0.0",
        documents_count=100,
        llm_provider="openai",
    )
    assert health.documents_count == 100
    assert health.document_count == 100
