"""Pydantic models for the AI Research Analyzer RAG Chatbot.

Defines request/response schemas for document processing, chat interactions,
and evaluation results.
"""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata associated with a document chunk stored in the vector database.

    Attributes:
        filename: Original name of the uploaded file.
        page_number: Page number within the source document.
        chunk_id: Unique identifier for this text chunk.
        upload_timestamp: Timestamp when the document was uploaded.
    """

    filename: str = "Unknown"
    page_number: int = 1
    chunk_id: str = "0"
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("page_number", mode="before")
    @classmethod
    def coerce_page_number(cls, v: Any) -> int:
        """Coerce page number to an integer with safe default fallback."""
        if v is None:
            return 1
        try:
            return int(v)
        except (ValueError, TypeError):
            return 1


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
        timestamp: When the message was created (defaults to now).
    """

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint.

    Supports both 'query' and 'question' field names for flexible client integrations.

    Attributes:
        query: The user's question or prompt.
        session_id: Unique session identifier (auto-generated if not provided).
        include_evaluation: Whether to include RAG evaluation in the response.
    """

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(..., alias="question", validation_alias="query")
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    include_evaluation: bool = False

    @computed_field
    @property
    def question(self) -> str:
        """Alias for query."""
        return self.query

    def __init__(self, **data: Any):
        # Support either query or question seamlessly
        if "question" in data and "query" not in data:
            data["query"] = data.pop("question")
        super().__init__(**data)


class EvaluationScore(BaseModel):
    """A single evaluation metric score.

    Attributes:
        metric_name: Name of the evaluation metric.
        score: Numeric score between 0 and 1.
        reasoning: Explanation of how the score was determined.
        details: Optional additional details about the evaluation.
    """

    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    details: dict[str, Any] | None = None

    @computed_field
    @property
    def name(self) -> str:
        """Alias for metric_name for frontend compatibility."""
        return self.metric_name


class EvaluationResult(BaseModel):
    """Aggregated evaluation result containing multiple metric scores.

    Attributes:
        scores: List of individual metric scores.
        overall_score: Weighted overall score between 0 and 1.
        verdict: Final pass/warn/fail verdict.
        evaluated_at: Timestamp when the evaluation was performed.
    """

    scores: list[EvaluationScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["PASS", "WARN", "FAIL"]
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def metrics(self) -> list[dict[str, Any]]:
        """List of metric dicts for frontend compatibility."""
        return [
            {
                "name": s.metric_name,
                "score": s.score,
                "reasoning": s.reasoning,
                "details": s.details,
            }
            for s in self.scores
        ]


class ChatResponse(BaseModel):
    """Response payload from the chat endpoint.

    Attributes:
        answer: The generated answer text.
        sources: List of document metadata for retrieved sources.
        session_id: The session identifier for this conversation.
        evaluation: Optional evaluation result if requested.
    """

    answer: str
    sources: list[DocumentMetadata]
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    evaluation: EvaluationResult | None = None


class DocumentUploadResponse(BaseModel):
    """Response payload after a document upload and processing.

    Attributes:
        filename: Name of the uploaded file.
        num_chunks: Number of text chunks created from the document.
        status: Processing status message.
    """

    filename: str
    num_chunks: int
    status: str

    @computed_field
    @property
    def chunks_created(self) -> int:
        """Alias for num_chunks for frontend compatibility."""
        return self.num_chunks


class EvaluationRequest(BaseModel):
    """Request payload for standalone evaluation of a RAG response.

    Attributes:
        question: The original question that was asked.
        answer: The generated answer to evaluate.
        context: List of retrieved context passages used for generation.
        ground_truth: Optional known-correct answer for comparison.
    """

    question: str
    answer: str
    context: list[str]
    ground_truth: str | None = None


class HealthResponse(BaseModel):
    """Response payload for the health check endpoint.

    Attributes:
        status: Current service status.
        version: Application version string.
        documents_count: Number of documents in the vector store.
        llm_provider: Active LLM provider name.
    """

    status: str
    version: str
    documents_count: int
    llm_provider: str

    @computed_field
    @property
    def document_count(self) -> int:
        """Alias for documents_count for frontend compatibility."""
        return self.documents_count
