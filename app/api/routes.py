"""FastAPI routes for the AI Research Analyzer API.

Provides endpoints for document upload, RAG-based chat, standalone
evaluation, health checks, and collection management.
"""

import functools
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentMetadata,
    DocumentUploadResponse,
    EvaluationRequest,
    EvaluationResult,
    HealthResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["AI Research Analyzer"])


# ---------------------------------------------------------------------------
# Dependency injection helpers (singleton via lru_cache)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application Settings instance.

    Returns:
        Settings: Application settings loaded from environment / .env file.
    """
    return Settings()


@functools.lru_cache(maxsize=1)
def get_document_processor() -> DocumentProcessor:
    """Return a cached DocumentProcessor instance.

    Returns:
        DocumentProcessor: Singleton document processor service.
    """
    return DocumentProcessor(get_settings())


@functools.lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    """Return a cached VectorStoreService instance.

    Returns:
        VectorStoreService: Singleton vector store service.
    """
    return VectorStoreService(get_settings())


@functools.lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return a cached RAGService instance.

    Returns:
        RAGService: Singleton RAG pipeline service.
    """
    return RAGService(get_settings(), get_vector_store())


def get_eval_engine() -> "EvaluationEngine":
    """Return a cached EvaluationEngine instance.

    The import is deferred to avoid circular imports and to tolerate
    the evaluation module not yet being present during early development.

    Returns:
        EvaluationEngine: Singleton evaluation engine service.
    """
    return _cached_eval_engine()


@functools.lru_cache(maxsize=1)
def _cached_eval_engine() -> "EvaluationEngine":
    """Inner cached factory for EvaluationEngine.

    Returns:
        EvaluationEngine: Cached evaluation engine instance.
    """
    from app.evaluation.engine import EvaluationEngine  # noqa: WPS433

    return EvaluationEngine(get_settings())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    eval_engine: "EvaluationEngine" = Depends(get_eval_engine),
) -> ChatResponse:
    """Process a user query through the RAG pipeline and return an answer.

    Optionally evaluates the generated response when
    ``request.include_evaluation`` is ``True``.

    Args:
        request: Incoming chat request containing the user query.
        rag_service: Injected RAG pipeline service.
        eval_engine: Injected evaluation engine service.

    Returns:
        ChatResponse with the generated answer, source metadata, and
        optional evaluation result.

    Raises:
        HTTPException: 500 if the RAG pipeline or evaluation fails.
    """
    try:
        logger.info("Chat request received: %.200s", request.query)

        # Run the RAG pipeline
        rag_result = await rag_service.query(request.query)

        # Build source metadata from the pipeline result
        sources: list[DocumentMetadata] = []
        for src in rag_result.get("sources", []):
            fname = src.get("filename") or src.get("source") or "Unknown"
            raw_page = src.get("page_number")
            if raw_page is None:
                raw_page = src.get("page")
            try:
                p_num = int(raw_page) if raw_page is not None else 1
            except (ValueError, TypeError):
                p_num = 1

            raw_chunk = src.get("chunk_id")
            if raw_chunk is None:
                raw_chunk = src.get("chunk_index", "0")

            sources.append(
                DocumentMetadata(
                    filename=fname,
                    page_number=p_num,
                    chunk_id=str(raw_chunk),
                    upload_timestamp=src.get(
                        "upload_timestamp", datetime.now(timezone.utc)
                    ),
                )
            )

        response = ChatResponse(
            answer=rag_result.get("answer", ""),
            sources=sources,
            session_id=request.session_id,
        )

        # Optionally run evaluation
        if request.include_evaluation:
            try:
                eval_request = EvaluationRequest(
                    question=request.query,
                    answer=rag_result.get("answer", ""),
                    context=rag_result.get("context", []),
                )
                evaluation = await eval_engine.evaluate(eval_request)
                response.evaluation = evaluation
            except Exception as exc:
                logger.warning("Evaluation failed (non-fatal): %s", exc)

        return response

    except Exception as exc:
        logger.exception("Chat endpoint failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {exc}",
        ) from exc


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """Upload a PDF document, process it, and add chunks to the vector store.

    Args:
        file: The uploaded PDF file.
        doc_processor: Injected document processor service.
        vector_store: Injected vector store service.

    Returns:
        DocumentUploadResponse with the filename, chunk count, and status.

    Raises:
        HTTPException: 400 if the file is not a PDF.
        HTTPException: 500 if processing or storage fails.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    try:
        logger.info("Uploading file: %s", file.filename)

        # Read and save the uploaded file
        file_content = await file.read()
        file_path = await doc_processor.process_uploaded_file(
            file_content, file.filename
        )

        # Extract text and split into chunks
        documents = await doc_processor.process_pdf(file_path, file.filename)

        # Add chunks to the vector store
        num_chunks = await vector_store.add_documents(documents)

        logger.info(
            "Successfully processed '%s': %d chunks added",
            file.filename,
            num_chunks,
        )

        return DocumentUploadResponse(
            filename=file.filename,
            num_chunks=num_chunks,
            status="success",
        )

    except Exception as exc:
        logger.exception("Upload failed for file: %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {exc}",
        ) from exc


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate(
    request: EvaluationRequest,
    eval_engine: "EvaluationEngine" = Depends(get_eval_engine),
) -> EvaluationResult:
    """Run standalone evaluation on a RAG question-answer-context triple.

    Args:
        request: Evaluation request with question, answer, context, and
            optional ground truth.
        eval_engine: Injected evaluation engine service.

    Returns:
        EvaluationResult with metric scores, overall score, and verdict.

    Raises:
        HTTPException: 500 if the evaluation engine fails.
    """
    try:
        logger.info("Evaluation request for question: %.200s", request.question)
        result = await eval_engine.evaluate(request)
        return result

    except Exception as exc:
        logger.exception("Evaluation endpoint failed")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        ) from exc


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> HealthResponse:
    """Return current service health including document count and LLM provider.

    Args:
        settings: Injected application settings.
        vector_store: Injected vector store service.

    Returns:
        HealthResponse with status, version, document count, and LLM provider.
    """
    try:
        doc_count = await vector_store.get_document_count()
    except Exception as exc:
        logger.warning("Could not retrieve document count: %s", exc)
        doc_count = -1

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        documents_count=doc_count,
        llm_provider=settings.llm_provider,
    )


@router.delete("/documents")
async def delete_documents(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict[str, str]:
    """Delete the entire vector store collection.

    This action is **irreversible** — all stored document embeddings will
    be permanently removed.

    Args:
        vector_store: Injected vector store service.

    Returns:
        A dict with a success message.

    Raises:
        HTTPException: 500 if the deletion fails.
    """
    try:
        logger.info("Deleting vector store collection")
        await vector_store.delete_collection()
        return {"message": "All documents deleted successfully"}

    except Exception as exc:
        logger.exception("Failed to delete documents")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete documents: {exc}",
        ) from exc
