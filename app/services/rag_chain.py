"""RAG (Retrieval-Augmented Generation) pipeline service using LangGraph.

This module implements a stateful RAG pipeline as a LangGraph StateGraph
with three nodes: retrieve → generate → format_sources. It supports both
OpenAI and Google Generative AI backends, selectable via application settings.
"""

import logging
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.models.schemas import ChatRequest, ChatResponse, DocumentMetadata
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template for the generation step
# ---------------------------------------------------------------------------
_RAG_PROMPT_TEMPLATE = """\
You are a research analysis assistant. Answer the question based ONLY on the provided context. \
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Provide a detailed, well-structured answer with references to the source material."""

_RAG_PROMPT = ChatPromptTemplate.from_template(_RAG_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# LangGraph state definition
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    """Typed state dictionary flowing through the RAG graph.

    Attributes:
        question: The user's natural-language question.
        context: Plain-text chunks retrieved from the vector store.
        context_documents: Full LangChain ``Document`` objects with metadata.
        answer: The LLM-generated answer.
        sources: Extracted source metadata dicts for the response payload.
    """

    question: str
    context: list[str]
    context_documents: list[Document]
    answer: str
    sources: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------
class RAGService:
    """LangGraph-based Retrieval-Augmented Generation service.

    The service wires together retrieval from a ``VectorStoreService``,
    answer generation via an LLM, and source-metadata formatting into a
    compiled LangGraph ``StateGraph``.

    Args:
        settings: Application configuration (LLM provider, model, etc.).
        vector_store: Vector store used for similarity search.
    """

    def __init__(self, settings: Settings, vector_store: VectorStoreService) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.llm = self._init_llm()
        self.chain = self._build_graph()

        logger.info(
            "RAGService initialised (provider=%s, model=%s)",
            settings.LLM_PROVIDER,
            settings.LLM_MODEL,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_llm(self) -> Any:
        """Instantiate the chat LLM based on the configured provider.

        Returns:
            A LangChain chat model instance.

        Raises:
            ValueError: If ``settings.LLM_PROVIDER`` is not supported.
        """
        provider = self.settings.LLM_PROVIDER.lower()
        model_name = getattr(self.settings, "LLM_MODEL", "").strip()

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            if not model_name or "gemini" in model_name:
                model_name = "gpt-4o-mini"
            logger.info("Using OpenAI LLM: %s", model_name)
            return ChatOpenAI(
                model=model_name,
                api_key=self.settings.OPENAI_API_KEY,
                temperature=0.1,
            )

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            if not model_name or "gpt-" in model_name or model_name in ("gemini-1.5-flash", "gemini-2.5-flash"):
                model_name = "gemini-3.5-flash-lite"
            logger.info("Using Google Generative AI LLM: %s", model_name)
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.settings.GOOGLE_API_KEY,
            )

        raise ValueError(
            f"Unsupported LLM provider: '{self.settings.LLM_PROVIDER}'. "
            "Choose 'openai' or 'google'."
        )

    # ------------------------------------------------------------------
    # Graph node functions
    # ------------------------------------------------------------------

    async def _retrieve(self, state: RAGState) -> dict[str, Any]:
        """Retrieve relevant document chunks from the vector store.

        Args:
            state: Current graph state containing the user question.

        Returns:
            Dict update with ``context`` and ``context_documents``.
        """
        question: str = state["question"]
        logger.debug("Retrieving documents for question: %.120s…", question)

        try:
            documents: list[Document] = await self.vector_store.similarity_search(
                question
            )
            context = [doc.page_content for doc in documents]

            logger.info("Retrieved %d document chunks", len(documents))
            return {
                "context": context,
                "context_documents": documents,
            }
        except Exception:
            logger.exception("Error during document retrieval")
            return {
                "context": [],
                "context_documents": [],
            }

    async def _generate(self, state: RAGState) -> dict[str, str]:
        """Generate an answer from the retrieved context using the LLM.

        Args:
            state: Current graph state containing ``context`` and ``question``.

        Returns:
            Dict update with ``answer``.
        """
        context_text = "\n\n".join(state["context"]) if state["context"] else ""
        question = state["question"]

        logger.debug("Generating answer for question: %.120s…", question)

        try:
            generation_chain = _RAG_PROMPT | self.llm | StrOutputParser()
            answer: str = await generation_chain.ainvoke(
                {"context": context_text, "question": question}
            )

            logger.info("Answer generated (%d chars)", len(answer))
            return {"answer": answer}
        except Exception:
            logger.exception("Error during answer generation")
            return {
                "answer": (
                    "I'm sorry, I encountered an error while generating a response. "
                    "Please try again."
                )
            }

    async def _format_sources(self, state: RAGState) -> dict[str, list[dict[str, Any]]]:
        """Extract metadata from retrieved documents into a sources list.

        Args:
            state: Current graph state containing ``context_documents``.

        Returns:
            Dict update with ``sources`` — a list of metadata dicts.
        """
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()

        for doc in state.get("context_documents", []):
            metadata = doc.metadata if doc.metadata else {}
            # Deduplicate by source identifier (filename or title)
            source_key = metadata.get("source", metadata.get("title", ""))
            if source_key and source_key in seen:
                continue
            if source_key:
                seen.add(source_key)

            page_num = metadata.get("page_number") or metadata.get("page") or 1
            chunk_num = metadata.get("chunk_id") or metadata.get("chunk_index") or "0"
            sources.append(
                {
                    "title": metadata.get("title", "Unknown"),
                    "source": metadata.get("source", metadata.get("filename", "Unknown")),
                    "filename": metadata.get("filename", metadata.get("source", "Unknown")),
                    "page": page_num,
                    "page_number": page_num,
                    "chunk_id": str(chunk_num),
                    "chunk_index": str(chunk_num),
                    **{
                        k: v
                        for k, v in metadata.items()
                        if k not in {"title", "source", "filename", "page", "page_number", "chunk_id", "chunk_index"}
                    },
                }
            )

        logger.info("Formatted %d unique sources", len(sources))
        return {"sources": sources}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph RAG pipeline.

        Graph topology::

            START → retrieve → generate → format_sources → END

        Returns:
            The compiled LangGraph ``StateGraph`` runnable.
        """
        graph = StateGraph(RAGState)

        # Register nodes
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("format_sources", self._format_sources)

        # Define edges
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "format_sources")
        graph.add_edge("format_sources", END)

        compiled = graph.compile()
        logger.debug("RAG graph compiled successfully")
        return compiled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(self, question: str) -> dict[str, Any]:
        """Run the full RAG pipeline for a given question.

        Args:
            question: The user's natural-language question.

        Returns:
            Final graph state dict containing ``answer``, ``context``,
            ``context_documents``, and ``sources``.

        Raises:
            RuntimeError: If the graph invocation fails unexpectedly.
        """
        logger.info("RAG query: %.200s", question)

        try:
            result: dict[str, Any] = await self.chain.ainvoke(
                {"question": question}
            )
            return result
        except Exception as exc:
            logger.exception("RAG pipeline failed for question: %.200s", question)
            raise RuntimeError(
                "The RAG pipeline encountered an error. Please try again."
            ) from exc

    async def get_chat_response(self, request: ChatRequest) -> ChatResponse:
        """High-level helper that converts a ``ChatRequest`` into a ``ChatResponse``.

        Args:
            request: Incoming chat request with a ``question`` field.

        Returns:
            A fully-populated ``ChatResponse`` with answer text and sources.
        """
        query_text = getattr(request, "query", getattr(request, "question", ""))
        logger.info("Processing chat request: %.200s", query_text)

        result = await self.query(query_text)

        source_metadata = [
            DocumentMetadata(
                filename=src.get("filename") or src.get("source") or "Unknown",
                page_number=src.get("page_number") or src.get("page") or 1,
                chunk_id=str(src.get("chunk_id") or src.get("chunk_index") or "0"),
            )
            for src in result.get("sources", [])
        ]

        return ChatResponse(
            answer=result.get("answer", ""),
            sources=source_metadata,
            session_id=getattr(request, "session_id", None) or "default-session",
        )
