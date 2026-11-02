"""Vector store service for document embedding storage and similarity search.

Manages a ChromaDB-backed vector store with support for both OpenAI and
Google Generative AI embedding providers, configurable via application settings.
"""

import asyncio
import logging
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages document embeddings and similarity search via ChromaDB.

    Provides an async interface for adding documents to a persistent
    ChromaDB collection, performing similarity searches, and managing
    the underlying vector store lifecycle.

    Attributes:
        settings: Application settings for LLM provider and store configuration.
        embeddings: The embedding model instance (OpenAI or Google).
        vector_store: The ChromaDB-backed LangChain vector store.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the vector store service.

        Selects the embedding provider based on ``settings.LLM_PROVIDER``
        and configures ChromaDB with the persist directory and collection
        name from settings.

        Args:
            settings: Application settings with LLM_PROVIDER, OPENAI_API_KEY,
                GOOGLE_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
                CHROMA_COLLECTION_NAME, and TOP_K_RESULTS.

        Raises:
            ValueError: If LLM_PROVIDER is not ``"openai"`` or ``"google"``.
        """
        self.settings = settings
        self.embeddings = self._create_embeddings()
        self.vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        logger.info(
            "VectorStoreService initialized (provider=%s, collection=%s)",
            settings.LLM_PROVIDER,
            settings.CHROMA_COLLECTION_NAME,
        )

    def _create_embeddings(self) -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings:
        """Create the embedding model based on the configured LLM provider.

        Returns:
            An OpenAI or Google Generative AI embeddings instance.

        Raises:
            ValueError: If the configured LLM_PROVIDER is unsupported.
        """
        provider = self.settings.LLM_PROVIDER.lower()
        model_name = getattr(self.settings, "EMBEDDING_MODEL", "").strip()

        if provider == "openai":
            if not model_name or "gemini" in model_name:
                model_name = "text-embedding-3-small"
            logger.info("Using OpenAI embeddings (model=%s)", model_name)
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=self.settings.OPENAI_API_KEY,
            )
        elif provider == "google":
            if not model_name or "text-embedding-3" in model_name or "text-embedding-ada" in model_name:
                model_name = "gemini-embedding-001"
            elif model_name == "models/text-embedding-004" or model_name == "text-embedding-004":
                model_name = "gemini-embedding-001"
            logger.info("Using Google embeddings (model=%s)", model_name)
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.settings.GOOGLE_API_KEY,
            )
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{provider}'. "
                "Must be 'openai' or 'google'."
            )

    async def add_documents(
        self, documents: list[Document], batch_size: int = 25
    ) -> int:
        """Add documents to the vector store with automatic batching and rate-limit retries.

        Embeds and stores the provided LangChain Document objects in the
        ChromaDB collection. To prevent exceeding free-tier rate limits (e.g. Google
        Gemini 100 requests/minute), documents are processed in small batches with
        inter-batch throttling and automatic retry on 429 RESOURCE_EXHAUSTED.

        Args:
            documents: List of LangChain Document objects to embed and store.
            batch_size: Number of chunks to embed per batch (default 25).

        Returns:
            The number of documents successfully added.

        Raises:
            RuntimeError: If the documents cannot be added to the store after retries.
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return 0

        total_docs = len(documents)
        logger.info(
            "Indexing %d documents in batches of %d...", total_docs, batch_size
        )

        added_count = 0
        total_batches = (total_docs + batch_size - 1) // batch_size

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    await asyncio.to_thread(self.vector_store.add_documents, batch)
                    added_count += len(batch)
                    logger.info(
                        "Indexed batch %d/%d (%d/%d documents)",
                        batch_num,
                        total_batches,
                        added_count,
                        total_docs,
                    )
                    break
                except Exception as exc:
                    err_msg = str(exc)
                    is_rate_limit = (
                        "429" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                        or "quota" in err_msg.lower()
                    )
                    if is_rate_limit and attempt < max_retries:
                        # Parse retry delay from error if specified (e.g. retry in 15.2s)
                        delay_match = re.search(r"retry\s+(?:in\s+)?(\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                        wait_seconds = float(delay_match.group(1)) + 1.0 if delay_match else 16.0 + (attempt * 3.0)

                        logger.warning(
                            "Quota rate limit hit (429) on batch %d/%d (attempt %d/%d). "
                            "Sleeping %.1fs for rate limit window reset...",
                            batch_num,
                            total_batches,
                            attempt,
                            max_retries,
                            wait_seconds,
                        )
                        await asyncio.sleep(wait_seconds)
                    else:
                        logger.error(
                            "Failed to add batch %d/%d to vector store: %s",
                            batch_num,
                            total_batches,
                            exc,
                        )
                        raise RuntimeError(
                            f"Failed to add documents to vector store: {exc}"
                        ) from exc

            # Gentle breathing pause between batches to respect rate limits
            if i + batch_size < total_docs:
                await asyncio.sleep(1.0)

        logger.info("Successfully added all %d documents to vector store", added_count)
        return added_count

    async def similarity_search(
        self, query: str, k: int | None = None
    ) -> list[Document]:
        """Perform a similarity search against the vector store.

        Retrieves the top-k most similar documents to the given query
        using the configured embedding model.

        Args:
            query: The search query string.
            k: Number of results to return. Defaults to
                ``settings.TOP_K_RESULTS`` if not specified.

        Returns:
            A list of the most similar LangChain Document objects.

        Raises:
            RuntimeError: If the similarity search fails.
        """
        num_results = k if k is not None else self.settings.TOP_K_RESULTS

        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=num_results,
            )
            logger.info(
                "Similarity search for '%s' returned %d results",
                query[:50],
                len(results),
            )
            return results
        except Exception as exc:
            logger.error("Similarity search failed for query '%s': %s", query[:50], exc)
            raise RuntimeError(
                f"Similarity search failed: {exc}"
            ) from exc

    async def get_document_count(self) -> int:
        """Get the total number of documents in the vector store collection.

        Returns:
            The count of documents currently stored in the collection.
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            logger.debug("Vector store document count: %d", count)
            return count
        except Exception as exc:
            logger.error("Failed to get document count: %s", exc)
            raise RuntimeError(
                f"Failed to get document count: {exc}"
            ) from exc

    async def delete_collection(self) -> None:
        """Delete the entire vector store collection.

        Removes all documents and embeddings from the ChromaDB collection.
        This action is irreversible.
        """
        try:
            self.vector_store.delete_collection()
            logger.info(
                "Deleted collection '%s'",
                self.settings.CHROMA_COLLECTION_NAME,
            )
        except Exception as exc:
            logger.error(
                "Failed to delete collection '%s': %s",
                self.settings.CHROMA_COLLECTION_NAME,
                exc,
            )
            raise RuntimeError(
                f"Failed to delete collection "
                f"'{self.settings.CHROMA_COLLECTION_NAME}': {exc}"
            ) from exc
