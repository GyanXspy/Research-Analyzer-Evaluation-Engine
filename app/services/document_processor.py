"""Document processing service for PDF ingestion and text chunking.

Handles PDF text extraction via PyMuPDF and splits content into
semantically meaningful chunks using LangChain's recursive text splitter.
"""

import logging
import os
from pathlib import Path

import fitz  # pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes uploaded documents into chunked LangChain Document objects.

    Extracts text from PDFs using PyMuPDF and splits the content into
    overlapping chunks suitable for embedding and retrieval.

    Attributes:
        settings: Application settings containing chunk size, overlap, and upload directory.
        text_splitter: Configured recursive text splitter for chunking documents.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the document processor.

        Args:
            settings: Application settings with CHUNK_SIZE, CHUNK_OVERLAP,
                and UPLOAD_DIR configuration values.
        """
        self.settings = settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            "DocumentProcessor initialized (chunk_size=%d, chunk_overlap=%d)",
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP,
        )

    async def process_pdf(self, file_path: str, filename: str) -> list[Document]:
        """Extract text from a PDF and split it into chunked Document objects.

        Opens the PDF with PyMuPDF, extracts text page-by-page, then splits
        each page's text into overlapping chunks. Each chunk is returned as a
        LangChain Document with source metadata attached.

        Args:
            file_path: Absolute or relative path to the PDF file on disk.
            filename: Original filename for metadata tagging.

        Returns:
            A list of LangChain Document objects, each containing a text chunk
            and metadata with filename, page_number, and chunk_id.

        Raises:
            FileNotFoundError: If the PDF file does not exist at the given path.
            RuntimeError: If PyMuPDF fails to open or read the PDF.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error("PDF file not found: %s", file_path)
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents: list[Document] = []
        chunk_id = 0

        try:
            logger.info("Processing PDF: %s", filename)
            pdf_document = fitz.open(file_path)

            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                page_text = page.get_text()

                if not page_text.strip():
                    logger.debug(
                        "Skipping empty page %d in %s", page_number + 1, filename
                    )
                    continue

                # Split the page text into chunks
                chunks = self.text_splitter.split_text(page_text)

                for chunk_text in chunks:
                    doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "filename": filename,
                            "page_number": page_number + 1,
                            "chunk_id": chunk_id,
                            "source": filename,
                        },
                    )
                    documents.append(doc)
                    chunk_id += 1

            pdf_document.close()
            logger.info(
                "Successfully processed %s: %d chunks from %d pages",
                filename,
                len(documents),
                len(pdf_document) if not pdf_document.is_closed else page_number + 1,
            )

        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Failed to process PDF %s: %s", filename, exc)
            raise RuntimeError(f"Failed to process PDF '{filename}': {exc}") from exc

        return documents

    async def process_uploaded_file(
        self, file_content: bytes, filename: str
    ) -> str:
        """Save uploaded file content to the configured upload directory.

        Creates the upload directory if it does not already exist, then writes
        the raw file bytes to disk.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: Name to save the file as (including extension).

        Returns:
            The absolute file path where the file was saved.

        Raises:
            OSError: If the file cannot be written to disk.
        """
        try:
            upload_dir = Path(self.settings.UPLOAD_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)

            file_path = upload_dir / filename
            file_path.write_bytes(file_content)

            logger.info("Saved uploaded file to: %s", file_path)
            return str(file_path)

        except OSError as exc:
            logger.error("Failed to save uploaded file %s: %s", filename, exc)
            raise
