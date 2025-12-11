from langchain_community.document_loaders import Docx2txtLoader
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_core.document_loaders import BaseLoader
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from pathlib import Path
from typing import Type
import bisect

from app.exception import IngestionError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class IngestionService:
    SUPPORTED_LOADERS: dict[str, Type[BaseLoader]] = {
        ".pdf": DoclingLoader,
        ".docx": Docx2txtLoader,
    }

    def __init__(self):
        pass

    def _load(self, file_path: str) -> list[tuple[int, str]]:
        """Take a file_path and extract pages from the document"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in self.SUPPORTED_LOADERS:
            logger.error(f"Unsupported file type attempted: {file_ext}")
            raise IngestionError(f"Unsupported file type: {file_ext}")

        loader_class = self.SUPPORTED_LOADERS[file_ext]
        loader = loader_class(file_path)

        if file_ext == ".pdf":
            # Get Page no. for PDF files
            doc = loader.load()
            pages = [(i + 1, page.page_content) for i, page in enumerate(doc)]
            return pages
        else:
            # Just add Page 1 for other pageless documents
            text = loader.load()[0].page_content
            return [(1, text)]

    def _chunk(self, pages: list[tuple[int, str]], source_file: str) -> list[Document]:
        full_text = ""
        page_boundaries = []  # Store (char_index, page_number)

        current_char_index = 0
        for page_number, page_text in pages:
            page_boundaries.append(current_char_index)

            full_text += page_text

            current_char_index += len(page_text)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=50,
            add_start_index=True,
        )

        chunks = splitter.create_documents([full_text])

        final_chunks = []

        for chunk in chunks:
            start_index = chunk.metadata["start_index"]

            page_index = bisect.bisect_right(page_boundaries, start_index) - 1

            # Get the actual page number from document
            original_page_number = pages[page_index][0]

            chunk.metadata["source"] = source_file
            chunk.metadata["page"] = original_page_number

            final_chunks.append(chunk)

        return final_chunks

    def ingest(self, file_paths: list[str], vectorstore: VectorStore):
        for file_path in file_paths:
            pages = self._load(file_path)
            filename = Path(file_path).name

            chunks = self._chunk(pages, source_file=filename)

            try:
                vectorstore.add_documents(chunks)
            except GoogleGenerativeAIError as e:
                logger.exception("Google GenAI embedding failed during ingestion")
                raise IngestionError(
                    "Please check your Google Generative AI API key."
                ) from e
            except Exception as e:
                logger.exception("Unexpected ingestion error")
                raise IngestionError("Failed to add documents to vectorstore.") from e
