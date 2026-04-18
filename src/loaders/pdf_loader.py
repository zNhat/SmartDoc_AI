from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFDocumentLoader:
    """Load a PDF file and normalize metadata for downstream RAG usage."""

    def load(self, file_path: str | Path) -> List[Document]:
        file_path = str(file_path)
        logger.info("Loading PDF: %s", file_path)
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()

        normalized_docs: List[Document] = []
        for idx, doc in enumerate(docs):
            metadata = dict(doc.metadata or {})
            metadata["source"] = metadata.get("source", file_path)
            metadata["file_name"] = Path(file_path).name
            metadata["file_type"] = "pdf"
            metadata["page"] = metadata.get("page", idx + 1)
            metadata["page_label"] = f"Page {metadata['page']}"
            normalized_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        logger.info("Loaded %s PDF pages", len(normalized_docs))
        return normalized_docs
