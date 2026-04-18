from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.processing.metadata import enrich_documents_metadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: List[Document]) -> List[Document]:
        logger.info(
            "Splitting %s documents with chunk_size=%s, chunk_overlap=%s",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )
        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
        chunks = enrich_documents_metadata(chunks)
        logger.info("Created %s chunks", len(chunks))
        return chunks
