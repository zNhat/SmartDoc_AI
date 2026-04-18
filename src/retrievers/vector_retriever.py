from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.vectorstore: Optional[FAISS] = None

    def build(self, documents: List[Document]) -> FAISS:
        logger.info("Building FAISS vector store from %s chunks", len(documents))
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore

    def save(self, save_dir: str | Path) -> None:
        if self.vectorstore is None:
            return
        logger.info("Saving vector store to %s", save_dir)
        self.vectorstore.save_local(str(save_dir))

    def load(self, save_dir: str | Path) -> Optional[FAISS]:
        save_dir = Path(save_dir)
        index_file = save_dir / "index.faiss"
        pkl_file = save_dir / "index.pkl"
        if not index_file.exists() or not pkl_file.exists():
            return None
        logger.info("Loading vector store from %s", save_dir)
        self.vectorstore = FAISS.load_local(
            str(save_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    def search(self, query: str, k: int = 4, filters: dict | None = None) -> List[Document]:
        if self.vectorstore is None:
            return []
        if filters:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k, filter=filters)
            return [doc for doc, _ in docs_and_scores]
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 4):
        if self.vectorstore is None:
            raise ValueError("Vector store has not been built yet.")
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
