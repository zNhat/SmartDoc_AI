from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingFactory:
    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self.model_name = model_name or settings.embedding_model
        self.device = device

    def create(self) -> HuggingFaceEmbeddings:
        logger.info("Loading embedding model: %s", self.model_name)
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
