"""
Câu hỏi 9 - Re-ranking với Cross-Encoder
=========================================
Module này thêm bước đánh giá và sắp xếp lại thứ tự các chunk
sau khi retrieval để tối ưu hóa độ liên quan.

Pipeline:
  Query → Vector Retrieval (Bi-encoder) → Re-ranking (Cross-Encoder) → Top-K chunks → LLM
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Đại diện cho một đoạn văn bản được truy xuất từ cơ sở dữ liệu."""
    chunk_id: str
    text: str
    source: str                    # tên file / tài liệu
    page: int = 0
    initial_score: float = 0.0    # điểm từ vector search (bi-encoder)
    rerank_score: float = 0.0     # điểm sau cross-encoder
    rank_before: int = 0
    rank_after: int = 0


@dataclass
class RerankedResult:
    """Kết quả sau khi re-rank, bao gồm các chunk đã được sắp xếp lại."""
    query: str
    chunks: List[Chunk]
    top_k: int
    elapsed_ms: float = 0.0

    @property
    def top_chunks(self) -> List[Chunk]:
        return self.chunks[:self.top_k]


# ---------------------------------------------------------------------------
# Cross-Encoder Re-ranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Re-ranker sử dụng mô hình Cross-Encoder để tính điểm liên quan
    giữa query và từng chunk.

    Cross-Encoder khác Bi-Encoder:
      - Bi-Encoder: encode query và document riêng → cosine similarity (nhanh, scale tốt)
      - Cross-Encoder: đưa cả (query, document) vào cùng 1 lần → attention đầy đủ (chậm hơn, chính xác hơn)

    Do đó Cross-Encoder thường được dùng ở bước 2 (re-rank top-N từ bi-encoder)
    thay vì chạy trên toàn bộ corpus.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy load

    def _load_model(self):
        """Lazy-load model để tránh import lúc khởi động."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            print(f"[Re-ranker] Đang tải mô hình: {self.model_name} ...")
            self._model = CrossEncoder(self.model_name, device=self.device)
            print(f"[Re-ranker] Tải xong.")
        except ImportError:
            raise ImportError(
                "Cần cài đặt: pip install sentence-transformers\n"
                "Hoặc dùng FallbackReranker nếu chưa có GPU/RAM đủ lớn."
            )

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 5,
    ) -> RerankedResult:
        """
        Sắp xếp lại danh sách chunk theo điểm Cross-Encoder.

        Args:
            query:  câu hỏi của người dùng
            chunks: danh sách chunk từ vector retrieval
            top_k:  số chunk giữ lại sau re-rank

        Returns:
            RerankedResult với chunks đã được sắp xếp lại
        """
        if not chunks:
            return RerankedResult(query=query, chunks=[], top_k=top_k)

        self._load_model()

        # Đánh số thứ tự gốc
        for i, c in enumerate(chunks):
            c.rank_before = i + 1

        # Tạo pairs (query, chunk_text) để đưa vào cross-encoder
        pairs = [(query, c.text) for c in chunks]

        start = time.perf_counter()
        scores = self._model.predict(pairs)          # numpy array
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Gán điểm và sắp xếp
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        for i, c in enumerate(sorted_chunks):
            c.rank_after = i + 1

        print(f"[Re-ranker] Re-ranked {len(chunks)} chunks → giữ top {top_k} | {elapsed_ms:.1f} ms")

        return RerankedResult(
            query=query,
            chunks=sorted_chunks,
            top_k=top_k,
            elapsed_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Fallback Re-ranker (không cần GPU, dùng keyword overlap)
# ---------------------------------------------------------------------------

class KeywordReranker:
    """
    Fallback Re-ranker đơn giản dựa trên keyword overlap.
    Dùng khi không có sentence-transformers hoặc để test nhanh.
    Không chính xác bằng Cross-Encoder nhưng không cần model nặng.
    """

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 5,
    ) -> RerankedResult:
        query_tokens = set(query.lower().split())

        for i, chunk in enumerate(chunks):
            chunk.rank_before = i + 1
            chunk_tokens = set(chunk.text.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            # Kết hợp overlap với điểm gốc
            chunk.rerank_score = overlap * 0.6 + chunk.initial_score * 0.4

        sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        for i, c in enumerate(sorted_chunks):
            c.rank_after = i + 1

        return RerankedResult(query=query, chunks=sorted_chunks, top_k=top_k)


# ---------------------------------------------------------------------------
# Pipeline tích hợp: Retrieval → Re-rank
# ---------------------------------------------------------------------------

class RAGRerankerPipeline:
    """
    Pipeline hoàn chỉnh: nhận query → truy xuất chunks → re-rank → trả về top-K.

    Tích hợp với retriever hiện có của dự án bằng cách override phương thức
    `retrieve_chunks`.
    """

    def __init__(
        self,
        reranker: Optional[CrossEncoderReranker | KeywordReranker] = None,
        retrieve_top_n: int = 20,   # số chunk lấy từ vector DB
        rerank_top_k: int = 5,      # số chunk giữ lại sau re-rank
    ):
        self.reranker = reranker or KeywordReranker()
        self.retrieve_top_n = retrieve_top_n
        self.rerank_top_k = rerank_top_k

    def retrieve_chunks(self, query: str) -> List[Chunk]:
        """
        [GHI ĐÈ phương thức này] Gọi vector database của dự án và trả về List[Chunk].

        Ví dụ tích hợp với FAISS / ChromaDB / Qdrant:
            results = vector_db.similarity_search(query, k=self.retrieve_top_n)
            return [Chunk(chunk_id=r.id, text=r.page_content,
                          source=r.metadata['source'],
                          page=r.metadata.get('page', 0),
                          initial_score=r.score)
                    for r in results]
        """
        raise NotImplementedError("Cần implement retrieve_chunks() cho dự án của bạn.")

    def run(self, query: str) -> RerankedResult:
        """Chạy toàn bộ pipeline retrieval → re-rank."""
        chunks = self.retrieve_chunks(query)
        result = self.reranker.rerank(query, chunks, top_k=self.rerank_top_k)
        return result

    def format_context(self, result: RerankedResult) -> str:
        """Định dạng các chunk thành context string để đưa vào LLM prompt."""
        parts = []
        for i, chunk in enumerate(result.top_chunks, 1):
            parts.append(
                f"[Đọc từ tài liệu - Nguồn: {chunk.source}, Trang: {chunk.page}]\n"
                f"{chunk.text}\n"
                f"(Điểm liên quan: {chunk.rerank_score:.4f})"
            )
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------------------------

def demo():
    """Minh họa pipeline re-ranking với dữ liệu giả."""

    # Mô phỏng chunks từ vector retrieval
    mock_chunks = [
        Chunk(
            chunk_id="c1", text="Điều 15: Người lao động có quyền nghỉ phép năm 12 ngày sau 1 năm làm việc.",
            source="luat_lao_dong.pdf", page=42, initial_score=0.81
        ),
        Chunk(
            chunk_id="c2", text="Chương 3 quy định về hợp đồng lao động và các điều khoản chấm dứt.",
            source="luat_lao_dong.pdf", page=18, initial_score=0.76
        ),
        Chunk(
            chunk_id="c3", text="Người lao động được nghỉ ốm tối đa 30 ngày mỗi năm theo quy định bảo hiểm xã hội.",
            source="bhxh_2024.pdf", page=5, initial_score=0.74
        ),
        Chunk(
            chunk_id="c4", text="Điều 113: Số ngày nghỉ phép năm tính theo thâm niên, tăng thêm 1 ngày mỗi 5 năm.",
            source="luat_lao_dong.pdf", page=45, initial_score=0.72
        ),
        Chunk(
            chunk_id="c5", text="Mức lương tối thiểu vùng áp dụng từ ngày 01/07/2024 theo Nghị định 74/2024.",
            source="nghi_dinh_74.pdf", page=2, initial_score=0.65
        ),
    ]

    query = "Người lao động được nghỉ phép bao nhiêu ngày mỗi năm?"

    print("=" * 65)
    print("DEMO: Re-ranking với Cross-Encoder")
    print("=" * 65)
    print(f"Query: {query}\n")
    print("Thứ tự BAN ĐẦU (từ vector search):")
    for i, c in enumerate(mock_chunks, 1):
        print(f"  {i}. [{c.initial_score:.2f}] {c.text[:70]}...")

    # Dùng KeywordReranker (không cần model để chạy demo)
    reranker = KeywordReranker()
    result = reranker.rerank(query, mock_chunks, top_k=3)

    print("\nThứ tự SAU KHI RE-RANK (top 3):")
    for chunk in result.top_chunks:
        arrow = "↑" if chunk.rank_after < chunk.rank_before else ("↓" if chunk.rank_after > chunk.rank_before else "=")
        print(
            f"  Rank {chunk.rank_after} {arrow} (trước: {chunk.rank_before}) "
            f"| score={chunk.rerank_score:.4f} "
            f"| {chunk.text[:60]}..."
        )

    print("\nContext đưa vào LLM:")
    pipeline = RAGRerankerPipeline(reranker=reranker, rerank_top_k=3)
    # Giả lập đã có chunks, gọi thẳng format_context
    print(pipeline.format_context(result))


if __name__ == "__main__":
    demo()