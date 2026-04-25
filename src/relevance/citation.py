from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re


@dataclass
class Citation:
    citation_id: int
    source_file: str
    page: int
    chunk_id: str
    excerpt: str
    rerank_score: float = 0.0
    section: str = ""

    @property
    def label(self) -> str:
        return f"[{self.citation_id}]"


class CitationBuilder:
    """
    Xây danh sách Citation từ List[Chunk] đã re-rank.
    Mọi nội dung đều được ghi rõ "đọc từ tài liệu".
    """

    MAX_EXCERPT_LEN = 200

    def build_from_chunks(self, chunks: List) -> List[Citation]:
        """
        Nhận List[Chunk] (từ reranker.py), trả về List[Citation].
        Mỗi chunk → 1 citation với id tuần tự [1], [2], ...
        """
        citations = []
        for i, chunk in enumerate(chunks, 1):
            excerpt = chunk.text[:self.MAX_EXCERPT_LEN]
            if len(chunk.text) > self.MAX_EXCERPT_LEN:
                excerpt += "..."

            citations.append(Citation(
                citation_id=i,
                source_file=chunk.source,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                excerpt=excerpt,
                rerank_score=getattr(chunk, "rerank_score", 0.0),
                section=getattr(chunk, "section", ""),
            ))
        return citations

    def build_context(self, chunks: List) -> str:
        """
        Tạo context string để đưa vào LLM prompt.
        Mỗi chunk được đánh số [1], [2], ... và ghi rõ "đọc từ tài liệu".
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Đọc từ tài liệu: «{chunk.source}» — Trang {chunk.page}\n"
                f"{chunk.text}"
            )
        return "\n\n".join(parts)

    def build_system_prompt_addition(self) -> str:
        """
        Đoạn thêm vào system prompt để LLM luôn nhúng citation [1][2]...
        """
        return (
            "HƯỚNG DẪN TRÍCH DẪN NGUỒN:\n"
            "- Mỗi thông tin trong câu trả lời PHẢI được gắn số trích dẫn [1], [2], ... "
            "tương ứng với tài liệu tham khảo.\n"
            "- Ví dụ: \"Người lao động được nghỉ phép 12 ngày mỗi năm [1].\"\n"
            "- KHÔNG bịa đặt thông tin ngoài ngữ cảnh được cung cấp.\n"
            "- Nếu không có trong tài liệu, hãy nói rõ: "
            "\"Tài liệu không đề cập đến vấn đề này.\"\n"
            "- Tất cả nội dung đều được đọc từ tài liệu gốc."
        )