from typing import List, Optional, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def _filter_documents(documents: List, filter_dict: Dict) -> List:
    """
    Lọc danh sách documents theo metadata.
    Dùng để BM25 chỉ tìm kiếm trong tập con phù hợp với filter.
    """
    if not filter_dict or not documents:
        return documents

    filtered = []
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        if all(meta.get(k) == v for k, v in filter_dict.items()):
            filtered.append(doc)

    # Nếu không có doc nào khớp, trả về toàn bộ để tránh lỗi
    return filtered if filtered else documents

def get_hybrid_retriever(
    vector_store,
    documents: List,
    k: int = 5,
    filter_dict: Optional[Dict] = None,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
):
    """
    Câu hỏi 7: Kết hợp FAISS (semantic search) + BM25 (keyword search).

    Args:
        vector_store  : FAISS vector store đã được build sẵn.
        documents     : Danh sách Document gốc (dùng cho BM25).
        k             : Số lượng kết quả mỗi retriever trả về.
        filter_dict   : Bộ lọc metadata (chỉ áp dụng cho FAISS và BM25).
        vector_weight : Trọng số cho semantic search (mặc định 0.6).
        bm25_weight   : Trọng số cho keyword search  (mặc định 0.4).

    Returns:
        EnsembleRetriever kết hợp cả hai.
    """
    # 1. Semantic retriever (FAISS)
    search_kwargs = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    # 2. Keyword retriever (BM25)
    bm25_docs = _filter_documents(documents, filter_dict)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    # 3. EnsembleRetriever – kết hợp hai nguồn
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    return ensemble_retriever


def get_retrieval_comparison(
    vector_store,
    documents: List,
    query: str,
    k: int = 5,
) -> Dict[str, Any]:
    """
    So sánh performance giữa:
      - Pure Vector Search (FAISS only)
      - Pure Keyword Search (BM25 only)
      - Hybrid Search (FAISS + BM25)

    Returns:
        dict với kết quả và metrics của từng phương pháp.
    """
    # Pure Vector Search
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    vector_docs = faiss_retriever.invoke(query)
    vector_contents = {d.page_content[:120] for d in vector_docs}

    # Pure BM25 (Keyword) Search
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    bm25_docs = bm25_retriever.invoke(query)
    bm25_contents = {d.page_content[:120] for d in bm25_docs}

    # Hybrid Search
    hybrid_retriever = get_hybrid_retriever(vector_store, documents, k=k)
    hybrid_docs = hybrid_retriever.invoke(query)
    hybrid_contents = {d.page_content[:120] for d in hybrid_docs}

    # Overlap Analysis
    overlap = {
        "vector_and_bm25_common": len(vector_contents & bm25_contents),
        "vector_only"           : len(vector_contents - bm25_contents),
        "bm25_only"             : len(bm25_contents - vector_contents),
        "hybrid_extra_vs_vector": len(hybrid_contents - vector_contents),
    }

    return {
        "vector": {"docs": vector_docs, "count": len(vector_docs)},
        "bm25"  : {"docs": bm25_docs,   "count": len(bm25_docs)},
        "hybrid": {"docs": hybrid_docs,  "count": len(hybrid_docs)},
        "overlap": overlap,
    }