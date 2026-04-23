from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(documents):
    """Nhúng text và lưu vào FAISS Vector Store"""
    
    # Khởi tạo mô hình embedding đa ngôn ngữ (tối ưu cho tiếng Việt)
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Tạo FAISS index
    vector_store = FAISS.from_documents(documents, embedder)
    return vector_store

def get_retriever(vector_store, search_type: str, k_value: int):
    """Cấu hình bộ truy xuất dữ liệu"""
    kwargs = {"k": k_value}
    if search_type == "mmr":
        kwargs["fetch_k"] = k_value * 4
        kwargs["lambda_mult"] = 0.7
        
    return vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)