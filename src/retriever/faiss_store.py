import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. KHỞI TẠO EMBEDDING 
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def create_vector_store(documents):
    """Nhúng text và tạo FAISS Vector Store trong RAM"""
    vector_store = FAISS.from_documents(documents, embedder)
    return vector_store

def save_vector_store(vector_store, session_id):
    """
    Lưu Vector Store xuống ổ cứng tại: vector_store/[session_id]/
    """
    base_path = "vector_store"
    # Tạo thư mục gốc nếu chưa có
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Tạo đường dẫn riêng cho phiên chat
    save_path = os.path.join(base_path, session_id)
    vector_store.save_local(save_path)
    return save_path

def load_vector_store(session_id):
    """
    Nạp lại Vector Store từ ổ cứng nếu tồn tại
    """
    load_path = os.path.join("vector_store", session_id)
    index_file = os.path.join(load_path, "index.faiss")
    
    if os.path.exists(index_file):
        return FAISS.load_local(
            load_path, 
            embedder, 
            allow_dangerous_deserialization=True
        )
    return None

def get_retriever(vector_store, search_type: str, k_value: int):
    """Cấu hình bộ truy xuất dữ liệu"""
    kwargs = {"k": k_value}
    if search_type == "mmr":
        kwargs["fetch_k"] = k_value * 4
        kwargs["lambda_mult"] = 0.7
        
    return vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)