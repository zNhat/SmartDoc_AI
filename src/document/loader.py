import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def process_pdf(file_path: str, chunk_size: int, chunk_overlap: int):
    """Đọc file PDF và cắt thành các chunk nhỏ"""
    
    # 1. Đọc nội dung
    loader = PDFPlumberLoader(file_path)
    raw_docs = loader.load()
    
    # 2. Cắt chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(raw_docs)
    
    # Tuân thủ yêu cầu: Chỉ hiện log thông báo 1 lần cuối cùng sau khi cắt xong
    logger.info(f"Hoàn tất xử lý: Đã cắt thành công {len(documents)} chunks.")
    
    return documents