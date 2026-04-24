import logging
import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.schema import Document
import docx2txt   # ✅ thay thế python-docx

from src.document.splitter import split_documents

logger = logging.getLogger(__name__)

def process_document(file_path: str, chunk_size: int, chunk_overlap: int):
    """Đọc file PDF hoặc DOCX và trả về danh sách chunks"""

    ext = os.path.splitext(file_path)[1].lower()

    # ====================== LOAD FILE ======================
    if ext == ".pdf":
        loader = PDFPlumberLoader(file_path)
        raw_docs = loader.load()

    elif ext == ".docx":
        # ✅ dùng docx2txt thay vì python-docx
        full_text = docx2txt.process(file_path)

        # xử lý trường hợp file rỗng
        if not full_text or full_text.strip() == "":
            raise ValueError("File DOCX không có nội dung!")

        raw_docs = [Document(page_content=full_text)]

    else:
        raise ValueError("Chỉ hỗ trợ PDF và DOCX")

    # ====================== SPLIT ======================
    documents = split_documents(raw_docs, chunk_size, chunk_overlap)

    return documents, len(documents)