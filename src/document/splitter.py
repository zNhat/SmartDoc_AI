from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(raw_docs, chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""] # Ưu tiên cắt ở đoạn văn, rồi mới đến dòng
    )
    
    chunks = text_splitter.split_documents(raw_docs)
    
    # In log kiểm tra
    print(f"Đã cắt {len(raw_docs)} tài liệu thành {len(chunks)} đoạn nhỏ.")
    
    return chunks