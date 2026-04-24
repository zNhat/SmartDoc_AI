from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(raw_docs, chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_documents(raw_docs)