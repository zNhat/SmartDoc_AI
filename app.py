import streamlit as st
import os
import tempfile
import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ====================== CONFIG LOGGING (7.2.5) ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="SmartDoc AI", page_icon="🤖", layout="wide")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⚙️ Cài đặt & Thông tin")
    st.info("Vector Store: **FAISS**")
    
    st.subheader("🔧 Tùy chỉnh tham số")
    
    # 7.2.2 Chunk parameters
    chunk_size = st.slider("Chunk Size", 500, 2000, 1500, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, 200, 50)
    
    # 7.2.4 Retrieval parameters
    search_type = st.selectbox("Phương thức tìm kiếm", ["similarity", "mmr"], index=0)
    k_value = st.slider("Số đoạn lấy về (k)", 3, 8, 5)
    
    # 7.2.3 Thay đổi LLM model
    llm_model = st.selectbox(
        "Mô hình LLM",
        ["qwen2.5:7b", "llama2:7b", "mistral:7b"],
        index=0
    )
    
    st.divider()
    st.markdown("""
    ### Hướng dẫn sử dụng:
    1. Tải lên file PDF  
    2. Điều chỉnh tham số (chunk sẽ tự cắt lại khi thay đổi)  
    3. Đặt câu hỏi
    """)
    st.caption("Bài tập lớn OSSD 2026 - SmartDoc AI")

# ====================== MAIN AREA ======================
st.title("📄 SmartDoc AI - Intelligent Document Q&A System")
st.markdown("Hệ thống tra cứu tài liệu thông minh sử dụng **RAG**.")

# ====================== FILE UPLOAD ======================
st.subheader("1. Tải lên tài liệu")
uploaded_file = st.file_uploader("Chọn một tệp PDF", type=['pdf'])

# ====================== XỬ LÝ TÀI LIỆU ======================
if uploaded_file is not None:
    # Kiểm tra xem có cần re-process không (chunk thay đổi hoặc chưa có vector store)
    need_reprocess = (
        "vector_store" not in st.session_state or
        st.session_state.get("current_file") != uploaded_file.name or
        st.session_state.get("last_chunk_size") != chunk_size or
        st.session_state.get("last_chunk_overlap") != chunk_overlap
    )

    if need_reprocess:
        with st.spinner("Đang xử lý / cắt lại tài liệu với tham số mới..."):
            try:
                # Lưu file tạm để đọc
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                logger.info(f"Processing file: {uploaded_file.name} | Chunk: {chunk_size}/{chunk_overlap}")

                # Load PDF (chỉ load 1 lần)
                loader = PDFPlumberLoader(tmp_path)
                raw_docs = loader.load()

                # Split với tham số mới
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                documents = text_splitter.split_documents(raw_docs)

                # Embedding
                embedder = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

                # Tạo vector store mới
                vector_store = FAISS.from_documents(documents, embedder)

                # Lưu vào session_state
                st.session_state['vector_store'] = vector_store
                st.session_state['raw_docs'] = raw_docs          # lưu để tái sử dụng
                st.session_state['current_file'] = uploaded_file.name
                st.session_state['last_chunk_size'] = chunk_size
                st.session_state['last_chunk_overlap'] = chunk_overlap

                logger.info(f" Processing {len(documents)} chunks ")
                st.success(f"✅ Tài liệu đã sẵn sàng!")

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        st.success("✅ Tài liệu đã sẵn sàng!")

st.divider()

# ====================== QUESTION SECTION ======================
st.subheader("2. Đặt câu hỏi")
user_question = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Tóm tắt nội dung chính của tài liệu...")

if user_question:
    if uploaded_file is None:
        st.warning("Vui lòng tải lên tài liệu PDF trước!")
    elif "vector_store" not in st.session_state:
        st.warning("Vui lòng đợi tài liệu được xử lý xong!")
    else:
        logger.info(f"Query: {user_question}")

        with st.spinner("Đang sinh câu trả lời..."):
            try:
                vector_store = st.session_state['vector_store']
                
                # Retriever
                kwargs = {"k": k_value}
                if search_type == "mmr":
                    kwargs["fetch_k"] = k_value * 4
                    kwargs["lambda_mult"] = 0.7

                retriever = vector_store.as_retriever(
                    search_type=search_type,
                    search_kwargs=kwargs
                )
                
                relevant_docs = retriever.invoke(user_question)
                logger.info(f"Retrieved {len(relevant_docs)} documents")

                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Prompt
                vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
                is_vietnamese = any(char in user_question.lower() for char in vietnamese_chars)

                if is_vietnamese:
                    prompt_template = """Sử dụng ngữ cảnh sau đây để trả lời đầy đủ và chính xác.
                        Hãy tổng hợp thông tin, dùng gạch đầu dòng nếu cần.
                        Nếu không có thông tin, hãy nói rõ.

                        Ngữ cảnh: {context}

                        Câu hỏi: {user_input}

                        Trả lời:"""
                else:
                    prompt_template = """Use the following context to provide a comprehensive answer.
                        Synthesize information and use bullet points when appropriate.
                        If not in the context, clearly state so.

                        Context: {context}

                        Question: {user_input}

                        Answer:"""

                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_input"])

                llm = Ollama(
                    model=llm_model,
                    temperature=0.3,
                    top_p=0.9,
                    repeat_penalty=1.1
                )

                final_prompt = prompt.format(context=context, user_input=user_question)
                answer = llm.invoke(final_prompt)

                st.markdown("### 💡 Trả lời:")
                st.info(answer)

                with st.expander("📑 Xem nguồn tham khảo"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Đoạn {i}:**")
                        st.caption(doc.page_content)

            except Exception as e:
                st.error("❌ Lỗi khi gọi LLM. Kiểm tra Ollama đang chạy.")
                st.error(f"Lỗi: {str(e)}")
                logger.error(f"Error: {e}")

st.caption("SmartDoc AI - Built with LangChain + Ollama + Streamlit | Spring 2026")