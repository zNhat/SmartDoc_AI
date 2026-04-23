import streamlit as st
import os
import tempfile
import uuid

# Import UI
from src.ui.sidebar import render_sidebar
from src.ui.chat_area import render_chat_history

# Import Core RAG Logic
from src.document.loader import process_pdf
from src.retriever.faiss_store import create_vector_store, get_retriever
from src.llm.generator import generate_answer

# ====================== KHỞI TẠO STATE (MULTI-SESSION) ======================
st.set_page_config(page_title="SmartDoc AI", page_icon="🤖", layout="wide")

# Khởi tạo từ điển chứa tất cả các phiên với cấu trúc lưu trữ tài liệu riêng biệt
if "all_sessions" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.all_sessions = {
        new_id: {
            "title": "Cuộc trò chuyện mới", 
            "messages": [],
            "vector_store": None,  # Lưu FAISS Vector Store riêng cho phiên này
            "file_name": None      # Lưu tên file riêng cho phiên này
        }
    }
    st.session_state.current_session_id = new_id

# Lấy ID và dữ liệu của phiên hiện tại
curr_id = st.session_state.current_session_id
curr_session = st.session_state.all_sessions[curr_id]

# ====================== SIDEBAR ======================
# Hàm render_sidebar trả về các tham số config (chunk_size, k_value, etc.)
config_params = render_sidebar()

# ====================== MAIN AREA ======================
st.title("📄 SmartDoc AI")

# Kiểm tra xem phiên hiện tại đã có tài liệu chưa để mở/đóng expander       
has_doc = curr_session["vector_store"] is not None

with st.expander("📁 Quản lý Tài liệu", expanded=not has_doc):
    uploaded_file = st.file_uploader(f"Tải lên PDF cho phiên: {curr_session['title']}", type=['pdf'])
    
    if uploaded_file is not None:
        # Chỉ xử lý nếu file mới khác với file đã lưu trong phiên này
        if curr_session["file_name"] != uploaded_file.name:
            with st.spinner("Đang xử lý tài liệu cho riêng phiên chat này..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Thực hiện Document Processing Flow
                    documents = process_pdf(tmp_path, config_params["chunk_size"], config_params["chunk_overlap"])
                    vector_store = create_vector_store(documents)

                    # Lưu trực tiếp vào ngăn chứa của phiên hiện tại
                    st.session_state.all_sessions[curr_id]['vector_store'] = vector_store
                    st.session_state.all_sessions[curr_id]['file_name'] = uploaded_file.name
                    
                    st.success(f"✅ Tài liệu '{uploaded_file.name}' đã sẵn sàng cho phiên này!")
                    st.rerun()

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    
    # Hiển thị trạng thái tài liệu của phiên hiện tại
    if curr_session["file_name"]:
        st.info(f"📄 Phiên này đang sử dụng tài liệu: **{curr_session['file_name']}**")

st.divider()

# ====================== KHU VỰC CHAT ======================
# Truyền danh sách tin nhắn của phiên hiện hành vào hàm hiển thị
render_chat_history(curr_session["messages"])

if prompt_text := st.chat_input("Nhập câu hỏi của bạn vào đây..."):
    # Kiểm tra vector store trong phiên hiện tại
    if curr_session["vector_store"] is None:
        st.error("Vui lòng tải lên tài liệu PDF cho phiên chat này trước!")
    else:
        # Cập nhật tiêu đề phiên nếu đây là câu hỏi đầu tiên
        if len(curr_session["messages"]) == 0:
            short_title = prompt_text[:30] + ("..." if len(prompt_text) > 30 else "")
            st.session_state.all_sessions[curr_id]["title"] = short_title

        # Lưu câu hỏi vào phiên hiện tại
        st.session_state.all_sessions[curr_id]["messages"].append({"role": "user", "content": prompt_text})
            
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # XỬ LÝ AI RAG (Query Processing Flow)
        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu tài liệu và suy nghĩ..."):
                try:
                    # Truy xuất từ Vector Store riêng của phiên này
                    retriever = get_retriever(
                        curr_session['vector_store'], 
                        config_params["search_type"], 
                        config_params["k_value"]
                    )
                    relevant_docs = retriever.invoke(prompt_text)
                    
                    # Sinh câu trả lời qua mô hình LLM (Ollama)
                    answer = generate_answer(prompt_text, relevant_docs, config_params["llm_model"])
                    
                    st.markdown(answer)
                    
                    source_texts = [doc.page_content for doc in relevant_docs]
                    with st.expander("📑 Xem nguồn tham khảo"):
                        for i, doc_text in enumerate(source_texts, 1):
                            # Đảm bảo nội dung citation được ghi nhận đúng
                            st.markdown(f"**Đoạn {i} (chỗ này ghi lại nói là đọc từ tài liệu):**")
                            st.caption(doc_text)

                    # Lưu câu trả lời vào phiên hiện tại
                    st.session_state.all_sessions[curr_id]["messages"].append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": source_texts
                    })
                    
                    st.rerun()

                except Exception as e:
                    st.error("❌ Lỗi xử lý AI. Hãy kiểm tra lại kết nối Ollama.")
                    st.error(f"Chi tiết: {str(e)}")