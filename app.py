import streamlit as st
import os
import tempfile
import uuid

from src.utils.storage import save_sessions_to_disk, load_sessions_from_disk
from src.ui.sidebar import render_sidebar
from src.ui.chat_area import render_chat_history
from src.document.loader import process_document   # ✅ đổi import
from src.retriever.faiss_store import create_vector_store, save_vector_store, load_vector_store, get_retriever
from src.llm.generator import generate_answer

# ====================== KHỞI TẠO ======================
st.set_page_config(page_title="SmartDoc AI", page_icon="🤖", layout="wide")

if "all_sessions" not in st.session_state:
    saved_data = load_sessions_from_disk()
    if saved_data:
        st.session_state.all_sessions = saved_data
        st.session_state.current_session_id = list(saved_data.keys())[0]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.all_sessions = {
            new_id: {
                "title": "Cuộc trò chuyện mới",
                "messages": [],
                "vector_store": None,
                "file_name": None
            }
        }
        st.session_state.current_session_id = new_id

curr_id = st.session_state.current_session_id
curr_session = st.session_state.all_sessions[curr_id]

# Nạp Vector Store từ ổ cứng
if curr_session["vector_store"] is None and curr_session["file_name"] is not None:
    loaded_vs = load_vector_store(curr_id)
    if loaded_vs:
        st.session_state.all_sessions[curr_id]["vector_store"] = loaded_vs

# ====================== 1. SIDEBAR ======================
config_params = render_sidebar()

# ====================== 2. QUẢN LÝ TÀI LIỆU ======================
st.title("📄 SmartDoc AI")

has_doc = curr_session["file_name"] is not None

with st.expander("📁 Quản lý Tài liệu", expanded=not has_doc):

    # ✅ hỗ trợ PDF + DOCX
    uploaded_file = st.file_uploader(
        f"Tải lên tài liệu cho phiên: {curr_session['title']}",
        type=['pdf', 'docx']
    )

    if uploaded_file is not None:
        # ✅ đảm bảo re-process khi file mới
        if curr_session["file_name"] != uploaded_file.name:
            with st.spinner("Đang xử lý tài liệu..."):
                try:
                    # tạo file tạm với đúng đuôi
                    suffix = os.path.splitext(uploaded_file.name)[1]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # ✅ gọi hàm mới
                    documents, chunk_count = process_document(
                        tmp_path,
                        config_params["chunk_size"],
                        config_params["chunk_overlap"]
                    )

                    vector_store = create_vector_store(documents)
                    save_vector_store(vector_store, curr_id)

                    st.session_state.all_sessions[curr_id]['vector_store'] = vector_store
                    st.session_state.all_sessions[curr_id]['file_name'] = uploaded_file.name
                    save_sessions_to_disk(st.session_state.all_sessions)

                    # ✅ chỉ log 1 lần duy nhất (đúng yêu cầu đề)
                    st.success(f"📦 Đã cắt tài liệu thành {chunk_count} chunks!")

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")

                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)

    # Hiển thị trạng thái file
    if curr_session["file_name"]:
        st.info(f"📄 Tài liệu đang liên kết: **{curr_session['file_name']}**")

st.divider()

# ====================== 3. KHU VỰC CHAT ======================
render_chat_history(curr_session["messages"])

if prompt_text := st.chat_input("Nhập câu hỏi..."):
    if curr_session["vector_store"] is None:
        st.error("Vui lòng tải lên tài liệu trước!")
    else:
        if len(curr_session["messages"]) == 0:
            st.session_state.all_sessions[curr_id]["title"] = prompt_text[:25] + "..."

        st.session_state.all_sessions[curr_id]["messages"].append({
            "role": "user",
            "content": prompt_text
        })

        save_sessions_to_disk(st.session_state.all_sessions)

        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu và suy nghĩ..."):
                try:
                    retriever = get_retriever(
                        curr_session['vector_store'],
                        config_params["search_type"],
                        config_params["k_value"]
                    )

                    relevant_docs = retriever.invoke(prompt_text)

                    answer = generate_answer(
                        prompt_text,
                        relevant_docs,
                        config_params["llm_model"]
                    )

                    st.markdown(answer)

                    source_texts = [doc.page_content for doc in relevant_docs]

                    with st.expander("📑 Xem nguồn tham khảo"):
                        for i, doc_text in enumerate(source_texts, 1):
                            st.markdown(f"**Đoạn {i} (đọc từ tài liệu):**")
                            st.caption(doc_text)

                    st.session_state.all_sessions[curr_id]["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_texts
                    })

                    save_sessions_to_disk(st.session_state.all_sessions)
                    st.rerun()

                except Exception as e:
                    st.error("❌ Lỗi AI.")
                    st.exception(e)