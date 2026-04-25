import streamlit as st
import os
import tempfile
import uuid
from src.relevance.reranker import Chunk, KeywordReranker
from src.relevance.citation import CitationBuilder
from src.utils.storage import save_sessions_to_disk, load_sessions_from_disk
from src.ui.sidebar import render_sidebar
from src.ui.chat_area import render_chat_history
from src.document.loader import process_document
from src.retriever.faiss_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_retriever
)

# ✅ dùng pipeline mới của Thành viên 5
from src.llm.generator import generate_conversational_answer


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


# ====================== NẠP VECTOR STORE ======================
if curr_session.get("vector_store") is None and curr_session.get("file_name") is not None:
    loaded_vs = load_vector_store(curr_id)

    if loaded_vs:
        st.session_state.all_sessions[curr_id]["vector_store"] = loaded_vs
        curr_session = st.session_state.all_sessions[curr_id]


# ====================== 1. SIDEBAR ======================
config_params = render_sidebar()


# ====================== 2. QUẢN LÝ TÀI LIỆU ======================
st.title("📄 SmartDoc AI")

has_doc = curr_session.get("file_name") is not None

with st.expander("📁 Quản lý Tài liệu", expanded=not has_doc):

    uploaded_file = st.file_uploader(
        f"Tải lên tài liệu cho phiên: {curr_session['title']}",
        type=["pdf", "docx"]
    )

    if uploaded_file is not None:
        # Bắt buộc xử lý lại nếu upload file mới
        if curr_session.get("file_name") != uploaded_file.name:
            with st.spinner("Đang xử lý tài liệu..."):
                try:
                    suffix = os.path.splitext(uploaded_file.name)[1]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    documents, chunk_count = process_document(
                        tmp_path,
                        config_params["chunk_size"],
                        config_params["chunk_overlap"]
                    )

                    # Gắn metadata để source/citation hiển thị rõ file nào
                    for doc in documents:
                        if not hasattr(doc, "metadata") or doc.metadata is None:
                            doc.metadata = {}

                        doc.metadata["file_name"] = uploaded_file.name
                        doc.metadata["file_type"] = suffix.replace(".", "").lower()

                    vector_store = create_vector_store(documents)
                    save_vector_store(vector_store, curr_id)

                    st.session_state.all_sessions[curr_id]["vector_store"] = vector_store
                    st.session_state.all_sessions[curr_id]["file_name"] = uploaded_file.name

                    save_sessions_to_disk(st.session_state.all_sessions)

                    st.success(f"📦 Đã cắt tài liệu thành {chunk_count} chunks!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Lỗi xử lý tài liệu: {str(e)}")

                finally:
                    if "tmp_path" in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)

    if curr_session.get("file_name"):
        st.info(f"📄 Tài liệu đang liên kết: **{curr_session['file_name']}**")


st.divider()


# ====================== 3. KHU VỰC CHAT ======================
render_chat_history(curr_session["messages"])


if prompt_text := st.chat_input("Nhập câu hỏi..."):
    if curr_session.get("vector_store") is None:
        st.error("Vui lòng tải lên tài liệu trước!")

    else:
        # Lưu lịch sử trước câu hỏi hiện tại để Conversational RAG hiểu câu hỏi nối tiếp
        previous_messages = list(curr_session["messages"])

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
            with st.spinner("Đang tra cứu, suy luận và tự kiểm tra câu trả lời..."):
                try:
                    retriever = get_retriever(
                        curr_session["vector_store"],
                        config_params["search_type"],
                        config_params["k_value"]
                    )

                     # ====================== THÀNH VIÊN 4: RE-RANK + CITATION ======================
                    raw_docs = retriever.get_relevant_documents(prompt_text)

                    chunks = [
                        Chunk(
                            chunk_id=str(i),
                            text=doc.page_content,
                            source=doc.metadata.get("file_name", "Tài liệu đã upload"),
                            page=doc.metadata.get("page", i+1),
                            initial_score=1.0 - i * 0.05
                        )
                        for i, doc in enumerate(raw_docs)
                    ]

                    reranker = KeywordReranker()
                    reranked = reranker.rerank(prompt_text, chunks, top_k=5)

                    builder = CitationBuilder()
                    citations = builder.build_from_chunks(reranked.top_chunks)
                    # ==========================================================================

                    # ✅ Pipeline mới:
                    # - Memory
                    # - Query rewriting
                    # - Multi-hop retrieval
                    # - Self-RAG verification
                    # - Confidence scoring
                    result = generate_conversational_answer(
                        query=prompt_text,
                        retriever=retriever,
                        chat_messages=previous_messages,
                        llm_model=config_params["llm_model"]
                    )

                    # ====================== THÀNH VIÊN 4: GHI ĐÈ SOURCES ======================
                    if citations:
                        result["sources"] = [
                            {
                                "index": c.citation_id,
                                "source": c.source_file,
                                "page": c.page,
                                "content": c.excerpt,
                                "score": round(c.rerank_score, 4),
                            }
                            for c in citations
                        ]
                    # ==========================================================================

                    answer = result.get("answer", "Không tạo được câu trả lời.")
                    confidence_score = result.get("confidence_score", 70)
                    rewritten_query = result.get("rewritten_query", prompt_text)
                    sub_questions = result.get("sub_questions", [])
                    self_check = result.get("self_check", {})
                    sources = result.get("sources", [])

                    # ====================== HIỂN THỊ CÂU TRẢ LỜI ======================
                    st.markdown(answer)

                    st.divider()

                    # ====================== ADVANCED RAG INFO ======================
                    st.markdown("### 🧠 Advanced RAG Information")

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.metric("Confidence", f"{confidence_score}%")
                        st.progress(confidence_score / 100)

                    with col2:
                        st.markdown("**Câu hỏi đã được viết lại:**")
                        st.info(rewritten_query)

                    with st.expander("🔍 Multi-hop questions"):
                        if sub_questions:
                            for i, question in enumerate(sub_questions, start=1):
                                st.markdown(f"**Hop {i}:** {question}")
                        else:
                            st.caption("Không có câu hỏi con.")

                    with st.expander("✅ Self-RAG Verification"):
                        is_supported = self_check.get("is_supported", True)
                        reason = self_check.get("reason", "Không có")
                        missing_info = self_check.get("missing_info", "Không có")

                        st.markdown(f"**Được hỗ trợ bởi tài liệu:** `{is_supported}`")
                        st.markdown(f"**Lý do:** {reason}")
                        st.markdown(f"**Thông tin còn thiếu:** {missing_info}")

                    # ====================== NGUỒN THAM KHẢO ======================
                    with st.expander("📑 Xem nguồn tham khảo"):
                        if sources:
                            for src in sources:
                                index = src.get("index", "")
                                source_name = src.get("source", "Tài liệu đã upload")
                                page = src.get("page", "Không rõ")
                                content = src.get("content", "")

                                st.markdown(
                                    f"**Đoạn {index} - đọc từ tài liệu**  \n"
                                    f"File: `{source_name}`  \n"
                                    f"Trang: `{page}`"
                                )
                                st.caption(content)
                                st.divider()
                        else:
                            st.caption("Không có nguồn tham khảo.")

                    source_texts = [src.get("content", "") for src in sources]

                    # ====================== LƯU VÀO SESSION ======================
                    st.session_state.all_sessions[curr_id]["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_texts,
                        "advanced_rag": {
                            "rewritten_query": rewritten_query,
                            "sub_questions": sub_questions,
                            "confidence_score": confidence_score,
                            "self_check": self_check
                        }
                    })

                    save_sessions_to_disk(st.session_state.all_sessions)
                    st.rerun()

                except Exception as e:
                    st.error("❌ Lỗi AI.")
                    st.exception(e)