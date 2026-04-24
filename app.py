import streamlit as st
import os
import tempfile
import uuid
from datetime import datetime

from src.utils.storage import save_sessions_to_disk, load_sessions_from_disk
from src.ui.sidebar import render_sidebar
from src.ui.chat_area import render_chat_history
from src.document.loader import process_document
from src.retriever.faiss_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_retriever,
    update_vector_store,
)
from src.retriever.hybrid import get_hybrid_retriever, get_retrieval_comparison

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
                "title"         : "Cuộc trò chuyện mới",
                "messages"      : [],
                "vector_store"  : None,
                "file_name"     : None,
                "uploaded_files": [],   # Câu hỏi 8: danh sách metadata các file
                "documents"     : [],   # runtime-only cho BM25 (Câu hỏi 7)
            }
        }
        st.session_state.current_session_id = new_id

# Backward-compat: thêm các key mới vào session cũ nếu thiếu
for sid, sess in st.session_state.all_sessions.items():
    sess.setdefault("uploaded_files", [])
    sess.setdefault("documents", [])


curr_id      = st.session_state.current_session_id
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

has_doc = bool(curr_session.get("uploaded_files") or curr_session.get("file_name"))

with st.expander("📁 Quản lý Tài liệu", expanded=not has_doc):

    # ── Câu hỏi 8: loại tài liệu (user tự phân loại) ──────────────
    doc_category = st.selectbox(
        "🏷️ Loại tài liệu",
        ["Hợp đồng", "Báo cáo", "Hướng dẫn", "Quy định", "Nghiên cứu", "Khác"],
        key="doc_category_select",
        help="Phân loại sẽ được lưu vào metadata và dùng để lọc khi tìm kiếm"
    )

    # ── Câu hỏi 8: upload NHIỀU file cùng lúc ────────────────────
    uploaded_files = st.file_uploader(
        f"Tải lên tài liệu cho phiên: {curr_session['title']}",
        type=["pdf", "docx"],
        accept_multiple_files=True,          # ← cho phép chọn nhiều file
        help="Chọn một hoặc nhiều file PDF/DOCX cùng lúc"
    )

    if uploaded_files:
        # Chỉ xử lý file mới (chưa có trong session)
        existing_names = {f["name"] for f in curr_session.get("uploaded_files", [])}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        if new_files:
            with st.spinner(f"⏳ Đang xử lý {len(new_files)} tài liệu mới..."):
                all_new_docs = []
                processed_meta = []
                upload_date = datetime.now().strftime("%Y-%m-%d %H:%M")

                for uf in new_files:
                    try:
                        suffix   = os.path.splitext(uf.name)[1]
                        tmp_path = None

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(uf.getvalue())
                            tmp_path = tmp_file.name

                        documents, chunk_count = process_document(
                            tmp_path,
                            config_params["chunk_size"],
                            config_params["chunk_overlap"],
                        )

                        # ── Gắn metadata đầy đủ (Câu hỏi 8) ─────────────
                        for doc in documents:
                            if not hasattr(doc, "metadata") or doc.metadata is None:
                                doc.metadata = {}
                            doc.metadata["file_name"]   = uf.name
                            doc.metadata["file_type"]   = suffix.replace(".", "").lower()
                            doc.metadata["upload_date"] = upload_date
                            doc.metadata["doc_category"]= doc_category

                        all_new_docs.extend(documents)
                        processed_meta.append({
                            "name"        : uf.name,
                            "upload_date" : upload_date,
                            "doc_category": doc_category,
                            "chunk_count" : chunk_count,
                        })

                    except Exception as e:
                        st.error(f"❌ Lỗi xử lý '{uf.name}': {e}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)

                if all_new_docs:
                    # ── Merge vào vector store hiện có hoặc tạo mới ──────
                    existing_vs = curr_session.get("vector_store")
                    if existing_vs is None:
                        vector_store = create_vector_store(all_new_docs)
                    else:
                        vector_store = update_vector_store(existing_vs, all_new_docs)

                    save_vector_store(vector_store, curr_id)

                    # Cập nhật session state
                    st.session_state.all_sessions[curr_id]["vector_store"] = vector_store
                    st.session_state.all_sessions[curr_id]["file_name"]    = new_files[-1].name

                    # Câu hỏi 8: cập nhật danh sách file metadata
                    uf_list = st.session_state.all_sessions[curr_id].get("uploaded_files", [])
                    uf_list.extend(processed_meta)
                    st.session_state.all_sessions[curr_id]["uploaded_files"] = uf_list

                    # Câu hỏi 7: lưu documents vào RAM cho BM25
                    existing_docs = st.session_state.all_sessions[curr_id].get("documents", [])
                    existing_docs.extend(all_new_docs)
                    st.session_state.all_sessions[curr_id]["documents"] = existing_docs

                    save_sessions_to_disk(st.session_state.all_sessions)

                    names_str = ", ".join(m["name"] for m in processed_meta)
                    st.success(f"✅ Đã xử lý {len(processed_meta)} file: {names_str}")
                    st.rerun()

    # ── Hiển thị danh sách file đã upload (Câu hỏi 8) ────────────
    uf_list = curr_session.get("uploaded_files", [])
    if uf_list:
        st.markdown("**📂 Tài liệu trong phiên này:**")
        for meta in uf_list:
            st.markdown(
                f"- 📄 **{meta['name']}**  \n"
                f"  🏷️ `{meta.get('doc_category','?')}` · "
                f"🕒 `{meta.get('upload_date','?')}` · "
                f"📦 `{meta.get('chunk_count',0)} chunks`"
            )
    elif curr_session.get("file_name"):
        # Backward compat: session cũ chưa có uploaded_files
        st.info(f"📄 Tài liệu đang liên kết: **{curr_session['file_name']}**")


st.divider()


# ====================== 3. KHU VỰC CHAT ======================
render_chat_history(curr_session["messages"])


if prompt_text := st.chat_input("Nhập câu hỏi..."):
    if curr_session.get("vector_store") is None:
        st.error("Vui lòng tải lên tài liệu trước!")

    else:
        previous_messages = list(curr_session["messages"])

        if len(curr_session["messages"]) == 0:
            st.session_state.all_sessions[curr_id]["title"] = prompt_text[:25] + "..."

        st.session_state.all_sessions[curr_id]["messages"].append({
            "role"   : "user",
            "content": prompt_text
        })
        save_sessions_to_disk(st.session_state.all_sessions)

        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu, suy luận và tự kiểm tra câu trả lời..."):
                try:
                    vector_store = curr_session["vector_store"]
                    documents    = curr_session.get("documents", [])
                    search_type  = config_params["search_type"]
                    k_value      = config_params["k_value"]
                    filter_dict  = config_params.get("filter_dict")

                    # ── Câu hỏi 7: chọn retriever theo loại tìm kiếm ──────
                    search_method = search_type  # để hiển thị sau

                    if search_type == "hybrid" and documents:
                        retriever = get_hybrid_retriever(
                            vector_store  = vector_store,
                            documents     = documents,
                            k             = k_value,
                            filter_dict   = filter_dict,
                            vector_weight = config_params.get("vector_weight", 0.6),
                            bm25_weight   = config_params.get("bm25_weight", 0.4),
                        )
                    elif search_type == "hybrid" and not documents:
                        # Fallback: documents chưa có trong RAM (session cũ reload)
                        st.warning(
                            "⚠️ Hybrid search cần re-upload tài liệu để BM25 hoạt động. "
                            "Đang dùng pure vector search."
                        )
                        retriever = get_retriever(vector_store, "similarity", k_value, filter_dict)
                        search_method = "similarity (fallback)"
                    else:
                        retriever = get_retriever(vector_store, search_type, k_value, filter_dict)

                    # ── Pipeline chính ─────────────────────────────────────
                    result = generate_conversational_answer(
                        query         = prompt_text,
                        retriever     = retriever,
                        chat_messages = previous_messages,
                        llm_model     = config_params["llm_model"]
                    )

                    answer           = result.get("answer", "Không tạo được câu trả lời.")
                    confidence_score = result.get("confidence_score", 70)
                    rewritten_query  = result.get("rewritten_query", prompt_text)
                    sub_questions    = result.get("sub_questions", [])
                    self_check       = result.get("self_check", {})
                    sources          = result.get("sources", [])

                    # ── Câu trả lời ────────────────────────────────────────
                    st.markdown(answer)
                    st.divider()

                    # ── Advanced RAG Info ──────────────────────────────────
                    st.markdown("### 🧠 Advanced RAG Information")

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Confidence", f"{confidence_score}%")
                        st.progress(confidence_score / 100)
                        st.caption(f"🔍 Phương pháp: `{search_method}`")

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
                        reason       = self_check.get("reason", "Không có")
                        missing_info = self_check.get("missing_info", "Không có")
                        st.markdown(f"**Được hỗ trợ bởi tài liệu:** `{is_supported}`")
                        st.markdown(f"**Lý do:** {reason}")
                        st.markdown(f"**Thông tin còn thiếu:** {missing_info}")

                    # ── Câu hỏi 7: So sánh hiệu suất retrieval ───────────
                    if search_type == "hybrid" and documents:
                        with st.expander("📊 So sánh Hybrid vs Pure Vector Search"):
                            try:
                                cmp = get_retrieval_comparison(
                                    vector_store, documents, rewritten_query, k=k_value
                                )
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric("🔷 Pure Vector", f"{cmp['vector']['count']} docs")
                                with c2:
                                    st.metric("🔑 BM25 Keyword", f"{cmp['bm25']['count']} docs")
                                with c3:
                                    st.metric("⚡ Hybrid", f"{cmp['hybrid']['count']} docs")

                                overlap = cmp["overlap"]
                                st.markdown(
                                    f"- Chung (Vector ∩ BM25): **{overlap['vector_and_bm25_common']}** đoạn  \n"
                                    f"- Chỉ trong Vector: **{overlap['vector_only']}** đoạn  \n"
                                    f"- Chỉ trong BM25: **{overlap['bm25_only']}** đoạn  \n"
                                    f"- Hybrid bổ sung so với Vector thuần: **{overlap['hybrid_extra_vs_vector']}** đoạn"
                                )
                            except Exception as e_cmp:
                                st.caption(f"Không lấy được so sánh: {e_cmp}")

                    # ── Câu hỏi 8: Nguồn từ document nào ─────────────────
                    with st.expander("📑 Xem nguồn tham khảo"):
                        if sources:
                            for src in sources:
                                index       = src.get("index", "")
                                source_name = src.get("source", "Tài liệu đã upload")
                                page        = src.get("page", "Không rõ")
                                content     = src.get("content", "")
                                meta        = src.get("metadata", {})
                                cat         = meta.get("doc_category", "")
                                up_date     = meta.get("upload_date", "")

                                badge = f"🏷️ `{cat}`" if cat else ""
                                date_badge = f"🕒 `{up_date}`" if up_date else ""

                                st.markdown(
                                    f"**Đoạn {index}** — 📄 `{source_name}` · Trang `{page}`  \n"
                                    f"{badge}  {date_badge}"
                                )
                                st.caption(content)
                                st.divider()
                        else:
                            st.caption("Không có nguồn tham khảo.")

                    source_texts = [src.get("content", "") for src in sources]

                    # ── Lưu vào session ─────────────────────────────────
                    st.session_state.all_sessions[curr_id]["messages"].append({
                        "role"   : "assistant",
                        "content": answer,
                        "sources": source_texts,
                        "advanced_rag": {
                            "rewritten_query": rewritten_query,
                            "sub_questions"  : sub_questions,
                            "confidence_score": confidence_score,
                            "self_check"     : self_check,
                            "search_method"  : search_method,
                        }
                    })

                    save_sessions_to_disk(st.session_state.all_sessions)
                    st.rerun()

                except Exception as e:
                    st.error("❌ Lỗi AI.")
                    st.exception(e)