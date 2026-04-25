import streamlit as st
import uuid
import os
import shutil

from src.utils.storage import save_sessions_to_disk


def render_sidebar():
    """Render sidebar với quản lý phiên, cài đặt hệ thống và lọc metadata (Câu hỏi 8)."""
    with st.sidebar:
        st.title("🗂️ Lịch sử hội thoại")

        curr_id = st.session_state.current_session_id

        # ── 1. KHỐI CHỨC NĂNG HỆ THỐNG ────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Chat mới", use_container_width=True):
                new_id = str(uuid.uuid4())
                st.session_state.all_sessions[new_id] = {
                    "title"         : "Cuộc trò chuyện mới",
                    "messages"      : [],
                    "vector_store"  : None,
                    "file_name"     : None,
                    "uploaded_files": [],
                    "documents"     : [],
                }
                st.session_state.current_session_id = new_id
                save_sessions_to_disk(st.session_state.all_sessions)
                st.rerun()

        with col2:
            if st.button("🗑️ Xóa Chat", use_container_width=True, type="secondary"):
                if curr_id in st.session_state.all_sessions:
                    path = os.path.join("vector_store", curr_id)
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    del st.session_state.all_sessions[curr_id]

                if not st.session_state.all_sessions:
                    new_id = str(uuid.uuid4())
                    st.session_state.all_sessions = {
                        new_id: {
                            "title"         : "Cuộc trò chuyện mới",
                            "messages"      : [],
                            "vector_store"  : None,
                            "file_name"     : None,
                            "uploaded_files": [],
                            "documents"     : [],
                        }
                    }
                    st.session_state.current_session_id = new_id
                else:
                    st.session_state.current_session_id = list(
                        st.session_state.all_sessions.keys()
                    )[0]

                save_sessions_to_disk(st.session_state.all_sessions)
                st.rerun()

        if st.button(
            "🧹 Xóa File (Clear Vector)",
            use_container_width=True,
            help="Chỉ xóa tài liệu của phiên này, giữ lại tin nhắn",
        ):
            sess = st.session_state.all_sessions[curr_id]
            if sess.get("file_name") or sess.get("uploaded_files"):
                path = os.path.join("vector_store", curr_id)
                if os.path.exists(path):
                    shutil.rmtree(path)

                st.session_state.all_sessions[curr_id].update({
                    "vector_store"  : None,
                    "file_name"     : None,
                    "uploaded_files": [],
                    "documents"     : [],
                })
                save_sessions_to_disk(st.session_state.all_sessions)
                st.success("Đã xóa tài liệu của phiên này!")
                st.rerun()
            else:
                st.warning("Phiên này chưa có tài liệu để xóa.")

        st.divider()

        # ── 2. DANH SÁCH PHIÊN HỘI THOẠI ──────────────────────────
        st.write("**Gần đây**")
        for s_id, s_data in reversed(list(st.session_state.all_sessions.items())):
            is_active = s_id == st.session_state.current_session_id
            if st.button(
                f"💬 {s_data['title']}",
                key=f"sb_{s_id}",
                disabled=is_active,
                use_container_width=True,
            ):
                st.session_state.current_session_id = s_id
                st.rerun()

        st.divider()

        # ── 3. CÀI ĐẶT HỆ THỐNG ───────────────────────────────────
        with st.expander("⚙️ Cài đặt hệ thống"):
            st.info("Vector Store: **FAISS**")
            chunk_size    = st.slider("Chunk Size",    500,  2000, 1500, 100)
            chunk_overlap = st.slider("Chunk Overlap",  50,   300,  200,  50)

            search_type = st.selectbox(
                "Tìm kiếm",
                ["similarity", "mmr", "hybrid"],
                index=0,
                help=(
                    "• similarity: Pure vector search (FAISS)\n"
                    "• mmr: Maximal Marginal Relevance\n"
                    "• hybrid: Vector + BM25 keyword (Câu hỏi 7)"
                ),
            )

            # Chỉ hiện trọng số khi chọn hybrid
            vector_weight = 0.6
            bm25_weight   = 0.4
            if search_type == "hybrid":
                vector_weight = st.slider(
                    "Trọng số Semantic (FAISS)", 0.1, 0.9, 0.6, 0.1,
                    help="Phần còn lại sẽ là trọng số BM25"
                )
                bm25_weight = round(1.0 - vector_weight, 1)
                st.caption(f"→ BM25 weight: **{bm25_weight}**")

            k_value   = st.slider("Số đoạn (k)", 3, 8, 5)
            llm_model = st.selectbox(
                "Mô hình LLM",
                ["qwen2.5:7b", "llama3.2:1b", "qwen2.5:0.5b",
                 "qwen2.5:1.5b", "qwen2.5:3b", "llama2:7b"],
                index=0,
            )

        # ── 4. LỌC METADATA – Câu hỏi 8 ──────────────────────────
        curr_session   = st.session_state.all_sessions[curr_id]
        uploaded_files = curr_session.get("uploaded_files", [])

        filter_dict = {}

        if uploaded_files:
            with st.expander("🔍 Lọc theo Metadata", expanded=False):
                st.caption("Chỉ tìm kiếm trong tài liệu được chọn")

                # Lọc theo tên file
                file_names = ["Tất cả"] + [f["name"] for f in uploaded_files]
                selected_file = st.selectbox("📄 Tên file", file_names, key="filter_file")
                if selected_file != "Tất cả":
                    filter_dict["file_name"] = selected_file

                # Lọc theo loại tài liệu
                categories = list({f.get("doc_category", "") for f in uploaded_files if f.get("doc_category")})
                if categories:
                    cat_options = ["Tất cả"] + sorted(categories)
                    selected_cat = st.selectbox("🏷️ Loại tài liệu", cat_options, key="filter_cat")
                    if selected_cat != "Tất cả":
                        filter_dict["doc_category"] = selected_cat

                if filter_dict:
                    st.success(f"🔎 Đang lọc: {filter_dict}")
                else:
                    st.info("Đang tìm trong toàn bộ tài liệu")

        return {
            "chunk_size"    : chunk_size,
            "chunk_overlap" : chunk_overlap,
            "search_type"   : search_type,
            "k_value"       : k_value,
            "llm_model"     : llm_model,
            "vector_weight" : vector_weight,
            "bm25_weight"   : bm25_weight,
            "filter_dict"   : filter_dict if filter_dict else None,
        }
