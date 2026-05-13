import html
import re
import streamlit as st


def render_highlighted_context(content: str):
    safe_content = html.escape(content or "")

    st.markdown(
        f"""
        <div style="
            background-color: rgba(255, 193, 7, 0.12);
            border-left: 4px solid #FFC107;
            padding: 12px 14px;
            border-radius: 8px;
            line-height: 1.55;
            white-space: pre-wrap;
            margin-top: 8px;
            margin-bottom: 12px;
        ">
            {safe_content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def should_hide_sources(answer: str) -> bool:
    answer_lower = str(answer).lower()

    not_found_markers = [
        "không tìm thấy thông tin này trong tài liệu",
        "tài liệu không đề cập",
        "không có trong tài liệu",
        "không tìm thấy",
    ]

    return any(marker in answer_lower for marker in not_found_markers)


def infer_page_from_content(content: str):
    if not content:
        return None

    lines = [line.strip() for line in str(content).splitlines() if line.strip()]

    for line in reversed(lines[-5:]):
        if re.fullmatch(r"\d{1,4}", line):
            return line

    return None


def get_page_display(src: dict) -> str:
    meta = src.get("metadata", {}) or {}

    page = src.get("page", None)
    page_from_metadata = False

    if page is None or page == "" or page == "Không rõ":
        page = meta.get("page", None)
        page_from_metadata = page is not None

    if page is None:
        page = meta.get("page_number", None)
        page_from_metadata = page is not None

    if page is None:
        page = meta.get("page_label", None)
        page_from_metadata = page is not None

    if page is None or page == "" or page == "Không rõ":
        page = infer_page_from_content(src.get("content", ""))
        page_from_metadata = False

    if page is None or page == "":
        return "Không rõ"

    try:
        page_int = int(page)

        if page_from_metadata:
            return str(page_int + 1)

        return str(page_int)

    except Exception:
        return str(page)


def normalize_sources(message):
    source_details = message.get("source_details", [])

    if source_details:
        return source_details

    old_sources = message.get("sources", [])

    return [
        {
            "index": i + 1,
            "source": "Tài liệu đã upload",
            "page": "Không rõ",
            "content": src,
            "metadata": {},
        }
        for i, src in enumerate(old_sources)
    ]

def render_source_block(src):
    """
    Hàm này KHÔNG dùng expander nữa, chỉ in nội dung thẳng ra
    (vì đã được bọc bởi expander lớn ở bên ngoài)
    """
    index = src.get("index", "")
    source_name = src.get("source", "Tài liệu đã upload")
    page = get_page_display(src)
    source_content = src.get("content", "")
    meta = src.get("metadata", {}) or {}

    cat = meta.get("doc_category", "")
    up_date = meta.get("upload_date", "")
    rerank_score = meta.get("rerank_score", src.get("score", None))
    rank_before = meta.get("rank_before", None)
    rank_after = meta.get("rank_after", None)
    reranker_name = meta.get("reranker", "")

    # In tiêu đề đoạn (in đậm)
    st.markdown(f"**Đoạn {index} — 📄 {source_name} · Trang {page}**")

    # In các thông số phụ
    info_parts = []
    if cat:
        info_parts.append(f"🏷️ `{cat}`")
    if up_date:
        info_parts.append(f"🕒 `{up_date}`")
    if reranker_name:
        info_parts.append(f"🧠 `{reranker_name}`")
    if rerank_score is not None:
        try:
            info_parts.append(f"⭐ Score: `{float(rerank_score):.4f}`")
        except Exception:
            info_parts.append(f"⭐ Score: `{rerank_score}`")
    if rank_before and rank_after:
        info_parts.append(f"↕️ Rank: `{rank_before}` → `{rank_after}`")

    if info_parts:
        st.caption(" · ".join(info_parts))

    render_highlighted_context(source_content)


def render_chat_history(messages_list):
    for message in messages_list:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            # 1. In nội dung câu trả lời
            st.markdown(content)

            if role != "assistant":
                continue

            # ============================
            # KHỐI ĐÁNH GIÁ (ADVANCED RAG)
            # =============================
            advanced_rag = message.get("advanced_rag")
            if advanced_rag:
                st.divider()
                st.markdown("###  Advanced RAG Information")

                col1, col2 = st.columns([1, 3])

                with col1:
                    score = advanced_rag.get("confidence_score", 0)
                    st.metric("Confidence", f"{score}%")
                    st.progress(score / 100)
                    st.caption(f" Phương pháp: `{advanced_rag.get('search_method', '')}`")

                with col2:
                    st.markdown("**Câu hỏi đã được viết lại:**")
                    st.info(advanced_rag.get("rewritten_query", ""))

                sub_questions = advanced_rag.get("sub_questions", [])
                with st.expander(" Multi-hop questions"):
                    if sub_questions:
                        for i, question in enumerate(sub_questions, start=1):
                            st.markdown(f"**Hop {i}:** {question}")
                    else:
                        st.caption("Không có câu hỏi con.")

                self_check = advanced_rag.get("self_check", {})
                if self_check:
                    with st.expander(" Self-RAG Verification"):
                        st.markdown(f"**Được hỗ trợ bởi tài liệu:** `{self_check.get('is_supported', True)}`")
                        st.markdown(f"**Lý do:** {self_check.get('reason', 'Không có')}")
                        st.markdown(f"**Thông tin còn thiếu:** {self_check.get('missing_info', 'Không có')}")

            # ======================
            # 3.  NGUỒN THAM KHẢO 
            # ======================
            if should_hide_sources(content):
                continue

            sources = normalize_sources(message)

            if not sources:
                continue
            
            with st.expander(" Xem toàn bộ nguồn tham khảo"):
                for src in sources:
                    render_source_block(src)
                    
                    # Thêm đường kẻ mờ phân tách giữa các đoạn 
                    if src != sources[-1]:
                        st.divider()