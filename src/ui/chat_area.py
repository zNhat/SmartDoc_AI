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

    st.markdown(
        f"### Đoạn {index} — 📄 `{source_name}` · Trang `{page}`"
    )

    info_parts = []

    if cat:
        info_parts.append(f"🏷️ Loại: `{cat}`")

    if up_date:
        info_parts.append(f"🕒 Upload: `{up_date}`")

    if reranker_name:
        info_parts.append(f"🧠 Reranker: `{reranker_name}`")

    if rerank_score is not None:
        try:
            info_parts.append(f"⭐ Rerank score: `{float(rerank_score):.4f}`")
        except Exception:
            info_parts.append(f"⭐ Rerank score: `{rerank_score}`")

    if rank_before and rank_after:
        info_parts.append(f"↕️ Rank trước: `{rank_before}` → sau re-rank: `{rank_after}`")

    if info_parts:
        st.caption(" · ".join(info_parts))

    st.markdown("**Context gốc được sử dụng:**")
    render_highlighted_context(source_content)

    st.divider()


def render_chat_history(messages_list):
    for message in messages_list:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

            if role != "assistant":
                continue

            if should_hide_sources(content):
                continue

            sources = normalize_sources(message)

            if not sources:
                continue

            with st.expander("📑 Xem nguồn tham khảo"):
                for src in sources:
                    render_source_block(src)