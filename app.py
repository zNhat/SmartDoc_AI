import streamlit as st
import os
import tempfile
import uuid
import html
import re
from datetime import datetime

from src.relevance.reranker import Chunk, KeywordReranker, CrossEncoderReranker
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


# ====================== HELPER: SOURCE / PAGE ======================

def should_hide_sources(answer: str) -> bool:
    """
    Nếu câu trả lời là không tìm thấy thì không nên hiển thị nguồn,
    vì các chunk retrieve được không thật sự hỗ trợ câu trả lời.
    """
    answer_lower = str(answer).lower()

    not_found_markers = [
        "không tìm thấy thông tin này trong tài liệu",
        "tài liệu không đề cập",
        "không có trong tài liệu",
        "không tìm thấy",
    ]

    return any(marker in answer_lower for marker in not_found_markers)


def infer_page_from_content(content: str):
    """
    Fallback: nếu metadata không có page, thử lấy số trang ở cuối text PDF.
    Ví dụ cuối chunk có dòng '24' thì dùng 24.
    """
    if not content:
        return None

    lines = [line.strip() for line in str(content).splitlines() if line.strip()]

    for line in reversed(lines[-5:]):
        if re.fullmatch(r"\d{1,4}", line):
            return line

    return None


def get_page_display_from_source(src: dict) -> str:
    """
    Lấy page từ src hoặc metadata.
    Nếu không có metadata page thì fallback bằng số trang ở cuối context.
    """
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

        # Metadata page của PDFPlumber thường bắt đầu từ 0, nên cộng 1.
        # Nếu page lấy từ footer cuối context thì giữ nguyên.
        if page_from_metadata:
            return str(page_int + 1)

        return str(page_int)

    except Exception:
        return str(page)


def get_raw_page_for_chunk(metadata: dict):
    """
    Lấy page để đưa vào Chunk re-ranker.
    Không ép thành 'Không rõ' ở đây.
    """
    page = metadata.get("page", None)

    if page is None:
        page = metadata.get("page_number", None)

    if page is None:
        page = metadata.get("page_label", None)

    return page


# ====================== CÂU 9: RE-RANK RETRIEVER ADAPTER ======================

@st.cache_resource
def get_cached_cross_encoder_reranker():
    """
    Cache CrossEncoder để không load lại model nhiều lần.
    Nếu máy thiếu sentence-transformers hoặc lỗi thư viện, adapter sẽ fallback sang KeywordReranker.
    """
    return CrossEncoderReranker()


class RerankRetrieverAdapter:
    """
    Adapter bọc retriever gốc.

    Query
    -> base retriever lấy nhiều docs
    -> re-rank docs
    -> trả docs đã sắp xếp lại cho generate_conversational_answer()

    Nhờ vậy answer và sources dùng cùng một danh sách docs, không bị lệch nguồn.
    """

    def __init__(
        self,
        base_retriever,
        use_cross_encoder: bool = False,
        retrieve_top_n: int = 12,
        rerank_top_k: int = 5,
    ):
        self.base_retriever = base_retriever
        self.use_cross_encoder = use_cross_encoder
        self.retrieve_top_n = retrieve_top_n
        self.rerank_top_k = rerank_top_k

    def _base_retrieve(self, query: str):
        if hasattr(self.base_retriever, "invoke"):
            docs = self.base_retriever.invoke(query)
        else:
            docs = self.base_retriever.get_relevant_documents(query)

        return docs[: self.retrieve_top_n]

    def _docs_to_chunks(self, docs):
        chunks = []

        for i, doc in enumerate(docs):
            metadata = getattr(doc, "metadata", {}) or {}
            page_value = get_raw_page_for_chunk(metadata)

            chunks.append(
                Chunk(
                    chunk_id=str(i),
                    text=getattr(doc, "page_content", ""),
                    source=(
                        metadata.get("file_name")
                        or metadata.get("source")
                        or "Tài liệu đã upload"
                    ),
                    page=page_value if page_value is not None else 0,
                    initial_score=1.0 - i * 0.05,
                )
            )

        return chunks

    def _get_reranker(self):
        if self.use_cross_encoder:
            try:
                return get_cached_cross_encoder_reranker()
            except Exception as e:
                st.warning(
                    "⚠️ Không bật được CrossEncoderReranker, "
                    f"chuyển sang KeywordReranker. Lý do: {e}"
                )
                return KeywordReranker()

        return KeywordReranker()

    def _rerank_docs(self, query: str, docs):
        if not docs:
            return []

        chunks = self._docs_to_chunks(docs)
        reranker = self._get_reranker()
        reranker_name = reranker.__class__.__name__

        reranked_result = reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=self.rerank_top_k,
        )

        doc_map = {str(i): doc for i, doc in enumerate(docs)}
        reranked_docs = []

        for chunk in reranked_result.top_chunks:
            doc = doc_map.get(chunk.chunk_id)

            if doc is None:
                continue

            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}

            doc.metadata["rerank_score"] = chunk.rerank_score
            doc.metadata["rank_before"] = chunk.rank_before
            doc.metadata["rank_after"] = chunk.rank_after
            doc.metadata["reranker"] = reranker_name

            reranked_docs.append(doc)

        return reranked_docs

    def invoke(self, query: str):
        docs = self._base_retrieve(query)
        return self._rerank_docs(query, docs)

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


# ====================== CÂU 5: HIGHLIGHT CONTEXT ======================

def render_highlighted_context(content: str):
    """
    Highlight context gốc được dùng làm nguồn.
    Đáp ứng yêu cầu: highlight đoạn văn được sử dụng để trả lời.
    """
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
                "file_name": None,
                "uploaded_files": [],
                "documents": [],
            }
        }
        st.session_state.current_session_id = new_id

# Backward-compat: thêm các key mới vào session cũ nếu thiếu
for sid, sess in st.session_state.all_sessions.items():
    sess.setdefault("uploaded_files", [])
    sess.setdefault("documents", [])


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


# ====================== CÀI ĐẶT CÂU 9: RE-RANKING ======================

with st.sidebar.expander("🧠 Re-ranking - Câu 9"):
    use_cross_encoder = st.checkbox(
        "Bật Cross-Encoder Re-ranker",
        value=False,
        help=(
            "Bật để dùng CrossEncoderReranker thật. "
            "Lần đầu có thể chậm và cần sentence-transformers."
        ),
    )

    retrieve_top_n = st.slider(
        "Số chunk lấy trước re-rank",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
    )

    default_rerank_top_k = min(max(config_params.get("k_value", 5), 3), 10)

    rerank_top_k = st.slider(
        "Số chunk giữ lại sau re-rank",
        min_value=3,
        max_value=10,
        value=default_rerank_top_k,
        step=1,
    )

config_params["use_cross_encoder"] = use_cross_encoder
config_params["retrieve_top_n"] = retrieve_top_n
config_params["rerank_top_k"] = rerank_top_k


# ====================== 2. QUẢN LÝ TÀI LIỆU ======================

st.title("📄 SmartDoc AI")

has_doc = bool(curr_session.get("uploaded_files") or curr_session.get("file_name"))

with st.expander("📁 Quản lý Tài liệu", expanded=not has_doc):

    doc_category = st.selectbox(
        "🏷️ Loại tài liệu",
        ["Hợp đồng", "Báo cáo", "Hướng dẫn", "Quy định", "Nghiên cứu", "Khác"],
        key="doc_category_select",
        help="Phân loại sẽ được lưu vào metadata và dùng để lọc khi tìm kiếm",
    )

    uploaded_files = st.file_uploader(
        f"Tải lên tài liệu cho phiên: {curr_session['title']}",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Chọn một hoặc nhiều file PDF/DOCX cùng lúc",
    )

    if uploaded_files:
        existing_names = {f["name"] for f in curr_session.get("uploaded_files", [])}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        if new_files:
            with st.spinner(f"⏳ Đang xử lý {len(new_files)} tài liệu mới..."):
                all_new_docs = []
                processed_meta = []
                upload_date = datetime.now().strftime("%Y-%m-%d %H:%M")

                for uf in new_files:
                    tmp_path = None

                    try:
                        suffix = os.path.splitext(uf.name)[1]

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(uf.getvalue())
                            tmp_path = tmp_file.name

                        documents, chunk_count = process_document(
                            tmp_path,
                            config_params["chunk_size"],
                            config_params["chunk_overlap"],
                        )

                        for doc in documents:
                            if not hasattr(doc, "metadata") or doc.metadata is None:
                                doc.metadata = {}

                            doc.metadata["file_name"] = uf.name
                            doc.metadata["file_type"] = suffix.replace(".", "").lower()
                            doc.metadata["upload_date"] = upload_date
                            doc.metadata["doc_category"] = doc_category

                        all_new_docs.extend(documents)
                        processed_meta.append(
                            {
                                "name": uf.name,
                                "upload_date": upload_date,
                                "doc_category": doc_category,
                                "chunk_count": chunk_count,
                            }
                        )

                    except Exception as e:
                        st.error(f"❌ Lỗi xử lý '{uf.name}': {e}")

                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)

                if all_new_docs:
                    existing_vs = curr_session.get("vector_store")

                    if existing_vs is None:
                        vector_store = create_vector_store(all_new_docs)
                    else:
                        vector_store = update_vector_store(existing_vs, all_new_docs)

                    save_vector_store(vector_store, curr_id)

                    st.session_state.all_sessions[curr_id]["vector_store"] = vector_store
                    st.session_state.all_sessions[curr_id]["file_name"] = new_files[-1].name

                    uf_list = st.session_state.all_sessions[curr_id].get("uploaded_files", [])
                    uf_list.extend(processed_meta)
                    st.session_state.all_sessions[curr_id]["uploaded_files"] = uf_list

                    existing_docs = st.session_state.all_sessions[curr_id].get("documents", [])
                    existing_docs.extend(all_new_docs)
                    st.session_state.all_sessions[curr_id]["documents"] = existing_docs

                    save_sessions_to_disk(st.session_state.all_sessions)

                    names_str = ", ".join(m["name"] for m in processed_meta)
                    st.success(f"✅ Đã xử lý {len(processed_meta)} file: {names_str}")
                    st.rerun()

    uf_list = curr_session.get("uploaded_files", [])

    if uf_list:
        st.markdown("**📂 Tài liệu trong phiên này:**")

        for meta in uf_list:
            st.markdown(
                f"- 📄 **{meta['name']}**  \n"
                f"  🏷️ `{meta.get('doc_category', '?')}` · "
                f"🕒 `{meta.get('upload_date', '?')}` · "
                f"📦 `{meta.get('chunk_count', 0)} chunks`"
            )

    elif curr_session.get("file_name"):
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

        st.session_state.all_sessions[curr_id]["messages"].append(
            {
                "role": "user",
                "content": prompt_text,
            }
        )
        save_sessions_to_disk(st.session_state.all_sessions)

        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu, re-rank, suy luận và tự kiểm tra câu trả lời..."):
                try:
                    vector_store = curr_session["vector_store"]
                    documents = curr_session.get("documents", [])
                    search_type = config_params["search_type"]
                    k_value = config_params["k_value"]
                    filter_dict = config_params.get("filter_dict")

                    retrieve_top_n = config_params.get("retrieve_top_n", 12)
                    rerank_top_k = config_params.get("rerank_top_k", k_value)
                    use_cross_encoder = config_params.get("use_cross_encoder", False)

                    base_k = max(k_value, retrieve_top_n)
                    search_method = search_type

                    if search_type == "hybrid" and documents:
                        base_retriever = get_hybrid_retriever(
                            vector_store=vector_store,
                            documents=documents,
                            k=base_k,
                            filter_dict=filter_dict,
                            vector_weight=config_params.get("vector_weight", 0.6),
                            bm25_weight=config_params.get("bm25_weight", 0.4),
                        )

                    elif search_type == "hybrid" and not documents:
                        st.warning(
                            "⚠️ Hybrid search cần re-upload tài liệu để BM25 hoạt động. "
                            "Đang dùng pure vector search."
                        )
                        base_retriever = get_retriever(
                            vector_store,
                            "similarity",
                            base_k,
                            filter_dict,
                        )
                        search_method = "similarity (fallback)"

                    else:
                        base_retriever = get_retriever(
                            vector_store,
                            search_type,
                            base_k,
                            filter_dict,
                        )

                    retriever = RerankRetrieverAdapter(
                        base_retriever=base_retriever,
                        use_cross_encoder=use_cross_encoder,
                        retrieve_top_n=retrieve_top_n,
                        rerank_top_k=rerank_top_k,
                    )

                    reranker_name = (
                        "CrossEncoderReranker"
                        if use_cross_encoder
                        else "KeywordReranker"
                    )
                    search_method = f"{search_method} + {reranker_name}"

                    result = generate_conversational_answer(
                        query=prompt_text,
                        retriever=retriever,
                        chat_messages=previous_messages,
                        llm_model=config_params["llm_model"],
                    )

                    answer = result.get("answer", "Không tạo được câu trả lời.")
                    confidence_score = result.get("confidence_score", 70)
                    rewritten_query = result.get("rewritten_query", prompt_text)
                    sub_questions = result.get("sub_questions", [])
                    self_check = result.get("self_check", {})
                    sources = result.get("sources", [])

                    if should_hide_sources(answer):
                        sources = []
                    else:
                        for src in sources:
                            src["page"] = get_page_display_from_source(src)

                    st.markdown(answer)
                    st.divider()

                    st.markdown("### 🧠 Advanced RAG Information")

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.metric("Confidence", f"{confidence_score}%")
                        st.progress(confidence_score / 100)
                        st.caption(f"🔍 Phương pháp: `{search_method}`")
                        st.caption(f"📥 Retrieve top N: `{retrieve_top_n}`")
                        st.caption(f"🏆 Rerank top K: `{rerank_top_k}`")

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

                    if search_type == "hybrid" and documents:
                        with st.expander("📊 So sánh Hybrid vs Pure Vector Search"):
                            try:
                                cmp = get_retrieval_comparison(
                                    vector_store,
                                    documents,
                                    rewritten_query,
                                    k=k_value,
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

                    st.markdown("### 📑 Xem nguồn tham khảo")

                    if sources:
                        for src in sources:
                            index = src.get("index", "")
                            source_name = src.get("source", "Tài liệu đã upload")
                            page = get_page_display_from_source(src)
                            content = src.get("content", "")
                            meta = src.get("metadata", {}) or {}

                            cat = meta.get("doc_category", "")
                            up_date = meta.get("upload_date", "")
                            rerank_score = meta.get("rerank_score", src.get("score", None))
                            rank_before = meta.get("rank_before", None)
                            rank_after = meta.get("rank_after", None)
                            reranker_source = meta.get("reranker", "")

                            title = f"Đoạn {index} — 📄 {source_name} · Trang {page}"

                            with st.expander(title, expanded=False):
                                info_cols = st.columns(3)

                                with info_cols[0]:
                                    if cat:
                                        st.caption(f"🏷️ Loại: {cat}")

                                with info_cols[1]:
                                    if up_date:
                                        st.caption(f"🕒 Upload: {up_date}")

                                with info_cols[2]:
                                    if rerank_score is not None:
                                        try:
                                            st.caption(f"⭐ Rerank score: {float(rerank_score):.4f}")
                                        except Exception:
                                            st.caption(f"⭐ Rerank score: {rerank_score}")

                                if reranker_source:
                                    st.caption(f"🧠 Reranker: `{reranker_source}`")

                                if rank_before and rank_after:
                                    st.caption(
                                        f"↕️ Rank trước: `{rank_before}` → sau re-rank: `{rank_after}`"
                                    )

                                st.markdown("**Context gốc được sử dụng:**")
                                render_highlighted_context(content)

                    else:
                        st.caption("Không có nguồn tham khảo.")

                    source_texts = [src.get("content", "") for src in sources]

                    st.session_state.all_sessions[curr_id]["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_texts,
                            "source_details": sources,
                            "advanced_rag": {
                                "rewritten_query": rewritten_query,
                                "sub_questions": sub_questions,
                                "confidence_score": confidence_score,
                                "self_check": self_check,
                                "search_method": search_method,
                                "retrieve_top_n": retrieve_top_n,
                                "rerank_top_k": rerank_top_k,
                                "use_cross_encoder": use_cross_encoder,
                            },
                        }
                    )

                    save_sessions_to_disk(st.session_state.all_sessions)
                    st.rerun()

                except Exception as e:
                    st.error("❌ Lỗi AI.")
                    st.exception(e)