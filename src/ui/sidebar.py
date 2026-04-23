import streamlit as st
import uuid
import os
import shutil
from src.utils.storage import save_sessions_to_disk

def render_sidebar():
    """Render giao diện sidebar với các tính năng quản lý phiên và tài liệu"""
    with st.sidebar:
        st.title("🗂️ Lịch sử hội thoại")
        
        curr_id = st.session_state.current_session_id
        
        # 1. KHỐI CHỨC NĂNG HỆ THỐNG
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Chat mới", use_container_width=True):
                new_id = str(uuid.uuid4())
                st.session_state.all_sessions[new_id] = {
                    "title": "Cuộc trò chuyện mới", "messages": [], "vector_store": None, "file_name": None
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
                        new_id: {"title": "Cuộc trò chuyện mới", "messages": [], "vector_store": None, "file_name": None}
                    }
                    st.session_state.current_session_id = new_id
                else:
                    st.session_state.current_session_id = list(st.session_state.all_sessions.keys())[0]
                
                save_sessions_to_disk(st.session_state.all_sessions)
                st.rerun()

        # NÚT XÓA FILE NẰM Ở ĐÂY
        if st.button("🧹 Xóa File (Clear Vector)", use_container_width=True, help="Chỉ xóa tài liệu của phiên này, giữ lại tin nhắn"):
            if st.session_state.all_sessions[curr_id]["file_name"]:
                # Xóa ổ cứng
                path = os.path.join("vector_store", curr_id)
                if os.path.exists(path):
                    shutil.rmtree(path)
                
                # Xóa RAM & Lưu lại
                st.session_state.all_sessions[curr_id]["vector_store"] = None
                st.session_state.all_sessions[curr_id]["file_name"] = None
                save_sessions_to_disk(st.session_state.all_sessions)
                
                st.success("Đã xóa tài liệu của phiên này!")
                st.rerun()
            else:
                st.warning("Phiên này chưa có tài liệu để xóa.")
            
        st.divider()
        
        # 2. DANH SÁCH CÁC PHIÊN HỘI THOẠI
        st.write("**Gần đây**")
        for s_id, s_data in reversed(list(st.session_state.all_sessions.items())):
            is_active = (s_id == st.session_state.current_session_id)
            if st.button(f"💬 {s_data['title']}", key=f"sb_{s_id}", disabled=is_active, use_container_width=True):
                st.session_state.current_session_id = s_id
                st.rerun()
            
        st.divider()
        
        # 3. CÀI ĐẶT THAM SỐ
        with st.expander("⚙️ Cài đặt hệ thống"):
            st.info("Vector Store: **FAISS**")
            chunk_size = st.slider("Chunk Size", 500, 2000, 1500, 100)
            chunk_overlap = st.slider("Chunk Overlap", 50, 300, 200, 50)
            search_type = st.selectbox("Tìm kiếm", ["similarity", "mmr"], index=0)
            k_value = st.slider("Số đoạn (k)", 3, 8, 5)
            llm_model = st.selectbox("Mô hình LLM", ["qwen2.5:7b", "llama2:7b"], index=0)
        
        return {
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
            "search_type": search_type, "k_value": k_value, "llm_model": llm_model
        }