import streamlit as st
import uuid

def render_sidebar():
    """Render giao diện sidebar hỗ trợ đa phiên hội thoại và quản lý lịch sử"""
    with st.sidebar:
        st.title("🗂️ Lịch sử hội thoại")
        
        # Xác định ID của phiên hội thoại đang hoạt động
        curr_id = st.session_state.get("current_session_id")
        
        # 1. KHỐI CHỨC NĂNG CHÍNH
        col1, col2 = st.columns(2)
        with col1:
            # Tạo chat mới
            if st.button("➕ Chat mới", use_container_width=True, help="Tạo một phiên trò chuyện mới"):
                new_id = str(uuid.uuid4())
                if "all_sessions" not in st.session_state:
                    st.session_state.all_sessions = {}
                
                st.session_state.all_sessions[new_id] = {
                    "title": "Cuộc trò chuyện mới", 
                    "messages": [],
                    "vector_store": None,
                    "file_name": None
                }
                st.session_state.current_session_id = new_id
                st.rerun()
                
        with col2:
            # Xóa đoạn chat hiện tại
            if st.button("🗑️ Xóa Chat", use_container_width=True, help="Xóa hoàn toàn đoạn hội thoại này"):
                if curr_id in st.session_state.all_sessions:
                    del st.session_state.all_sessions[curr_id]
                
                if not st.session_state.all_sessions:
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
                else:
                    st.session_state.current_session_id = list(st.session_state.all_sessions.keys())[0]
                
                st.rerun()
            
        st.divider()
        
        # 2. DANH SÁCH CÁC PHIÊN HỘI THOẠI
        st.write("**Các cuộc trò chuyện gần đây**")
        
        if "all_sessions" in st.session_state:
            # Hiển thị danh sách theo thứ tự mới nhất lên đầu
            for s_id, s_data in reversed(list(st.session_state.all_sessions.items())):
                is_active = (s_id == st.session_state.current_session_id)
                
                if st.button(
                    f"💬 {s_data['title']}", 
                    key=f"sidebar_{s_id}", 
                    disabled=is_active, 
                    use_container_width=True
                ):
                    st.session_state.current_session_id = s_id
                    st.rerun()
        else:
            st.caption("Chưa có lịch sử trò chuyện.")
            
        st.divider()
        
        # 3. CÀI ĐẶT THAM SỐ HỆ THỐNG
        with st.expander("⚙️ Cài đặt hệ thống"):
            st.info("Vector Store: **FAISS**")
            chunk_size = st.slider("Chunk Size", 500, 2000, 1500, 100)
            chunk_overlap = st.slider("Chunk Overlap", 50, 300, 200, 50)
            search_type = st.selectbox("Phương thức tìm kiếm", ["similarity", "mmr"], index=0)
            k_value = st.slider("Số đoạn lấy về (k)", 3, 8, 5)
            llm_model = st.selectbox("Mô hình LLM", ["qwen2.5:7b", "llama2:7b", "mistral:7b"], index=0)

        st.caption("Bài tập lớn OSSD 2026 - SmartDoc AI")
        
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "search_type": search_type,
            "k_value": k_value,
            "llm_model": llm_model
        }