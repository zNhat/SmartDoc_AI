import streamlit as st

def render_chat_history(messages_list):
    """Hiển thị toàn bộ tin nhắn của phiên hiện tại"""
    for message in messages_list:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📑 Xem nguồn tham khảo"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**Đoạn {i} (chỗ này ghi lại nói là đọc từ tài liệu):**")
                        st.caption(doc)