import json
import os

HISTORY_FILE = "sessions.json"

def save_sessions_to_disk(all_sessions):
    """Lưu tiêu đề và tin nhắn của tất cả các phiên vào file JSON"""
    serializable_data = {}
    for s_id, s_data in all_sessions.items():
        serializable_data[s_id] = {
            "title": s_data["title"],
            "messages": s_data["messages"],
            "file_name": s_data.get("file_name")
        }
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=4)

def load_sessions_from_disk():
    """Nạp lại dữ liệu từ file JSON nếu tồn tại"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Khôi phục cấu trúc, vector_store sẽ được nạp lại sau từ faiss_store
            for s_id in data:
                data[s_id]["vector_store"] = None
            return data
    return None