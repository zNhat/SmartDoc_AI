import json
import os

HISTORY_FILE = "sessions.json"


def save_sessions_to_disk(all_sessions):
    """
    Lưu tiêu đề, tin nhắn, metadata file của tất cả phiên vào JSON.
    Các trường không serialize được (vector_store, documents) bị bỏ qua.
    """
    serializable_data = {}
    for s_id, s_data in all_sessions.items():
        serializable_data[s_id] = {
            "title"         : s_data.get("title", "Cuộc trò chuyện mới"),
            "messages"      : s_data.get("messages", []),
            "file_name"     : s_data.get("file_name"),
            # lưu danh sách metadata của các file đã upload
            "uploaded_files": s_data.get("uploaded_files", []),
        }

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=4)


def load_sessions_from_disk():
    """Nạp lại dữ liệu từ file JSON nếu tồn tại."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Khôi phục cấu trúc; vector_store và documents sẽ được nạp lại sau
        for s_id in data:
            data[s_id]["vector_store"] = None
            data[s_id]["documents"]    = []   # runtime-only, cần upload lại để dùng hybrid
            # backward-compat: nếu file cũ chưa có uploaded_files
            if "uploaded_files" not in data[s_id]:
                data[s_id]["uploaded_files"] = []

        return data
    return None