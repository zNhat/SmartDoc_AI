# SmartDoc AI - Local RAG System

**Yêu cầu:** Máy tính đã cài đặt sẵn Python 3.8+ và [Ollama](https://ollama.com/).

## 🚀 Cài đặt & Khởi chạy nhanh

Thực hiện lần lượt các lệnh sau trong Terminal/Command Prompt:

```bash
# 1. Tải mã nguồn và di chuyển vào thư mục
git clone [repository-url]
cd SmartDoc_AI

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Tải mô hình AI 
ollama pull qwen2.5:7b

# 5. Khởi chạy ứng dụng web
streamlit run app.py