# SmartDoc AI - Local RAG System

**Yêu cầu:** Máy tính đã cài đặt sẵn Python 3.12.9+ và [Ollama](https://ollama.com/).

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

1. Document & Processing Layer (src/document/)
loader.py: Trình nạp tài liệu. Sử dụng pdfplumber để trích xuất text và table từ PDF.

splitter.py: Thực hiện cắt nhỏ văn bản (Chunking).

Lưu ý: Cấu hình RecursiveCharacterTextSplitter. Chỉ hiển thị log thông báo cuối cùng khi đã hoàn tất toàn bộ quá trình cắt để tối ưu UI.

2. Data & Retriever Layer (src/retriever/)
faiss_store.py: Quản lý Vector Database. Chịu trách nhiệm lưu/nạp (Save/Load local) chỉ mục FAISS.

hybrid.py: (Yêu cầu nâng cao - Q7) Triển khai kết hợp Vector Search và BM25 Keyword Search.

3. Model & Logic Layer (src/llm/)
generator.py: Tích hợp LangChain và Ollama để gọi model qwen2.5:7b.

prompts.py: Thiết kế System Prompt. Hỗ trợ phát hiện ngôn ngữ tự động (Vi/En).

memory.py: Quản lý ngữ cảnh hội thoại (Chat History) cho từng phiên.

4. Relevance & Optimization Layer (src/relevance/)
citation.py: (Yêu cầu nâng cao - Q5) Trích xuất số trang và vị trí nguồn.

Quy định: Hiển thị nguồn với nhãn "đọc từ tài liệu".

reranker.py: (Yêu cầu nâng cao - Q9) Triển khai Cross-Encoder để xếp hạng lại kết quả tìm kiếm.

5. Presentation Layer (src/ui/)
sidebar.py: Quản lý Multi-session (Tạo/Xóa chat), Sidebar config và Project Status.

chat_area.py: Render giao diện chat, xử lý hiển thị bubble chat và expander nguồn dẫn.