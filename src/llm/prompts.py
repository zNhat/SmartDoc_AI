def build_query_rewrite_prompt(user_question: str, history_context: str) -> str:
    return f"""
Bạn là hệ thống hỗ trợ Conversational RAG.

Nhiệm vụ:
- Dựa vào lịch sử hội thoại, hãy viết lại câu hỏi mới của người dùng thành một câu hỏi độc lập.
- Nếu câu hỏi hiện tại đã đầy đủ nghĩa, giữ nguyên.
- Không trả lời câu hỏi.
- Chỉ trả về đúng một câu hỏi đã được viết lại.

Lịch sử hội thoại:
{history_context}

Câu hỏi hiện tại:
{user_question}

Câu hỏi độc lập:
""".strip()


def build_multi_hop_prompt(user_question: str, rewritten_question: str, history_context: str) -> str:
    return f"""
Bạn là hệ thống phân tích câu hỏi cho Advanced RAG.

Nhiệm vụ:
- Tách câu hỏi thành tối đa 3 câu hỏi con cần truy xuất tài liệu.
- Nếu câu hỏi yêu cầu "liệt kê" hoặc hỏi về "các danh sách/tùy chọn", BẮT BUỘC phải sinh ra một câu hỏi con yêu cầu "tìm chi tiết của từng mục".
- Nếu câu hỏi đơn giản, chỉ trả về 1 câu hỏi.
- Không giải thích.
- Mỗi dòng là một câu hỏi con.
- Không đánh số thứ tự.

Lịch sử hội thoại:
{history_context}

Câu hỏi gốc:
{user_question}

Câu hỏi đã viết lại:
{rewritten_question}

Các câu hỏi con:
""".strip()


def build_answer_prompt(
    user_question: str,
    rewritten_question: str,
    history_context: str,
    context: str
) -> str:
    return f"""
Bạn là SmartDoc AI, một công cụ trích xuất dữ liệu tài liệu chính xác và chi tiết tuyệt đối.

QUY TẮC BẮT BUỘC (TUYỆT ĐỐI TUÂN THỦ):
1. TRÍCH XUẤT VÉT CẠN CHI TIẾT: ... (giữ nguyên của bạn)
2. KHÔNG TÓM TẮT: ... (giữ nguyên của bạn)
3. CHUẨN XÁC NGUỒN: ... (giữ nguyên)
4. NGOẠI LỆ: ... (giữ nguyên)
5. ĐỊNH DẠNG: Câu trả lời phải luôn bắt đầu bằng câu: "Nội dung dưới đây được đọc từ tài liệu:\n"
6. NGÔN NGỮ: Bắt buộc trả lời 100% bằng tiếng Việt.
7. XUỐNG DÒNG RÕ RÀNG: Giữa các đoạn văn, tiêu đề và các gạch đầu dòng BẮT BUỘC phải sử dụng dấu xuống dòng kép (khoảng trắng) để tách biệt. Tuyệt đối không viết dính chùm thành một khối chữ khổng lồ.
8. TRÌNH BÀY LIST: Mỗi mục con phải nằm trên một dòng riêng biệt, bắt đầu bằng dấu "- " hoặc "* ".

Lịch sử hội thoại:
{history_context}

Câu hỏi gốc của người dùng:
{user_question}

Câu hỏi độc lập đã được viết lại:
{rewritten_question}

Ngữ cảnh đọc từ tài liệu:
---
{context}
---

Câu trả lời nguyên văn và chi tiết đầy đủ mục con:
""".strip()


def build_self_check_prompt(
    user_question: str,
    rewritten_question: str,
    context: str,
    answer: str
) -> str:
    return f"""
Bạn là bộ kiểm tra Self-RAG cho hệ thống hỏi đáp tài liệu.

Nhiệm vụ:
- Kiểm tra câu trả lời có được hỗ trợ bởi tài liệu hay không.
- Kiểm tra xem câu trả lời có lùi sót các gạch đầu dòng chi tiết trong ngữ cảnh hay không.
- Không đánh giá theo kiến thức bên ngoài.
- Chấm confidence_score từ 0 đến 100.
- Nếu câu trả lời không đủ căn cứ hoặc thiếu sót mục con, hãy nêu lý do ngắn gọn.

Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.

Schema:
{{
  "is_supported": true,
  "confidence_score": 85,
  "reason": "Câu trả lời được hỗ trợ bởi các đoạn tài liệu liên quan và trích xuất đầy đủ mục con.",
  "missing_info": "Không có"
}}

Câu hỏi gốc:
{user_question}

Câu hỏi đã viết lại:
{rewritten_question}

Ngữ cảnh đọc từ tài liệu:
{context}

Câu trả lời cần kiểm tra:
{answer}

JSON:
""".strip()

def build_relevance_check_prompt(user_question: str, rewritten_question: str, context: str) -> str:
    return f"""
Bạn là bộ kiểm tra độ liên quan cho hệ thống RAG.

Nhiệm vụ:
- Chỉ dựa vào "Ngữ cảnh đọc từ tài liệu".
- Kiểm tra xem ngữ cảnh có chứa thông tin trực tiếp để trả lời câu hỏi hay không.
- Nếu câu hỏi hỏi kiến thức bên ngoài tài liệu, trả về can_answer=false.
- Nếu ngữ cảnh chỉ liên quan chung chung nhưng không đủ trả lời, trả về can_answer=false.
- Không dùng kiến thức bên ngoài.

Chỉ trả về JSON hợp lệ, không markdown.

Schema:
{{
  "can_answer": true,
  "confidence_score": 85,
  "reason": "Ngữ cảnh có thông tin trực tiếp để trả lời câu hỏi."
}}

Câu hỏi gốc:
{user_question}

Câu hỏi đã viết lại:
{rewritten_question}

Ngữ cảnh đọc từ tài liệu:
{context}

JSON:
""".strip()