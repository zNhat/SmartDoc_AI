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
Bạn là SmartDoc AI, một hệ thống hỏi đáp tài liệu theo kiến trúc RAG.

QUY TẮC BẮT BUỘC:
QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên phần "Ngữ cảnh đọc từ tài liệu".
2. Nếu tài liệu không có thông tin, bắt buộc trả lời: "Không tìm thấy thông tin này trong tài liệu."
3. Nếu câu hỏi không liên quan trực tiếp đến ngữ cảnh đọc từ tài liệu, bắt buộc trả lời: "Không tìm thấy thông tin này trong tài liệu."
4. Không được bịa thêm thông tin ngoài tài liệu.
5. Không được tự dùng kiến thức bên ngoài.
6. Câu trả lời phải bắt đầu bằng câu: "Nội dung dưới đây được đọc từ tài liệu:"
7. Trả lời bằng tiếng Việt, rõ ràng, có thể dùng gạch đầu dòng nếu cần.
8. Nếu câu hỏi là câu hỏi nối tiếp, hãy dùng lịch sử hội thoại để hiểu ý người dùng.

Lịch sử hội thoại:
{history_context}

Câu hỏi gốc của người dùng:
{user_question}

Câu hỏi độc lập đã được viết lại:
{rewritten_question}

Ngữ cảnh đọc từ tài liệu:
{context}

Câu trả lời:
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
- Không đánh giá theo kiến thức bên ngoài.
- Chấm confidence_score từ 0 đến 100.
- Nếu câu trả lời không đủ căn cứ, hãy nêu lý do ngắn gọn.

Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.

Schema:
{{
  "is_supported": true,
  "confidence_score": 85,
  "reason": "Câu trả lời được hỗ trợ bởi các đoạn tài liệu liên quan.",
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