from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def generate_answer(query: str, relevant_docs: list, llm_model: str):
    """Ghép ngữ cảnh và gọi Ollama sinh câu trả lời"""
    
    # Nối các chunk tìm được thành 1 đoạn văn bản ngữ cảnh
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Template chuẩn cho tiếng Việt
    prompt_template = """Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.
Hãy tổng hợp thông tin, dùng gạch đầu dòng nếu cần để trình bày rõ ràng.
Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là không tìm thấy.

Ngữ cảnh:
{context}

Câu hỏi: {user_input}

Trả lời:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_input"])
    
    # Khởi tạo Ollama với các tham số ổn định
    llm = Ollama(
        model=llm_model,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.1
    )
    
    final_prompt = prompt.format(context=context, user_input=query)
    
    # Sinh câu trả lời
    answer = llm.invoke(final_prompt)
    return answer