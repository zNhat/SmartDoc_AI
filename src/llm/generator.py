import json
import re
from typing import List, Dict, Any, Optional

from langchain_community.llms import Ollama

from src.llm.memory import build_history_context, has_chat_history
from src.llm.prompts import (
    build_query_rewrite_prompt,
    build_multi_hop_prompt,
    build_answer_prompt,
    build_self_check_prompt,
)


# =========================
# 1. LLM CONFIG
# =========================

def get_llm(llm_model: str, temperature: float = 0.3) -> Ollama:
    return Ollama(
        model=llm_model,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1
    )


def invoke_llm(llm: Ollama, prompt: str) -> str:
    response = llm.invoke(prompt)

    if response is None:
        return ""

    return str(response).strip()


# =========================
# 2. DOCUMENT / SOURCE UTILS
# =========================

def get_doc_metadata(doc: Any) -> Dict[str, Any]:
    metadata = getattr(doc, "metadata", None)

    if not metadata:
        return {}

    return dict(metadata)


def get_doc_content(doc: Any) -> str:
    content = getattr(doc, "page_content", "")

    if not content:
        return ""

    return str(content).strip()


def format_page_number(page_value: Any) -> str:
    if page_value is None:
        return "Không rõ"

    try:
        page_int = int(page_value)
        return str(page_int + 1)
    except Exception:
        return str(page_value)


def build_source_label(doc: Any, index: int) -> str:
    metadata = get_doc_metadata(doc)

    source = (
        metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("source")
        or "Tài liệu đã upload"
    )

    page = (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("page_label")
    )

    page_text = format_page_number(page)

    return f"Nguồn {index} | File: {source} | Trang: {page_text}"


def format_docs_as_context(docs: List[Any], max_chars_per_doc: int = 2200) -> str:
    if not docs:
        return "Không có đoạn tài liệu liên quan được truy xuất."

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        content = get_doc_content(doc)

        if not content:
            continue

        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."

        source_label = build_source_label(doc, i)

        context_parts.append(
            f"[Đoạn {i}]\n"
            f"{source_label}\n"
            f"Nội dung:\n{content}"
        )

    if not context_parts:
        return "Không có đoạn tài liệu liên quan được truy xuất."

    return "\n\n".join(context_parts)


def doc_unique_key(doc: Any) -> str:
    metadata = get_doc_metadata(doc)
    content = get_doc_content(doc)

    source = str(metadata.get("source", ""))
    page = str(metadata.get("page", ""))
    content_head = content[:800]

    return f"{source}|{page}|{content_head}"


def deduplicate_docs(docs: List[Any], max_docs: int = 8) -> List[Any]:
    unique_docs = []
    seen = set()

    for doc in docs:
        key = doc_unique_key(doc)

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

        if len(unique_docs) >= max_docs:
            break

    return unique_docs


def extract_sources(docs: List[Any]) -> List[Dict[str, Any]]:
    sources = []

    for i, doc in enumerate(docs, start=1):
        metadata = get_doc_metadata(doc)
        content = get_doc_content(doc)

        source = (
            metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("source")
            or "Tài liệu đã upload"
        )

        page = (
            metadata.get("page")
            or metadata.get("page_number")
            or metadata.get("page_label")
        )

        sources.append({
            "index": i,
            "source": source,
            "page": format_page_number(page),
            "content": content,
            "metadata": metadata
        })

    return sources


# =========================
# 3. QUERY REWRITING
# =========================

def rewrite_query(
    user_question: str,
    chat_messages: Optional[List[Dict[str, Any]]],
    llm_model: str
) -> str:
    if not chat_messages or not has_chat_history(chat_messages):
        return user_question.strip()

    history_context = build_history_context(chat_messages, max_turns=5)

    llm = get_llm(llm_model, temperature=0.1)
    prompt = build_query_rewrite_prompt(user_question, history_context)

    rewritten = invoke_llm(llm, prompt)
    rewritten = rewritten.replace('"', "").replace("'", "").strip()

    if not rewritten:
        return user_question.strip()

    if len(rewritten) > 500:
        return user_question.strip()

    return rewritten


# =========================
# 4. MULTI-HOP REASONING
# =========================

def parse_sub_questions(raw_text: str, fallback_question: str) -> List[str]:
    if not raw_text:
        return [fallback_question]

    lines = raw_text.splitlines()
    questions = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()

        if len(line) < 5:
            continue

        questions.append(line)

    if not questions:
        return [fallback_question]

    return questions[:3]


def generate_sub_questions(
    user_question: str,
    rewritten_question: str,
    chat_messages: Optional[List[Dict[str, Any]]],
    llm_model: str
) -> List[str]:
    history_context = build_history_context(chat_messages or [], max_turns=5)

    llm = get_llm(llm_model, temperature=0.1)
    prompt = build_multi_hop_prompt(
        user_question=user_question,
        rewritten_question=rewritten_question,
        history_context=history_context
    )

    raw = invoke_llm(llm, prompt)
    sub_questions = parse_sub_questions(raw, rewritten_question)

    if rewritten_question not in sub_questions:
        sub_questions.insert(0, rewritten_question)

    return sub_questions[:3]


def retrieve_docs_for_questions(
    retriever: Any,
    questions: List[str],
    max_docs: int = 8
) -> List[Any]:
    all_docs = []

    for question in questions:
        try:
            docs = retriever.invoke(question)
            if docs:
                all_docs.extend(docs)
        except Exception:
            continue

    return deduplicate_docs(all_docs, max_docs=max_docs)


# =========================
# 5. SELF-RAG CHECKING
# =========================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def normalize_confidence(value: Any) -> int:
    try:
        score = int(float(value))
    except Exception:
        score = 70

    if score < 0:
        score = 0

    if score > 100:
        score = 100

    return score


def self_check_answer(
    user_question: str,
    rewritten_question: str,
    context: str,
    answer: str,
    llm_model: str
) -> Dict[str, Any]:
    llm = get_llm(llm_model, temperature=0.0)

    prompt = build_self_check_prompt(
        user_question=user_question,
        rewritten_question=rewritten_question,
        context=context,
        answer=answer
    )

    raw_check = invoke_llm(llm, prompt)
    parsed = extract_json_from_text(raw_check)

    if not parsed:
        return {
            "is_supported": True,
            "confidence_score": 70,
            "reason": "Không parse được JSON self-check, dùng confidence mặc định.",
            "missing_info": "Không rõ"
        }

    return {
        "is_supported": bool(parsed.get("is_supported", True)),
        "confidence_score": normalize_confidence(parsed.get("confidence_score", 70)),
        "reason": str(parsed.get("reason", "Không có lý do.")),
        "missing_info": str(parsed.get("missing_info", "Không có"))
    }


# =========================
# 6. FINAL ANSWER
# =========================

def ensure_document_prefix(answer: str) -> str:
    answer = answer.strip()

    required_prefix = "Nội dung dưới đây được đọc từ tài liệu:"

    if not answer:
        return required_prefix + "\nKhông tìm thấy thông tin này trong tài liệu."

    if not answer.lower().startswith(required_prefix.lower()):
        answer = required_prefix + "\n" + answer

    return answer


def generate_answer_from_docs(
    user_question: str,
    rewritten_question: str,
    docs: List[Any],
    chat_messages: Optional[List[Dict[str, Any]]],
    llm_model: str
) -> Dict[str, Any]:
    history_context = build_history_context(chat_messages or [], max_turns=5)
    context = format_docs_as_context(docs)

    prompt = build_answer_prompt(
        user_question=user_question,
        rewritten_question=rewritten_question,
        history_context=history_context,
        context=context
    )

    llm = get_llm(llm_model, temperature=0.3)
    answer = invoke_llm(llm, prompt)
    answer = ensure_document_prefix(answer)

    self_check = self_check_answer(
        user_question=user_question,
        rewritten_question=rewritten_question,
        context=context,
        answer=answer,
        llm_model=llm_model
    )

    return {
        "answer": answer,
        "context": context,
        "self_check": self_check,
        "confidence_score": self_check["confidence_score"],
        "sources": extract_sources(docs)
    }


def generate_conversational_answer(
    query: str,
    retriever: Any,
    chat_messages: Optional[List[Dict[str, Any]]],
    llm_model: str
) -> Dict[str, Any]:
    rewritten_query = rewrite_query(
        user_question=query,
        chat_messages=chat_messages,
        llm_model=llm_model
    )

    sub_questions = generate_sub_questions(
        user_question=query,
        rewritten_question=rewritten_query,
        chat_messages=chat_messages,
        llm_model=llm_model
    )

    retrieved_docs = retrieve_docs_for_questions(
        retriever=retriever,
        questions=sub_questions,
        max_docs=8
    )

    result = generate_answer_from_docs(
        user_question=query,
        rewritten_question=rewritten_query,
        docs=retrieved_docs,
        chat_messages=chat_messages,
        llm_model=llm_model
    )

    result["original_query"] = query
    result["rewritten_query"] = rewritten_query
    result["sub_questions"] = sub_questions

    return result


# =========================
# 7. BACKWARD COMPATIBILITY
# =========================

def generate_answer(query: str, relevant_docs: list, llm_model: str):
    result = generate_answer_from_docs(
        user_question=query,
        rewritten_question=query,
        docs=relevant_docs,
        chat_messages=[],
        llm_model=llm_model
    )

    return result["answer"]