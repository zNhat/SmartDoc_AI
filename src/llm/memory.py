from typing import List, Dict, Any


def clean_text(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""

    text = str(text).strip()
    text = " ".join(text.split())

    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text


def get_recent_chat_history(
    messages: List[Dict[str, Any]],
    max_turns: int = 5
) -> List[Dict[str, str]]:
    if not messages:
        return []

    valid_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role in ["user", "assistant"] and content:
            valid_messages.append({
                "role": role,
                "content": clean_text(content)
            })

    return valid_messages[-max_turns * 2:]


def build_history_context(
    messages: List[Dict[str, Any]],
    max_turns: int = 5
) -> str:
    """
    Chuyển lịch sử chat thành văn bản để LLM hiểu ngữ cảnh.
    """
    recent_messages = get_recent_chat_history(messages, max_turns=max_turns)

    if not recent_messages:
        return "Chưa có lịch sử hội thoại trước đó."

    lines = []
    for msg in recent_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            lines.append(f"Người dùng: {content}")
        elif role == "assistant":
            lines.append(f"Trợ lý: {content}")

    return "\n".join(lines)


def has_chat_history(messages: List[Dict[str, Any]]) -> bool:
    return bool(get_recent_chat_history(messages, max_turns=1))