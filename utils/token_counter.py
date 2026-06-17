from typing import List, Optional, Any
import tiktoken


_ENCODING: Optional[tiktoken.Encoding] = None

# 每条消息的角色/格式开销（经验值，用于 Qwen 类模型）
_MESSAGE_OVERHEAD = 6


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def _extract_text(content: Any) -> str:
    """从消息 content 中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content or "")


def count_text_tokens(text: str) -> int:
    """计算纯文本的 Token 数。"""
    encoding = _get_encoding()
    return len(encoding.encode(text))


def count_message_tokens(messages: List[Any]) -> int:

    encoding = _get_encoding()
    total = 0
    for msg in messages:
        text = _extract_text(getattr(msg, "content", ""))
        total += len(encoding.encode(text)) + _MESSAGE_OVERHEAD
    return total
