import logging
from typing import Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from utils.token_counter import count_message_tokens, count_text_tokens

logger = logging.getLogger("agent")

# 默认配置
DEFAULT_KEEP_TURNS = 5

# 压缩时预留的摘要 Token 空间
_RESERVED_FOR_SUMMARY = 300  # 200字摘要 + SystemMessage 包装开销


def _compute_split_idx_by_tokens(
    messages: List[BaseMessage],
    human_indices: List[int],
    max_tokens: int,
) -> int:
   
    budget = max_tokens - _RESERVED_FOR_SUMMARY
    if budget <= 0:
        budget = max_tokens  # 极端情况：预算甚至不够摘要，那就截断但不摘要

    # 从最后一轮开始向前累计
    for turn_num in range(len(human_indices) - 1, -1, -1):
        turn_start = human_indices[turn_num]
        turn_end = human_indices[turn_num + 1] if turn_num + 1 < len(human_indices) else len(messages)

        # 计算这一轮的 Token
        turn_messages = messages[turn_start:turn_end]
        turn_tokens = count_message_tokens(turn_messages)

        if turn_tokens > budget:
            # 这一轮放不下了
            next_turn_start = (
                human_indices[turn_num + 1]
                if turn_num + 1 < len(human_indices)
                else len(messages)
            )
            return next_turn_start

        budget -= turn_tokens

    return 0  # 全部都能放下


def compress(
    messages: List[BaseMessage],
    keep_turns: int = DEFAULT_KEEP_TURNS,
    summary_model: Optional[Any] = None,
    max_tokens: Optional[int] = None,
) -> List[BaseMessage]:

    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    total_turns = len(human_indices)

    if total_turns == 0:
        return messages

    if max_tokens is not None:
        # ── 动态 Token 感知模式 ──
        current_tokens = count_message_tokens(messages)
        if current_tokens <= max_tokens:
            return messages  # 没超预算，不需压缩

        split_idx = _compute_split_idx_by_tokens(messages, human_indices, max_tokens)
        if split_idx == 0:
            return messages  # 全部都能放下

        early_messages = messages[:split_idx]
        recent_messages = messages[split_idx:]

        kept_turns = len([m for m in recent_messages if isinstance(m, HumanMessage)])
        logger.info(
            "Context compressed (token-aware): %d turns (%d tokens) → [summary] + %d turns",
            total_turns, current_tokens, kept_turns,
        )
    else:
        if total_turns <= keep_turns:
            return messages

        split_idx = human_indices[-keep_turns]
        early_messages = messages[:split_idx]
        recent_messages = messages[split_idx:]

        logger.info(
            "Context compressed (fixed-turns): %d turns → [summary] + %d turns",
            total_turns, keep_turns,
        )

    # ── 执行摘要 ──
    if summary_model is None:
        logger.warning("summary_model is None, fallback to truncation")
        return recent_messages

    summary_text = _summarize(early_messages, summary_model)

    if summary_text:
        result = [
            SystemMessage(content=f"以下是之前对话的摘要，请参考其中的关键信息：\n{summary_text}")
        ] + recent_messages
        after_tokens = count_message_tokens(result)
        logger.info(
            "Compression done: before=%d tokens, after=%d tokens, saved=%d tokens",
            count_message_tokens(messages),
            after_tokens,
            count_message_tokens(messages) - after_tokens,
        )
    else:
        result = recent_messages
        logger.warning("Summarization failed, fallback to recent %d turns", len(recent_messages))

    return result


def _summarize(messages: List[BaseMessage], model: Any) -> str:
    """调用 LLM 生成对话摘要。"""
    try:
        dialogue_lines = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "用户"
            elif isinstance(m, AIMessage):
                role = "助手"
            else:
                continue
            content = _extract_text(m.content)
            if content:
                dialogue_lines.append(f"{role}：{content}")

        if not dialogue_lines:
            return ""

        prompt = (
            "请用一段简洁的中文（不超过 200 字）总结以下对话的核心内容，"
            "包括用户的关键问题和助手给出的关键回答，不要遗漏重要信息：\n\n"
        ) + "\n".join(dialogue_lines)

        response = model.invoke([HumanMessage(content=prompt)])
        summary = _extract_text(getattr(response, "content", ""))
        return summary.strip() if summary else ""

    except Exception as e:
        logger.error("Failed to generate conversation summary: %s", e)
        return ""


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
