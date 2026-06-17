from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import rag_summarize, get_current_time, calculator, web_search
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from memory.context_compressor import compress as compress_context
from utils.token_counter import count_message_tokens
from typing import Any, Dict, Iterator, Optional


MAX_ITERATIONS_REACHED = "已超过最大工具调用次数，请简化您的问题或提供更明确的信息。"

class ReActAgent(Runnable):
    def __init__(
        self,
        memory_store: Any = None,
        context_max_tokens: int = 16000,
        max_iterations: int = 5,
    ):
        self.memory_store = memory_store
        self.context_max_tokens = context_max_tokens
        self.max_iterations = max_iterations
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_current_time, calculator, web_search],
            middleware=[report_prompt_switch, log_before_model, monitor_tool]
        )

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Any:
        return self.agent.invoke(input, config=config, context={"report": False})


    def stream(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Iterator[Any]:
        tool_call_count = 0
        for chunk in self.agent.stream(input, stream_mode="values", config=config, context={"report": False}):
            if chunk.get("messages") and len(chunk["messages"]) > 0:
                latest_message = chunk["messages"][-1]
                if self._is_ai_message(latest_message):
                    if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                        tool_call_count += 1
                        if tool_call_count > self.max_iterations:
                            break
                        continue
                    text = self._message_to_text(latest_message)
                    if text:
                        yield latest_message

    def execute_stream(self, query: str, session_id: Optional[str] = None) -> Iterator[str]:
        for event in self.stream_events(query, session_id=session_id):
            if event["type"] == "text":
                yield event["text"]

    def stream_events(
        self,
        query: str,
        session_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Iterator[Dict[str, Any]]:
        input_dict = self._build_input(query, session_id)
        tool_call_count = 0
        final_message = None

        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            config=config,
            context={"report": False},
        ):
            latest_message = self._get_latest_ai_message(chunk)
            if latest_message is None:
                continue

            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                tool_call_count += 1
                if tool_call_count > self.max_iterations:
                    break
                final_message = latest_message
                continue

            final_message = latest_message

            text = self._message_to_text(latest_message)
            if text:
                yield {"type": "text", "text": text}

        if tool_call_count > self.max_iterations:
            yield {"type": "text", "text": MAX_ITERATIONS_REACHED}

        metadata = self._extract_message_metadata(final_message)
        if session_id and final_message is not None:
            self._persist_turn(session_id, query, final_message, metadata)

        yield {"type": "metadata", "metadata": metadata}

    def _build_input(self, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        messages = []
        if session_id and self.memory_store is not None:
            history = self.memory_store.get_session_history(session_id).messages
            context_tokens = count_message_tokens(history)
            if context_tokens > self.context_max_tokens:
                history = compress_context(
                    history,
                    summary_model=chat_model,
                    max_tokens=self.context_max_tokens,
                )
            messages.extend(history)
        messages.append(HumanMessage(content=query))
        return {"messages": messages}

    def _get_latest_ai_message(self, chunk: Any) -> Any:
        """取 chunk 中最后一条 AI 消息，不要求有文本内容（工具调用消息文本常为空）。"""
        if not isinstance(chunk, dict):
            return None
        messages = chunk.get("messages")
        if not messages:
            return None
        latest_message = messages[-1]
        if self._is_ai_message(latest_message):
            return latest_message
        return None

    def _persist_turn(self, session_id: str, query: str, final_message: Any, metadata: Dict[str, Any]) -> None:
        if self.memory_store is None:
            return

        history = self.memory_store.get_session_history(session_id)
        history.add_messages(
            [
                HumanMessage(content=query, additional_kwargs={"metadata": {}}),
                self._build_persisted_ai_message(final_message, metadata),
            ]
        )

    def _is_ai_message(self, message: Any) -> bool:
        return isinstance(message, (AIMessage, AIMessageChunk))

    def _message_to_text(self, message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "".join(text_parts)
        return str(content or "")

    def _extract_delta(self, previous_text: str, current_text: str) -> str:
        current_text = current_text or ""
        if not previous_text:
            return current_text
        if current_text.startswith(previous_text):
            return current_text[len(previous_text):]
        return current_text

    def _build_persisted_ai_message(self, message: Any, metadata: Dict[str, Any]) -> AIMessage:
        additional_kwargs = self._to_jsonable(getattr(message, "additional_kwargs", {}) or {})
        additional_kwargs.pop("metadata", None)
        return AIMessage(
            content=self._message_to_text(message),
            additional_kwargs=additional_kwargs,
            response_metadata=self._to_jsonable(getattr(message, "response_metadata", {}) or {}),
            tool_calls=self._to_jsonable(getattr(message, "tool_calls", []) or []),
            invalid_tool_calls=self._to_jsonable(getattr(message, "invalid_tool_calls", []) or []),
            usage_metadata=self._to_jsonable(getattr(message, "usage_metadata", None)),
            id=getattr(message, "id", None),
            name=getattr(message, "name", None),
        )

    def _extract_message_metadata(self, message: Any) -> Dict[str, Any]:
        if message is None:
            return {
                "usage_metadata": {},
                "response_metadata": {},
                "additional_kwargs": {},
                "tool_calls": [],
                "invalid_tool_calls": [],
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "finish_reason": None,
                "model_name": None,
            }

        usage_metadata = self._to_jsonable(getattr(message, "usage_metadata", {}) or {})
        response_metadata = self._to_jsonable(getattr(message, "response_metadata", {}) or {})
        additional_kwargs = self._to_jsonable(getattr(message, "additional_kwargs", {}) or {})
        tool_calls = self._to_jsonable(getattr(message, "tool_calls", []) or [])
        invalid_tool_calls = self._to_jsonable(getattr(message, "invalid_tool_calls", []) or [])

        token_usage = self._extract_token_usage(usage_metadata, response_metadata)
        finish_reason = None
        model_name = None
        if isinstance(response_metadata, dict):
            finish_reason = response_metadata.get("finish_reason") or response_metadata.get("stop_reason")
            model_name = response_metadata.get("model_name") or response_metadata.get("model")

        return {
            "usage_metadata": usage_metadata,
            "response_metadata": response_metadata,
            "additional_kwargs": additional_kwargs,
            "tool_calls": tool_calls,
            "invalid_tool_calls": invalid_tool_calls,
            "token_usage": token_usage,
            "finish_reason": finish_reason,
            "model_name": model_name,
        }

    def _extract_token_usage(self, usage_metadata: Any, response_metadata: Any) -> Dict[str, int]:
        sources = []
        if isinstance(usage_metadata, dict):
            sources.append(usage_metadata)
            nested_usage = usage_metadata.get("token_usage")
            if isinstance(nested_usage, dict):
                sources.append(nested_usage)
        if isinstance(response_metadata, dict):
            nested_usage = response_metadata.get("token_usage")
            if isinstance(nested_usage, dict):
                sources.append(nested_usage)

        input_tokens = self._pick_first_int(sources, ["input_tokens", "prompt_tokens"])
        output_tokens = self._pick_first_int(sources, ["output_tokens", "completion_tokens"])
        total_tokens = self._pick_first_int(sources, ["total_tokens"])

        if total_tokens == 0 and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _pick_first_int(self, sources: list[Dict[str, Any]], keys: list[str]) -> int:
        for source in sources:
            for key in keys:
                value = source.get(key)
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        return 0

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(item) for item in value]
        return str(value)


