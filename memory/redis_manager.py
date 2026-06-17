import json
import time
import uuid
from typing import Any, Dict, List, Optional

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)


class RedisChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, redis_client: redis.Redis, session_id: str, ttl: Optional[int] = None):
        self.redis_client = redis_client
        self.session_id = session_id
        self.ttl = ttl
        self.key = f"chat_messages:{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:
        try:
            items = self.redis_client.lrange(self.key, 0, -1)
        except redis.RedisError:
            items = []
        if not items:
            return []
        dicts = [json.loads(item) for item in items]
        return messages_from_dict(dicts)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        dicts = messages_to_dict(messages)
        for d in dicts:
            self.redis_client.rpush(self.key, json.dumps(d, ensure_ascii=False))
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        self.redis_client.delete(self.key)


class RedisMemoryStore:

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        index_key: str = "memory:sessions",
        ttl: Optional[int] = None,
    ):
        self.redis_client = redis.from_url(redis_url)
        self.redis_client.ping()  # fail early if cannot connect
        self.index_key = index_key
        self.ttl = ttl

    def list_sessions(self) -> List[Dict[str, Any]]:
        raw = self.redis_client.hgetall(self.index_key)
        sessions = []
        expired = []
        now = time.time()
        for sid, data in raw.items():
            sid_str = sid.decode() if isinstance(sid, bytes) else sid
            session = json.loads(data)
            # 有 TTL 配置时，检查 ttl_epoch 是否过期
            if self.ttl is not None:
                ttl_epoch = session.get("ttl_epoch", 0)
                if ttl_epoch and now - ttl_epoch > self.ttl:
                    expired.append(sid_str)
                    continue
            sessions.append(session)
        for sid in expired:
            self.redis_client.hdel(self.index_key, sid)
            self.redis_client.delete(f"chat_messages:{sid}")
        return sorted(sessions, key=lambda item: item.get("create_time", ""), reverse=True)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        data = self.redis_client.hget(self.index_key, session_id)
        if not data:
            raise KeyError(f"Session not found: {session_id}")
        session = json.loads(data)
        # 有 TTL 配置时，检查 ttl_epoch 是否过期
        if self.ttl is not None:
            ttl_epoch = session.get("ttl_epoch", 0)
            if ttl_epoch and time.time() - ttl_epoch > self.ttl:
                self.redis_client.hdel(self.index_key, session_id)
                self.redis_client.delete(f"chat_messages:{session_id}")
                raise KeyError(f"Session expired: {session_id}")
        session["messages"] = self.get_session_messages(session_id)
        return session

    def create_session(self, title: str = "新会话") -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        session = {
            "id": session_id,
            "title": title,
            "create_time": created_at,
        }
        if self.ttl is not None:
            session["ttl_epoch"] = time.time()
        self.redis_client.hset(self.index_key, session_id, json.dumps(session, ensure_ascii=False))
        # 不预创建 chat_messages key，由首次 add_messages 自动创建
        return {**session, "messages": []}

    def delete_session(self, session_id: str) -> None:
        self.redis_client.hdel(self.index_key, session_id)
        self.redis_client.delete(f"chat_messages:{session_id}")

    def update_session_title(self, session_id: str, title: str) -> None:
        data = self.redis_client.hget(self.index_key, session_id)
        if not data:
            return
        session = json.loads(data)
        session["title"] = title
        self.redis_client.hset(self.index_key, session_id, json.dumps(session, ensure_ascii=False))

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        history = self._get_history(session_id)
        return [self._serialize_message(m) for m in history.messages]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return self._get_history(session_id)

    def add_user_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        history = self._get_history(session_id)
        history.add_message(HumanMessage(content=content, additional_kwargs={"metadata": metadata or {}}))

    def add_ai_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        metadata = metadata or {}
        history = self._get_history(session_id)
        history.add_message(
            AIMessage(
                content=content,
                additional_kwargs={"metadata": metadata},
                response_metadata=metadata.get("response_metadata", {}),
            )
        )

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(self.redis_client, session_id, ttl=self.ttl)

    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        role = "assistant"
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"

        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        metadata = self._to_jsonable(additional_kwargs.get("metadata", {}) or {})
        response_metadata = getattr(message, "response_metadata", {}) or {}
        usage_metadata = getattr(message, "usage_metadata", None)
        tool_calls = getattr(message, "tool_calls", []) or []
        invalid_tool_calls = getattr(message, "invalid_tool_calls", []) or []

        if response_metadata and "response_metadata" not in metadata:
            metadata["response_metadata"] = self._to_jsonable(response_metadata)
        if usage_metadata is not None and "usage_metadata" not in metadata:
            metadata["usage_metadata"] = self._to_jsonable(usage_metadata)
        if tool_calls and "tool_calls" not in metadata:
            metadata["tool_calls"] = self._to_jsonable(tool_calls)
        if invalid_tool_calls and "invalid_tool_calls" not in metadata:
            metadata["invalid_tool_calls"] = self._to_jsonable(invalid_tool_calls)

        return {
            "role": role,
            "content": self._message_to_text(message.content),
            "metadata": metadata,
        }


    def _message_to_text(self,content: Any) -> str:
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


    def _to_jsonable(self,value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(item) for item in value]
        return str(value)
