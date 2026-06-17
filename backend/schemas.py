from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── 请求模型 ──

class ChatRequest(BaseModel):
    query: str
    session_id: str


class CreateSessionRequest(BaseModel):
    title: str = "新会话"


class UpdateTitleRequest(BaseModel):
    title: str


# ── 响应模型 ──

class MessageResponse(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = {}


class SessionResponse(BaseModel):
    id: str
    title: str
    create_time: str
    messages: List[MessageResponse] = []


class SessionListItem(BaseModel):
    id: str
    title: str
    create_time: str
    message_count: int = 0
