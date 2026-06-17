import os,sys,json,logging

logger = logging.getLogger("agent")
from typing import Any, Dict, Iterator

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from agent.ReAct_agent import ReActAgent
from rag.vector_store import SweepRAGVectorStore
from memory.redis_manager import RedisMemoryStore
from utils.config_handler import agent_config, chroma_config
from utils.file_handler import get_file_md5_hex
from backend.schemas import ChatRequest, CreateSessionRequest, UpdateTitleRequest

# ── 初始化（模块级，单例） ──

DATA_DIR = os.path.join(current_dir, chroma_config.get("data_path", "data"))

_redis_cfg = agent_config.get("redis", {})
MEMORY_STORE = RedisMemoryStore(
    redis_url=_redis_cfg.get("url", "redis://localhost:6379/0"),
    index_key=_redis_cfg.get("index_key", "memory:sessions"),
    ttl=_redis_cfg.get("ttl"),
)
AGENT = ReActAgent(memory_store=MEMORY_STORE)
VECTOR_STORE = SweepRAGVectorStore()

# ── FastAPI 应用 ──

app = FastAPI(title="SweepRAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/sessions")
def list_sessions():
    items = MEMORY_STORE.list_sessions()
    return JSONResponse([
        {
            "id": s["id"],
            "title": s["title"],
            "create_time": s["create_time"],
            "message_count": MEMORY_STORE.redis_client.llen(f"chat_messages:{s['id']}"),
        }
        for s in items
    ])


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    try:
        session = MEMORY_STORE.get_session(session_id)
        return JSONResponse(session)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/sessions")
def create_session(body: CreateSessionRequest):
    session = MEMORY_STORE.create_session(title=body.title)
    return JSONResponse(session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    MEMORY_STORE.delete_session(session_id)
    return JSONResponse({"ok": True})


@app.put("/api/sessions/{session_id}/title")
def update_title(session_id: str, body: UpdateTitleRequest):
    MEMORY_STORE.update_session_title(session_id, body.title)
    return JSONResponse({"ok": True})


@app.post("/api/chat")
def chat(body: ChatRequest):
    def generate() -> Iterator[str]:
        try:
            for event in AGENT.stream_events(body.query, session_id=body.session_id):
                line = json.dumps(event, ensure_ascii=False)
                yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/knowledge/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file")

    save_path = os.path.join(DATA_DIR, file.filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    md5 = get_file_md5_hex(save_path)
    if VECTOR_STORE.check_md5_exists(md5):
        os.remove(save_path)
        return JSONResponse({"ok": False, "reason": "already_exists"})

    success = VECTOR_STORE.add_single_document(save_path, md5)
    if not success:
        return JSONResponse({"ok": False, "reason": "vectorization_failed"})

    return JSONResponse({"ok": True, "filename": file.filename})


@app.get("/api/knowledge/list")
def list_documents():
    if not os.path.exists(DATA_DIR):
        return JSONResponse([])
    allowed = tuple(chroma_config.get("allow_knowledge_file_type", []))
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(allowed)
    ]
    return JSONResponse(files)


@app.delete("/api/knowledge/{filename:path}")
def delete_document(filename: str):
    """删除指定文档：从 data/ 删除文件 + 从 Chroma 删除向量 + 清理 md5.txt。"""
    file_path = os.path.normpath(os.path.join(DATA_DIR, filename))
    # 安全检查：防止路径穿越
    if not file_path.startswith(os.path.normpath(DATA_DIR)):
        raise HTTPException(400, "Invalid path")

    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    # 计算 MD5 用于 Chroma 清理
    md5_hex = get_file_md5_hex(file_path)

    # 从 data/ 删除源文件
    os.remove(file_path)

    # 从 Chroma 和 md5.txt 删除
    if md5_hex:
        VECTOR_STORE._remove_document(md5_hex)
        # 重置 BM25 缓存，下次 get_retriever() 自动重建
        VECTOR_STORE._bm25_retriever = None

    logger.info(f"已删除文档: {filename}")
    return JSONResponse({"ok": True})



@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(current_dir, "frontend", "favicon.svg")
    if not os.path.exists(favicon_path):
        raise HTTPException(404, "favicon not found")
    with open(favicon_path, "r", encoding="utf-8") as f:
        svg = f.read()
    return HTMLResponse(content=svg, media_type="image/svg+xml")

@app.get("/")
def index():
    frontend_path = os.path.join(current_dir, "frontend", "index.html")
    if not os.path.exists(frontend_path):
        raise HTTPException(404, "frontend/index.html not found")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
