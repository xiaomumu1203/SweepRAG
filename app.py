import streamlit as st
import time, os, sys
from utils.path_tool import get_abs_path
from utils.config_handler import agent_config, chroma_config
from agent.ReAct_agent import ReActAgent
from rag.vector_store import SweepRAGVectorStore
from utils.file_handler import get_file_md5_hex
from memory.redis_manager import RedisMemoryStore

# 1. 强行锁定根目录，防止 Streamlit 找不到自定义包
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

DATA_DIR = get_abs_path(chroma_config.get("data_path"))

# Redis 记忆后端
_redis_cfg = agent_config.get("redis", {})
MEMORY_STORE = RedisMemoryStore(
    redis_url=_redis_cfg.get("url", "redis://localhost:6379/0"),
    index_key=_redis_cfg.get("index_key", "memory:sessions"),
    ttl=_redis_cfg.get("ttl"),
)
AGENT_VERSION = 2


def handle_document_upload(uploaded_file):
    """公共函数：处理文档上传和向量化"""
    if uploaded_file is None:
        return
    
    # 保存文件到 data 目录
    save_path = os.path.join(DATA_DIR, uploaded_file.name)
    
    # 写入上传的文件
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 计算 MD5
    md5 = get_file_md5_hex(save_path)
    
    # 检查是否已存在
    if st.session_state["vector_store"].check_md5_exists(md5):
        st.warning(f"⚠️ 文档 '{uploaded_file.name}' 已在知识库中，无需重复上传！")
        os.remove(save_path)  # 删掉重复上传的
    else:
        with st.status("正在向量化文档...", expanded=True) as status:
            st.write(f"✅ 文档已保存: {uploaded_file.name}")
            
            # 调用向量化
            success = st.session_state["vector_store"].add_single_document(save_path, md5)
            
            if success:
                status.update(label="✅ 向量化成功！", state="complete", expanded=False)
                st.success(f"文档 '{uploaded_file.name}' 已成功加入知识库！")
            else:
                status.update(label="❌ 向量化失败！", state="error", expanded=False)
                st.error(f"文档 '{uploaded_file.name}' 处理失败，请查看日志！")


def load_sessions():
    sessions = {}
    for session in MEMORY_STORE.list_sessions():
        sessions[session["id"]] = MEMORY_STORE.get_session(session["id"])
    return sessions


def save_sessions(sessions):
    st.session_state["sessions"] = load_sessions()


def create_new_session():
    return MEMORY_STORE.create_session()


# 标题
st.set_page_config(page_title="智能机器人智能客服", layout="centered")

# 初始化 session_state
if st.session_state.get("agent_version") != AGENT_VERSION:
    st.session_state["agent"] = ReActAgent(memory_store=MEMORY_STORE)
    st.session_state["agent_version"] = AGENT_VERSION
    
# 确保 vector_store 有新方法（兼容旧 session）
if (
    "vector_store" not in st.session_state 
    or not hasattr(st.session_state["vector_store"], "check_md5_exists")
):
    st.session_state["vector_store"] = SweepRAGVectorStore()

if "sessions" not in st.session_state:
    st.session_state["sessions"] = load_sessions()
else:
    st.session_state["sessions"] = load_sessions()

if "current_session_id" not in st.session_state:
    # 如果没有会话，创建一个新的
    if not st.session_state["sessions"]:
        new_session = create_new_session()
        st.session_state["sessions"][new_session["id"]] = new_session
        st.session_state["current_session_id"] = new_session["id"]
    else:
        # 否则选择第一个会话
        st.session_state["current_session_id"] = next(iter(st.session_state["sessions"].keys()))


# ========== 侧边栏：顶部三个标签页 ==========
with st.sidebar:
    tab1, tab2 = st.tabs(["对话历史", "知识库管理"])
    
    # --- 标签页1：对话历史 ---
    with tab1:
        st.subheader("对话历史")

        # 新建对话按钮
        if st.button("➕ 新建对话", use_container_width=True):
            new_session = create_new_session()
            st.session_state["sessions"] = load_sessions()
            st.session_state["current_session_id"] = new_session["id"]
            st.rerun()

        st.divider()

        # 显示会话列表
        for session_id, session in st.session_state["sessions"].items():
            # 会话标题（如果有用户消息，用第一条用户消息作为标题）
            title = session["title"]
            if session["messages"]:
                first_user_msg = next((m["content"] for m in session["messages"] if m["role"] == "user"), None)
                if first_user_msg:
                    title = first_user_msg[:10] + ("..." if len(first_user_msg) > 20 else "")

            # 显示会话按钮
            if st.button(
                    title,
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type="primary" if session_id == st.session_state["current_session_id"] else "secondary"
            ):
                st.session_state["current_session_id"] = session_id
                st.rerun()

        st.divider()

        # 删除当前对话按钮
        if len(st.session_state["sessions"]) > 1:
            if st.session_state.get("confirm_delete_session"):
                st.warning("确定删除当前会话？")
                c1, c2 = st.columns(2)
                if c1.button("✅ 确认", use_container_width=True):
                    MEMORY_STORE.delete_session(st.session_state["current_session_id"])
                    st.session_state["sessions"] = load_sessions()
                    st.session_state["current_session_id"] = next(iter(st.session_state["sessions"].keys()))
                    st.session_state["confirm_delete_session"] = False
                    st.rerun()
                if c2.button("❌ 取消", use_container_width=True):
                    st.session_state["confirm_delete_session"] = False
                    st.rerun()
            else:
                if st.button("🗑️ 删除当前对话", use_container_width=True, type="secondary"):
                    st.session_state["confirm_delete_session"] = True
                    st.rerun()
    
    # --- 标签页2：知识库管理 ---
    with tab2:
        st.subheader("📚 知识库管理")
        
        # 文件上传（调用公共函数）
        uploaded_file = st.file_uploader(
            "上传文档 (支持 PDF / TXT)", 
            type=["pdf", "txt"], 
            accept_multiple_files=False,
            key="sidebar_uploader"
        )
        handle_document_upload(uploaded_file)
        
        st.divider()
        
        # 显示已上传文档列表
        st.subheader("📂 已上传文档")
        
        # 遍历 data 目录
        if os.path.exists(DATA_DIR):
            files = os.listdir(DATA_DIR)
            # 过滤只显示允许的文件类型
            allowed_types = tuple(chroma_config["allow_knowledge_file_type"])
            doc_files = [f for f in files if f.lower().endswith(allowed_types)]
            
            if len(doc_files) > 0:
                for f in doc_files:
                    c1, c2 = st.columns([4, 2])
                    c1.text(f"• {f}")
                    pending = st.session_state.get("confirm_delete_doc")
                    if pending == f:
                        sub1, sub2 = c2.columns(2)
                        if sub1.button("✅", key=f"ok_{f}", help="确认删除", use_container_width=True):
                            file_path = os.path.join(DATA_DIR, f)
                            md5 = get_file_md5_hex(file_path)
                            if md5:
                                st.session_state["vector_store"]._remove_document(md5)
                                st.session_state["vector_store"]._bm25_retriever = None
                            os.remove(file_path)
                            st.success(f"已删除: {f}")
                            st.session_state["confirm_delete_doc"] = None
                            st.rerun()
                        if sub2.button("✕", key=f"no_{f}", help="取消", use_container_width=True):
                            st.session_state["confirm_delete_doc"] = None
                            st.rerun()
                    else:
                        if c2.button("🗑️", key=f"del_{f}", help=f"删除 {f}", use_container_width=True):
                            st.session_state["confirm_delete_doc"] = f
                            st.rerun()
            else:
                st.info("暂无文档，先上传一个吧！")
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            st.info("暂无文档，先上传一个吧！")


# ========== 主内容区：对话界面 ==========
current_session = st.session_state["sessions"][st.session_state["current_session_id"]]

st.title("SweepRAG 智能客服")
st.divider()

# --- 对话历史 ---
for message in current_session["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
query = st.chat_input()

if query:
    st.chat_message("user").write(query)
    current_session["messages"].append({"role": "user", "content": query, "metadata": {}})

    # 如果是第一条消息，更新会话标题
    if sum(1 for message in current_session["messages"] if message["role"] == "user") == 1:
        current_session["title"] = query[:20] + ("..." if len(query) > 20 else "")
        MEMORY_STORE.update_session_title(current_session["id"], current_session["title"])

    response_messages = []
    response_metadata = {}
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].stream_events(query, session_id=current_session["id"])

        def capture(generator, cache_list, metadata_holder):
            for event in generator:
                if event.get("type") == "metadata":
                    metadata_holder.update(event.get("metadata", {}))
                    continue

                chunk = event.get("text", "")
                if not chunk:
                    continue

                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages, response_metadata))
        assistant_content = "".join(response_messages)
        current_session["messages"].append(
            {"role": "assistant", "content": assistant_content, "metadata": response_metadata}
        )
        save_sessions(st.session_state["sessions"])
        st.rerun()
