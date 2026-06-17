# SweepRAG — 智能 RAG + ReAct Agent 客服系统

基于 **LangChain** + **通义千问 qwen3-max** 构建的扫地机器人智能客服系统，支持 **混合检索（向量 + BM25 双路召回 + RRF 融合）**、**ReAct 模式自主推理与工具调用**、**Redis 持久化记忆**、**摘要式上下文压缩**，同时提供 **Streamlit** 和 **FastAPI** 双前端支持。

## 项目结构

```text
SweepRAG/
├── agent/                     # ReAct Agent 核心
│   ├── ReAct_agent.py         # Agent 主类（LangChain Runnable）
│   ├── tools/
│   │   ├── agent_tools.py     # 工具定义（RAG检索/时间/计算器）
│   │   └── middleware.py      # 中间件（日志/监控）
├── backend/                   # FastAPI 后端
│   ├── main.py                # API 路由（会话/聊天/知识库）
│   └── schemas.py             # Pydantic 数据模型
├── config/                    # YAML 配置
│   ├── agent.yml              # Agent & Redis 配置
│   ├── chroma.yml             # 向量库 & 检索配置
│   ├── prompts.yml            # 提示词路径配置
│   └── rag.yml                # Embedding & 模型配置
├── data/                      # 知识库文档源文件
├── frontend/                  # FastAPI 前端页面
│   └── index.html             # 单页 HTML（SSE 流式）
├── memory/
│   ├── redis_manager.py       # Redis 会话/消息管理
│   └── context_compressor.py  # 摘要式上下文压缩（动态 Token 感知）
├── model/
│   └── factory.py             # LLM & Embedding 工厂
├── prompts/                   # 提示词模板
│   ├── main_prompt.txt        # Agent 系统提示词
│   └── rag_summarize.txt      # RAG 摘要提示词
├── rag/
│   ├── vector_store.py        # 向量库 + BM25 混合检索
│   └── rag_service.py         # RAG 检索服务
├── utils/                     # 工具函数
│   ├── config_handler.py      # YAML 配置加载
│   ├── file_handler.py        # 文件操作 / MD5
│   ├── path_tool.py           # 路径工具
│   ├── prompt_loader.py       # 提示词加载
│   ├── token_counter.py       # 本地 Token 计数（tiktoken）
│   └── logger_handler.py      # 日志配置
├── app.py                     # Streamlit 前端（可替换）
├── requirements.txt           # Python 依赖
```

## 快速开始
### 1. 环境要求
- Python 3.11+
- Redis 服务（本地或远程）
- 通义千问 API Key（[DashScope](https://dashscope.aliyun.com/)）

### 2. 安装依赖

```bash
git clone https://github.com/xiaomumu1203/SweepRAG.git
cd SweepRAG
python -m venv venv

# Windows (CMD / PowerShell):
.\venv\Scripts\activate

# Mac / Linux (Bash / Zsh):
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置环境变量

API Key 从环境变量读取，无需修改配置文件：
```bash
# 方式一（推荐）：在项目根目录创建 .env 文件，写入：
DASHSCOPE_API_KEY=sk-xxx

# 方式二（临时设置，仅当前终端有效）：
# Windows (CMD):
# set DASHSCOPE_API_KEY=sk-xxx

# Mac / Linux:
# export DASHSCOPE_API_KEY=sk-xxx
```

### 4. 启动
**方式一：Streamlit**
```bash
streamlit run app.py
```

**方式二：FastAPI**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 核心架构
### RAG 检索
```text
用户查询
    │
    ▼
混合检索（RRF 融合）
    ├── 向量检索（Chroma + Embedding） → 语义相似度
    └── BM25 关键词检索                  → 精确匹配
    │
    ▼
RRF 排序融合 → Top-K 结果
```


### ReAct Agent 推理
```text
用户输入 → _build_input（Token 检查 + 上下文压缩）
    │
    ▼
LLM 思考（qwen3-max）
    ├── 需要更多信息 → 调用工具（RAG/时间/计算器/联网搜索）
    └── 信息充足     → 生成最终回答
    │
    ▼
_persist_turn → Redis 存储
```

### 上下文压缩
当会话累计 Token 超过阈值时自动触发，使用 **tiktoken** 本地实时计数，不依赖 API 元数据：

```text
对话历史 > 16,000 Token
    │
    ├── 从后往前逐轮累计 Token，尽量填充预算
    └── 超出预算的部分 → LLM 压缩为 200 字摘要，注入 SystemPrompt
```

采用**动态 Token 感知压缩**，压缩器从最新一轮向前逐轮计算 Token 数，
在 `context_max_tokens` 预算内尽可能多地保留完整轮次（预留约 300 Token 给摘要），
而非一刀切固定保留轮数，充分利用上下文窗口容量。

### 记忆存储
- **Redis**：持久化会话、消息历史、Token 用量
- **消息结构**：`messages_to_dict / messages_from_dict` 序列化
- **Token 监控**：`response_metadata.token_usage` 记录 API 消耗；本地用 `tiktoken` 实时估算上下文大小

## 配置指南
| 文件 | 关键配置 | 说明 |
|------|---------|------|
| `config/agent.yml` | `redis.*`, `context_max_tokens` | 记忆存储、压缩阈值 |
| `config/chroma.yml` | `persist_directory`, `k`, `use_hybrid_search` | 向量库、检索参数 |
| `config/rag.yml` | `embedding_model_name`, `chunk_size` | Embedding 模型、分块大小 |
| `config/prompts.yml` | `*_path` | 提示词文件路径 |

## 技术栈
| 类别 | 技术 |
|------|------|
| **框架** | LangChain, LangGraph |
| **LLM** | 通义千问 qwen3-max（DashScope） |
| **向量库** | ChromaDB + text-embedding-v4 |
| **检索** | 向量检索 + BM25 双路召回 + RRF 融合 |
| **记忆** | Redis（持久化会话/消息） |
| **前端** | Streamlit / FastAPI + HTML/SSE |
| **配置** | YAML（PyYAML） |
| **Token 计数** | tiktoken（cl100k_base，本地实时） |
