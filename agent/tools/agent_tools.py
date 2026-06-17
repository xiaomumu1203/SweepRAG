import re
from datetime import datetime
from langchain_core.tools import tool
from ddgs import DDGS
from rag.rag_service import RagSummarizerService


rag = RagSummarizerService()

@tool(description='【第一优先】从公司知识库中检索产品资料，涵盖所有产品型号参数、功能介绍、使用说明、故障处理、维护保养等。所有产品相关问题必须优先调用此工具。')
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description='获取当前的日期和时间')
def get_current_time() -> str:
    """获取当前的日期和时间，适用于用户询问时间、日期、活动期限等场景"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@tool(description='执行数学计算，支持加减乘除四则运算')
def calculator(expression: str) -> str:

    # 只允许数字和基本运算符
    safe = re.sub(r'[^0-9+\-*/.() ]', '', expression)
    try:
        result = eval(safe, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


@tool(description='【第二优先/补充】联网搜索实时信息，仅当 rag_summarize 返回知识库信息不足时调用，用于获取实时价格、促销活动、市场行情等补充信息')
def web_search(query: str) -> str:
    """搜索互联网获取实时信息，返回前5条结果"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "未找到相关搜索结果"
        return "\n\n".join(
            f"标题：{r['title']}\n摘要：{r['body']}\n链接：{r['href']}"
            for r in results
        )
    except Exception as e:
        return f"联网搜索失败: {e}"
