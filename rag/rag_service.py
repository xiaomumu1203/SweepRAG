from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from rag.vector_store import SweepRAGVectorStore
from utils.prompt_loader import load_rag_prompts
from utils.logger_handler import logger


class RagSummarizerService(object):
    def __init__(self):
        self.vector_store = SweepRAGVectorStore()
        self.prompt_txt = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_txt)
        self.model = chat_model
        self.__chain = self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        """每次调用获取最新的检索器（混合检索 or 纯向量）"""
        retriever = self.vector_store.get_retriever()
        logger.debug(f"获取检索器: {type(retriever).__name__}")
        return retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        # 每次调用都获取最新的检索器（文档池变化时，BM25/混合检索会自动更新）
        context_docs = self.retriever_docs(query)

        context = ""

        counter = 0

        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: {doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.__chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
