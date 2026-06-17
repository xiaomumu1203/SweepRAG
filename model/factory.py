import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from utils.config_handler import rag_config
from typing import Optional
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


load_dotenv()


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        return ChatTongyi(model=rag_config['chat_model_name'],
                          dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"))


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(model=rag_config['embedding_model_name'],
                                   dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"))



chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()