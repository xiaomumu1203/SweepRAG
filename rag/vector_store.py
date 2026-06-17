import os
from langchain_core.documents import Document
from utils.path_tool import get_abs_path
from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import txt_loader,pdf_loader,listdir_with_allowed_type,get_file_md5_hex
from utils.logger_handler import logger
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


class SweepRAGVectorStore:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name= chroma_config.get("collection_name"),
            embedding_function= embedding_model,
            persist_directory= chroma_config.get("persist_directory")
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size= chroma_config.get("chunk_size"),
            chunk_overlap= chroma_config.get("chunk_overlap"),
            separators = chroma_config.get("separators"),
            length_function= len,
        )

        self._all_chunks: list[Document] = []
        self._bm25_retriever = None  # BM25索引缓存，惰性创建
        self.load_document()

        # 统一从 Chroma 重建 BM25 文档池，消除重复
        self._all_chunks = self._rebuild_chunks_from_chroma()
        logger.info(f"已加载 {len(self._all_chunks)} 个文档块用于混合检索")

    def _get_allowed_files(self):
        # 获取允许文件类型的路径
        data_path = get_abs_path(chroma_config.get("data_path"))
        allowed_types = tuple(chroma_config.get("allow_knowledge_file_type", []))
        return listdir_with_allowed_type(data_path, allowed_types)

    def _rebuild_chunks_from_chroma(self) -> list[Document]:
        try:
            results = self.vector_store.get(limit=9999)
            if not results or not results.get("documents"):
                return []
            docs = results["documents"]
            metadatas = results.get("metadatas") or []
            ids = results.get("ids") or []

            # 按 (内容, source_md5) 去重，保留第一次出现的 id
            seen: set[tuple[str, str]] = set()
            keep_ids: list[str] = []
            delete_ids: list[str] = []
            for i in range(len(docs)):
                md5 = (metadatas[i] or {}).get("source_md5", "") or ""
                key = (docs[i], md5)
                if key in seen:
                    delete_ids.append(ids[i])
                else:
                    seen.add(key)
                    keep_ids.append(ids[i])

            if delete_ids:
                self.vector_store.delete(ids=delete_ids)
                logger.info(f"去重清理: 删除了 {len(delete_ids)} 个重复文档块")

            return [
                Document(
                    page_content=docs[i],
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    id=ids[i],
                )
                for i in range(len(docs))
                if ids[i] in keep_ids
            ]
        except Exception as e:
            logger.warning(f"从 Chroma 重建文档池失败，回退到文件读取: {e}")
        return []

    def _remove_document(self, md5_hex: str):
        """从 Chroma 和 md5.txt 中删除指定 MD5 对应的所有文档块。"""
        deleted = 0
        try:
            # 先按 source_md5 标签删（适用于新上传的文件）
            ids_to_del = self.vector_store.get(
                where={"source_md5": md5_hex}, limit=9999
            ).get("ids", [])
            if ids_to_del:
                self.vector_store.delete(ids=ids_to_del)
                deleted = len(ids_to_del)
        except Exception as e:
            logger.warning(f"按 source_md5 删除 (MD5={md5_hex}) 失败: {e}")

        logger.info(f"已从 Chroma 删除 MD5={md5_hex} 的 {deleted} 个文档块")

        # 从 md5.txt 中删除该行
        md5_path = get_abs_path(chroma_config.get("md5_hex_store"))
        if os.path.exists(md5_path):
            with open(md5_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(md5_path, "w", encoding="utf-8") as f:
                f.writelines(line for line in lines if line.strip() != md5_hex)

    def get_retriever(self):
        """获取检索器：BM25+向量混合检索（RRF融合），不可用时回退纯向量检索"""
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": chroma_config.get("k", 3)})

        use_hybrid = chroma_config.get("use_hybrid_search", False)
        if use_hybrid and self._all_chunks:
            # 懒加载/重建BM25索引：从 Chroma 同步最新的文档池，确保和向量检索一致
            if self._bm25_retriever is None:
                self._all_chunks = self._rebuild_chunks_from_chroma()
                self._bm25_retriever = BM25Retriever.from_documents(self._all_chunks)
                self._bm25_retriever.k = chroma_config.get("bm25_k", 3)
                logger.info(f"BM25索引创建完成（{len(self._all_chunks)}个文档块）")
            else:
                logger.debug("复用已缓存的BM25索引")

            weights = chroma_config.get("hybrid_weights", [0.5, 0.5])
            logger.info(f"使用混合检索（RRF融合）: 向量权重={weights[0]}, BM25权重={weights[1]}")
            return EnsembleRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                weights=weights,
                c=60  # RRF 常数
            )

        logger.info("混合检索不可用，回退到纯向量检索")
        return vector_retriever

    def check_md5_exists(self, md5_hex: str):
        if not os.path.exists(get_abs_path(chroma_config.get("md5_hex_store"))):
            with open(get_abs_path(chroma_config.get("md5_hex_store")), 'w', encoding='utf-8') as f:
                pass
            return False
        with open(get_abs_path(chroma_config.get("md5_hex_store")),'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line == md5_hex:
                    return True
            return False

    def save_md5(self, md5_hex: str):
        with open(get_abs_path(chroma_config.get("md5_hex_store")),'a',encoding='utf-8') as f:
            f.write(md5_hex + "\n")

    def add_single_document(self, file_path: str, md5_hex: str):
        # 防护：Chroma 中已有相同 MD5 tag 的块 → 跳过（防止 md5.txt 丢失后的重复入库）
        try:
            dup = self.vector_store.get(where={"source_md5": md5_hex})
            if dup and dup.get("ids"):
                self.save_md5(md5_hex)
                return True
        except Exception:
            pass

        try:
            document: list[Document] = []
            if file_path.endswith('txt'):
                document = txt_loader(file_path)
            elif file_path.endswith('pdf'):
                document = pdf_loader(file_path)

            if not document:
                logger.warning(f"{file_path}知识库内没有有效内容")
                return False

            split_document: list[Document] = self.spliter.split_documents(document)

            if not split_document:
                logger.warning(f"{file_path}分片后没有有效内容")
                return False

            # 给每个块打上来源标签，便于文件删除时从 Chroma 连坐清理
            for doc in split_document:
                doc.metadata["source_md5"] = md5_hex

            self.vector_store.add_documents(split_document)
            self.save_md5(md5_hex)

            # 同步更新BM25文档池，清除BM25索引缓存（下次get_retriever时重建）
            self._all_chunks.extend(split_document)
            self._bm25_retriever = None
            logger.info(f"{file_path}内容加载成功，当前共{len(self._all_chunks)}个文档块")
            return True

        except Exception as e:
            logger.error(f"{file_path}内容加载失败，{str(e)}")
            return False

    def load_document(self):
        md5_path = get_abs_path(chroma_config.get("md5_hex_store"))

        # 收集 data/ 目录下所有文件的 MD5
        existing_md5s = set()
        for file_path in self._get_allowed_files():
            md5_hex = get_file_md5_hex(file_path)
            if md5_hex is None:
                logger.error(f"无法计算 {file_path} 的 MD5，跳过该文件")
                continue
            existing_md5s.add(md5_hex)

            if self.check_md5_exists(md5_hex):
                continue
            self.add_single_document(file_path, md5_hex)

        # 同步删除：md5.txt 中有但 data/ 里已不存在的文件 → 从 Chroma 连坐清理
        if not os.path.exists(md5_path):
            return
        with open(md5_path, 'r', encoding='utf-8') as f:
            stored_md5s = [line.strip() for line in f.readlines() if line.strip()]

        orphaned = [md5 for md5 in stored_md5s if md5 not in existing_md5s]
        for md5 in orphaned:
            self._remove_document(md5)

