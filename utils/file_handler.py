import os,hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader


#获得文件的md5用来去重
def get_file_md5_hex(file_path:str):

    #判断当前文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"{file_path}不存在！")
        return None

    #判断当前路径是否是文件
    if not os.path.isfile(file_path):
        logger.error(f"{file_path}不是文件！")
        return None

    md5 = hashlib.md5()

    chunk_size = 8*1024       #按照8kb分块
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
            md5_hex = md5.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算{file_path}文件的md5值失败，{str(e)}")
        return None


#获取允许的文件列表
def listdir_with_allowed_type(path:str, allowed_types:tuple[str]):
    files = []

    if not os.path.isdir(path):
        logger.error(f"{path}不是文件夹")
        return ()

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(file_path:str,password=None) -> list[Document]:
    return PyPDFLoader(file_path,password).load()

def txt_loader(file_path) -> list[Document]:
    return TextLoader(file_path,encoding="utf-8").load()

def md_loader(file_path: str) -> list[Document]:
    """使用 unstructured 加载 Markdown 文件"""
    from unstructured.partition.auto import partition
    elements = partition(filename=file_path)
    text = "\n\n".join(str(el) for el in elements)
    return [Document(page_content=text, metadata={"source": file_path})]

def docx_loader(file_path: str) -> list[Document]:
    """使用 unstructured 加载 Word 文档"""
    from unstructured.partition.auto import partition
    elements = partition(filename=file_path)
    text = "\n\n".join(str(el) for el in elements)
    return [Document(page_content=text, metadata={"source": file_path})]



