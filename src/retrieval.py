"""处理检索功能的模块."""

import os
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma  # 修改为 Chroma

from src.indexing import create_embeddings


def create_vector_store(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
    persist_directory: str = "./data/chroma_db",  # 添加持久化目录参数
) -> VectorStore:
    """
    创建并填充向量存储.
   
    Args:
        documents: 要添加到向量存储的文档块
        embeddings: 嵌入模型，如果未指定则创建默认模型
        persist_directory: Chroma数据库的持久化目录
       
    Returns:
        填充了文档的向量存储
    """
    if embeddings is None:
        embeddings = create_embeddings()
    
    # 确保持久化目录存在
    os.makedirs(persist_directory, exist_ok=True)
    
    # 创建Chroma向量数据库并添加文档
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # 持久化到磁盘
    vector_store.persist()
    
    return vector_store


def load_vector_store(
    persist_directory: str = "./data/chroma_db",
    embeddings: Optional[Embeddings] = None,
) -> VectorStore:
    """
    加载现有的向量数据库.
    
    Args:
        persist_directory: Chroma数据库的持久化目录
        embeddings: 嵌入模型，如果未指定则创建默认模型
        
    Returns:
        加载的向量数据库
    """
    if embeddings is None:
        embeddings = create_embeddings()
        
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"向量数据库目录不存在: {persist_directory}")
        
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def retrieve_documents(
    vector_store: VectorStore,
    query: str,
    top_k: int = 4,
    filter_dict: Optional[Dict] = None,
) -> List[Document]:
    """
    根据查询从向量存储中检索文档.
   
    Args:
        vector_store: 向量存储
        query: 查询文本
        top_k: 要检索的文档数量
        filter_dict: 可选的过滤条件
       
    Returns:
        检索到的文档列表
    """
    if filter_dict:
        docs = vector_store.similarity_search(query, k=top_k, filter=filter_dict)
    else:
        docs = vector_store.similarity_search(query, k=top_k)
   
    return docs