"""处理文档索引的模块."""

import os
from typing import Dict, List, Optional, Union

import bs4
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import get_env_var


def load_document(source: str) -> List[Document]:
    """
    加载文档内容，支持网页URL或本地文件.
    
    Args:
        source: 文档源，可以是URL或文件路径
        
    Returns:
        包含文档内容的文档列表
    """
    if source.startswith(("http://", "https://")):
        # 加载网页内容
        loader = WebBaseLoader(
            web_paths=(source,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "article", "content")
                )
            ),
        )
        docs = loader.load()
    elif source.endswith(".pdf"):
        # 加载PDF文件
        loader = PyPDFLoader(source)
        docs = loader.load()
    elif source.endswith((".txt", ".md")):
        # 加载文本文件
        loader = TextLoader(source)
        docs = loader.load()
    else:
        raise ValueError(f"不支持的文档类型: {source}")
    
    return docs


def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    将文档分割成更小的块.
    
    Args:
        docs: 要分割的文档列表
        chunk_size: 每个块的最大大小（字符数）
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        分割后的文档块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    doc_chunks = text_splitter.split_documents(docs)
    return doc_chunks


def create_embeddings() -> Embeddings:
    """
    创建文本嵌入模型.
    
    Returns:
        嵌入模型
    """
    model_name = get_env_var("EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model_name)


def load_and_process_data(
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    加载并处理文档，返回分块后的文档.
    
    Args:
        source: 文档源，可以是URL或文件路径
        chunk_size: 每个块的最大大小（字符数）
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        处理后的文档块
    """
    # 加载文档
    docs = load_document(source)
    
    # 分割文档
    doc_chunks = split_documents(docs, chunk_size, chunk_overlap)
    
    return doc_chunks