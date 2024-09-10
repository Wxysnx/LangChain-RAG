"""测试RAG链."""

import os
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag_chain import create_rag_chain


class TestRAGChain(unittest.TestCase):
    """RAG链的测试类."""

    def setUp(self):
        """设置测试环境."""
        # 创建测试文档
        self.test_docs = [
            Document(page_content="人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。", metadata={"source": "test"}),
            Document(page_content="机器学习是人工智能的一个子领域，专注于开发能从数据中学习的算法。", metadata={"source": "test"}),
            Document(page_content="深度学习是机器学习的一种方法，它使用神经网络进行学习。", metadata={"source": "test"}),
        ]

    @patch("src.rag_chain.create_vector_store")
    @patch("src.rag_chain.create_embeddings")
    @patch("src.rag_chain.ChatOpenAI")
    def test_create_rag_chain(self, mock_chat_openai, mock_create_embeddings, mock_create_vector_store):
        """测试RAG链的创建."""
        # 设置模拟对象
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_vector_store = MagicMock()
        mock_create_vector_store.return_value = mock_vector_store
        
        # 创建RAG链
        rag_chain = create_rag_chain(documents=self.test_docs)
        
        # 验证创建是否成功
        self.assertIsNotNone(rag_chain)
        mock_create_vector_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()