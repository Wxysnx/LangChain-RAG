![OpenAI](https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg){width=20}
![LangGraph](https://www.langchain.com/images/langgraph.svg){width=20}
![LangChain](https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain_favicon.ico){width=20}
![Chroma DB](https://www.trychroma.com/favicon.png){width=20}

## 项目概述

这个项目实现了一个基于LangChain和LangGraph的代理式检索增强生成(Agentic RAG)系统。不同于传统RAG系统，代理式RAG能够根据问题复杂度和检索结果动态调整工作流程，实现更智能的文档检索和问答能力。

本项目展示了我对现代RAG系统设计和实现的深入理解，结合了最新的LLM应用框架和最佳实践，适用于需要从特定知识库提取信息并生成高质量回答的各种场景。

## 技术栈

- **核心框架**：LangChain + LangGraph
- **语言模型**：OpenAI GPT模型系列
- **向量数据库**：Chroma DB (持久化向量存储)
- **嵌入模型**：OpenAI Embeddings
- **文档处理**：LangChain Document Loaders, RecursiveCharacterTextSplitter
- **工作流编排**：LangGraph StateGraph

## 核心功能与特性

### 1. 文档处理与索引 (src/indexing.py)
- 支持多种数据源的文档加载
- 使用RecursiveCharacterTextSplitter进行智能文档分块
- 通过OpenAI Embeddings生成文本向量表示

### 2. 智能检索模块 (src/retrieval.py)
- 利用Chroma DB实现高效的持久化向量存储与检索
- 支持可配置的检索参数(top k, 相似度阈值等)
- 实现基于元数据的过滤检索

### 3. 代理式RAG工作流 (src/rag_chain.py)
- 使用LangGraph构建有状态的多阶段工作流
- 实现查询分析、检索执行、结果评估和响应生成等节点
- 支持条件分支，根据检索结果质量决定下一步操作

### 4. 智能决策与反馈循环
- 自动评估检索结果相关性
- 支持查询改写以优化检索效果
- 在必要时提供无法找到答案的诚实回应