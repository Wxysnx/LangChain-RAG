"""RAG链的实现."""

from typing import Dict, List, Optional, TypedDict, Union

from langchain import hub
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from src.indexing import create_embeddings, load_and_process_data
from src.retrieval import create_vector_store, retrieve_documents
from src.utils import get_env_var

# 定义应用状态
class State(TypedDict):
    """RAG应用的状态."""
    
    question: str
    context: List[Document]
    answer: str


def create_rag_chain(
    documents: Optional[List[Document]] = None,
    source: Optional[str] = None,
    vector_store: Optional[VectorStore] = None,
    llm: Optional[BaseChatModel] = None,
) -> StateGraph:
    """
    创建RAG应用链.
    
    Args:
        documents: 预处理的文档列表，可选
        source: 文档源，如果未提供documents参数则必须提供
        vector_store: 预先创建的向量存储，可选
        llm: 语言模型，可选，默认使用OpenAI
        
    Returns:
        编译后的StateGraph实例
    """
    # 处理输入参数
    if documents is None and source is None and vector_store is None:
        raise ValueError("必须提供documents、source或vector_store参数之一")
    
    # 如果只有source参数，加载并处理数据
    if documents is None and source:
        documents = load_and_process_data(source)
    
    # 创建向量存储
    if vector_store is None and documents:
        embeddings = create_embeddings()
        vector_store = create_vector_store(documents, embeddings)
    
    # 创建语言模型
    if llm is None:
        model_name = get_env_var("LLM_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(model=model_name)
    
    # 加载RAG提示模板
    prompt = hub.pull("rlm/rag-prompt")
    
    # 定义检索步骤
    def retrieve(state: State):
        """从向量存储中检索相关文档."""
        if not vector_store:
            raise ValueError("向量存储未初始化")
        
        retrieved_docs = retrieve_documents(
            vector_store,
            state["question"],
            top_k=4,
        )
        return {"context": retrieved_docs}
    
    # 定义生成答案步骤
    def generate(state: State):
        """根据检索到的上下文生成回答."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    # 创建并编译图
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph


def create_simple_chain(
    documents: List[Document],
    llm: Optional[BaseChatModel] = None,
) -> Runnable:
    """
    创建一个简单的RAG链，不使用LangGraph.
    
    Args:
        documents: 文档列表
        llm: 语言模型，可选
        
    Returns:
        可运行的链
    """
    # 创建向量存储
    embeddings = create_embeddings()
    vector_store = create_vector_store(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 创建语言模型
    if llm is None:
        model_name = get_env_var("LLM_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(model=model_name)
    
    # 创建提示模板
    template = """你是一个有用的助手。使用以下检索到的上下文来回答问题。
    
    上下文:
    {context}
    
    问题: {question}
    
    如果你不知道答案，只需说你不知道，不要试图编造答案。尽量简洁地回答。
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 组装链
    chain = (
        {"context": retriever, "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain