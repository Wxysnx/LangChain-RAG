"""基本RAG应用示例."""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.indexing import load_and_process_data
from src.rag_chain import create_rag_chain, create_simple_chain

# 加载环境变量
load_dotenv()

# 检查API密钥
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY环境变量未设置。请在.env文件中设置或直接导出变量。"
    )

def main():
    """运行RAG演示."""
    print("=== 简单RAG应用演示 ===")
    
    # 加载示例数据 - 使用Lilian Weng的博客文章作为示例
    print("加载和处理文档...")
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    docs = load_and_process_data(url)
    print(f"文档已分割为 {len(docs)} 个块")
    
    # 创建RAG链
    print("创建RAG链...")
    rag_chain = create_rag_chain(documents=docs)
    
    # 或者使用简单链
    # simple_chain = create_simple_chain(docs)
    
    # 交互式问答
    print("\n=== RAG问答系统 ===")
    print("输入'exit'退出")
    
    while True:
        question = input("\n输入您的问题: ")
        if question.lower() == "exit":
            break
        
        # 使用RAG链获取答案
        print("正在查询...")
        result = rag_chain.invoke({"question": question})
        
        # 打印答案
        print("\n回答:")
        print(result["answer"])
        
        # 如果使用简单链
        # answer = simple_chain.invoke({"question": question})
        # print("\n回答:")
        # print(answer)

if __name__ == "__main__":
    main()