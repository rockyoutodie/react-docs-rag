"""
对比: RAG 回答 vs 纯 LLM 回答
看看 RAG 到底好在哪
"""

from openai import OpenAI
from rag import rag

llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def pure_llm(query: str) -> str:
    """不用 RAG,直接问 LLM"""
    response = llm_client.chat.completions.create(
        model="qwen2.5:14b",
        messages=[
            {"role": "user", "content": query}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def compare(query: str):
    print("\n" + "█" * 70)
    print(f"❓ 问题: {query}")
    print("█" * 70)
    
    print("\n" + "─" * 35 + " 纯 LLM " + "─" * 35)
    pure_answer = pure_llm(query)
    print(pure_answer)
    
    print("\n" + "─" * 35 + " RAG " + "─" * 35)
    rag_result = rag(query, top_k=3, verbose=False)
    print(rag_result["answer"])
    print("\n📚 引用来源:")
    for i, chunk in enumerate(rag_result["chunks"], 1):
        print(f"  [{i}] {chunk['source']}")


if __name__ == "__main__":
    # 这些问题考验"特定细节" - RAG 应该明显胜出
    queries = [
        "React 18 中 useTransition Hook 有什么用?",
        "useDeferredValue 和 useTransition 的区别是什么?",
        "React 文档推荐用哪种方式管理表单输入?",
    ]
    
    for q in queries:
        compare(q)
        input("\n按 Enter 继续...")