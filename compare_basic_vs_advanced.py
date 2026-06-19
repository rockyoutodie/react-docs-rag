"""
对比: 基础 RAG vs 进阶 RAG
"""

from rag import rag  # 昨天的基础版
from rag_advanced import advanced_rag  # 今天的进阶版


def compare(query: str):
    print("\n" + "█" * 70)
    print(f"❓ {query}")
    print("█" * 70)
    
    print("\n" + "─" * 30 + " 基础 RAG (top-3) " + "─" * 30)
    basic = rag(query, top_k=3, verbose=False)
    print(basic["answer"])
    print("\n来源:")
    for i, c in enumerate(basic["chunks"], 1):
        print(f"  [{i}] {c['source']}")
    
    print("\n" + "─" * 28 + " 进阶 RAG (Multi+Rerank) " + "─" * 28)
    advanced = advanced_rag(query, verbose=False)


if __name__ == "__main__":
    queries = [
        "React 推荐用哪种方式管理表单输入?",   # 昨天失败的
        "useMemo 和 useCallback 的区别?",
        "如何在 React 中发起网络请求?",
    ]
    
    for q in queries:
        compare(q)
        input("\n按 Enter 继续...")