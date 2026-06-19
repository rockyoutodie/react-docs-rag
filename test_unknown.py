"""
测试: RAG 系统能不能正确说"我不知道"?
"""

from rag import rag


# 这些问题在 React 文档里没有
out_of_scope_queries = [
    "请告诉我今天的天气怎么样?",
    "iPhone 17 什么时候发布?",
    "怎么用 React 做股票交易?",
    "Vue 的 reactive API 怎么用?",  # 这是 Vue 不是 React
]

for q in out_of_scope_queries:
    rag(q, top_k=3, verbose=False)
    print(f"❓ {q}")
    result = rag(q, top_k=3, verbose=False)
    print(f"📝 回答: {result['answer']}\n")
    print("=" * 70)