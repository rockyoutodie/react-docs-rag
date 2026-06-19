"""
对建好的向量数据库做语义搜索
"""

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from openai import OpenAI


class OllamaBgeM3Embedding(EmbeddingFunction):
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model = "bge-m3"
    
    def __call__(self, input: list[str]) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings


def search(query: str, top_k: int = 5):
    """语义搜索"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(
        name="react_docs",
        embedding_function=OllamaBgeM3Embedding()
    )
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    print(f"\n{'='*70}")
    print(f"🔍 查询: 「{query}」")
    print(f"{'='*70}\n")
    
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        similarity = 1 - dist / 2
        
        # 标题路径
        path_parts = []
        if meta.get("h1"): path_parts.append(meta["h1"])
        if meta.get("h2"): path_parts.append(meta["h2"])
        if meta.get("h3"): path_parts.append(meta["h3"])
        path = " > ".join(path_parts) if path_parts else "(no title)"
        
        print(f"【{i}】 相似度: {similarity:.4f}")
        print(f"  📄 文件: {meta['source']}")
        print(f"  📍 路径: {path}")
        print(f"  📝 内容: {doc[:300]}{'...' if len(doc) > 300 else ''}")
        print()


if __name__ == "__main__":
    # 测试 5 个真实的 React 问题
    queries = [
        "What is useState?",
        "How do React Hooks work?",
        "怎么处理副作用",   # 中文也试试 (bge-m3 中英都支持)
        "function component vs class component",
        "memoization in React",
    ]
    
    for query in queries:
        search(query, top_k=3)
        input("\n按 Enter 继续下一个查询...")