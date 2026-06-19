"""
完整 RAG 系统: 检索 + 生成
"""

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from openai import OpenAI


# ============================================================
# Embedding 函数(和入库时一致)
# ============================================================

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


# ============================================================
# 全局: LLM 客户端 + Chroma 客户端
# ============================================================

llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="react_docs",
    embedding_function=OllamaBgeM3Embedding()
)


# ============================================================
# Step 1: 检索 (Retrieval)
# ============================================================

def retrieve(query: str, top_k: int = 5):
    """根据查询,返回最相关的 top_k 个 chunks"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "content": doc,
            "source": meta.get("source", ""),
            "h1": meta.get("h1", ""),
            "h2": meta.get("h2", ""),
            "h3": meta.get("h3", ""),
            "similarity": 1 - dist / 2,  # 余弦相似度
        })
    
    return chunks


# ============================================================
# Step 2: 拼装 Prompt (Augmentation)
# ============================================================

RAG_SYSTEM_PROMPT = """你是一个基于参考资料回答问题的专业助手,专门帮助开发者理解 React 文档。

【回答原则】
1. 严格基于"参考资料"回答,不要使用你训练时的其他知识
2. 如果参考资料中没有相关信息,直接说"根据现有文档,我没找到相关信息"
3. 不要编造任何内容
4. 在回答的关键事实后,标注来源,例如 [1] [2]
5. 如果用户用中文提问,用中文回答;用英文提问,用英文回答
6. 回答要简洁、准确,有代码示例时优先展示代码"""


def build_prompt(query: str, chunks: list) -> str:
    """把检索到的 chunks 拼成 prompt"""
    
    context = ""
    for i, chunk in enumerate(chunks, 1):
        # 标题路径
        path_parts = []
        if chunk.get("h1"): path_parts.append(chunk["h1"])
        if chunk.get("h2"): path_parts.append(chunk["h2"])
        if chunk.get("h3"): path_parts.append(chunk["h3"])
        path = " > ".join(path_parts) if path_parts else "(无标题)"
        
        context += f"\n[{i}] 文件: {chunk['source']} | 章节: {path}\n"
        context += f"{chunk['content']}\n"
        context += "-" * 50
    
    user_prompt = f"""【参考资料】
{context}

【用户问题】
{query}

【你的回答】"""
    
    return user_prompt


# ============================================================
# Step 3: 调用 LLM (Generation)
# ============================================================

def generate(query: str, chunks: list, model: str = "qwen2.5:14b") -> str:
    """让 LLM 基于 chunks 生成回答"""
    user_prompt = build_prompt(query, chunks)
    
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,  # 低温度,减少胡说八道
    )
    
    return response.choices[0].message.content


# ============================================================
# RAG 主流程
# ============================================================

def rag(query: str, top_k: int = 5, verbose: bool = True):
    """完整 RAG 流程"""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"❓ 问题: {query}")
        print(f"{'='*70}")
    
    # 1. 检索
    if verbose:
        print(f"\n🔍 检索 Top-{top_k} 相关文档...")
    chunks = retrieve(query, top_k=top_k)
    
    if verbose:
        for i, chunk in enumerate(chunks, 1):
            path_parts = []
            if chunk.get("h1"): path_parts.append(chunk["h1"])
            if chunk.get("h2"): path_parts.append(chunk["h2"])
            path = " > ".join(path_parts) if path_parts else "(无标题)"
            
            print(f"  [{i}] {chunk['source']} → {path}")
            print(f"      相似度: {chunk['similarity']:.4f}")
            print(f"      预览: {chunk['content'][:100]}...")
    
    # 2. 生成
    if verbose:
        print(f"\n🤖 LLM 基于参考资料生成回答...")
    answer = generate(query, chunks)
    
    # 3. 返回
    if verbose:
        print(f"\n{'='*70}")
        print(f"📝 回答:")
        print(f"{'='*70}")
        print(answer)
        print()
    
    return {
        "query": query,
        "chunks": chunks,
        "answer": answer
    }


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    test_queries = [
        "What is useState?",
        "什么是 useState?",
        "How do I use useEffect to fetch data?",
        "React 中如何避免不必要的重新渲染?",
        "useMemo 和 useCallback 的区别是什么?",
    ]
    
    for query in test_queries:
        rag(query, top_k=3)
        input("\n按 Enter 继续...")