"""
进阶 RAG: Multi-Query + Rerank + 去重
"""

import json
import chromadb
from chromadb import EmbeddingFunction, Embeddings
from openai import OpenAI


# ============================================================
# Embedding 函数(和入库时一致)
# ============================================================

class OllamaBgeM3Embedding(EmbeddingFunction):
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model = "bge-m3"
    
    def __call__(self, input: list[str]) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.client.embeddings.create(model=self.model, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings


# ============================================================
# 全局
# ============================================================

llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="react_docs",
    embedding_function=OllamaBgeM3Embedding()
)


# ============================================================
# 技巧 1: Multi-Query 改写
# ============================================================

def generate_queries(original_query: str, n: int = 4) -> list[str]:
    """让 LLM 把一个问题改写成多个不同角度的查询"""
    
    prompt = f"""你是一个搜索查询优化专家。用户在搜索 React 技术文档。

用户的原始问题: "{original_query}"

请生成 {n} 个不同的搜索查询,用来检索相关文档。要求:
1. 从不同角度表达同一个需求
2. 优先用英文(因为文档是英文的)
3. 包含可能出现在文档里的专业术语
4. 每个查询一行,不要编号,不要其他内容

直接输出 {n} 个查询:"""

    response = llm_client.chat.completions.create(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,  # 稍高一点,增加多样性
    )
    
    # 解析成列表
    text = response.choices[0].message.content.strip()
    queries = [line.strip() for line in text.split("\n") if line.strip()]
    
    # 把原始问题也加进去
    queries = [original_query] + queries
    
    return queries[:n+1]


# ============================================================
# 技巧 2: 多查询检索 + 合并去重
# ============================================================

def multi_query_retrieve(original_query: str, top_k_per_query: int = 5) -> list[dict]:
    """用多个查询检索,合并去重"""
    
    # 1. 生成多个查询
    queries = generate_queries(original_query, n=4)
    
    print(f"\n🔀 Multi-Query 改写:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    # 2. 每个查询各自检索
    seen_ids = set()
    all_chunks = []
    
    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=top_k_per_query
        )
        
        for doc, meta, dist, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0]
        ):
            # 去重: 同一个 chunk 只保留一次
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            
            all_chunks.append({
                "id": chunk_id,
                "content": doc,
                "source": meta.get("source", ""),
                "h1": meta.get("h1", ""),
                "h2": meta.get("h2", ""),
                "similarity": 1 - dist / 2,
                "matched_query": query,  # 记录是哪个查询命中的
            })
    
    print(f"\n📊 合并去重后: {len(all_chunks)} 个候选 chunks (来自 {len(queries)} 个查询)")
    
    return all_chunks


# ============================================================
# 技巧 3: LLM Rerank
# ============================================================

def llm_rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """用 LLM 给候选 chunks 打分,选出最相关的 top_n 个"""
    
    if len(chunks) <= top_n:
        return chunks
    
    print(f"\n🎯 Rerank: 从 {len(chunks)} 个候选中精选 {top_n} 个...")
    
    # 给每个 chunk 打分
    scored = []
    for chunk in chunks:
        prompt = f"""判断下面这段文档对回答用户问题的"相关性",打分 0-10。

用户问题: {query}

文档片段:
{chunk['content'][:500]}

只输出一个 0-10 的数字,不要其他内容。10 表示高度相关并能直接回答问题,0 表示完全无关。"""

        response = llm_client.chat.completions.create(
            model="qwen2.5:14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        # 解析分数
        try:
            score_text = response.choices[0].message.content.strip()
            # 提取数字
            import re
            match = re.search(r"\d+", score_text)
            score = int(match.group()) if match else 0
        except:
            score = 0
        
        chunk["rerank_score"] = score
        scored.append(chunk)
    
    # 按 rerank 分数排序
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    print(f"   Rerank 完成,分数分布:")
    for chunk in scored[:top_n]:
        print(f"     [{chunk['rerank_score']}/10] {chunk['source']}")
    
    return scored[:top_n]


# ============================================================
# 技巧 4: 完整的进阶 RAG 流程
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
    context = ""
    for i, chunk in enumerate(chunks, 1):
        path_parts = []
        if chunk.get("h1"): path_parts.append(chunk["h1"])
        if chunk.get("h2"): path_parts.append(chunk["h2"])
        path = " > ".join(path_parts) if path_parts else "(无标题)"
        
        context += f"\n[{i}] 文件: {chunk['source']} | 章节: {path}\n"
        context += f"{chunk['content']}\n"
        context += "-" * 50
    
    return f"""【参考资料】
{context}

【用户问题】
{query}

【你的回答】"""


def advanced_rag(query: str, verbose: bool = True):
    """完整进阶 RAG: Multi-Query → Rerank → 生成"""
    
    print(f"\n{'='*70}")
    print(f"❓ 问题: {query}")
    print(f"{'='*70}")
    
    # 1. Multi-Query 检索
    candidates = multi_query_retrieve(query, top_k_per_query=5)
    
    # 2. Rerank 精排
    top_chunks = llm_rerank(query, candidates, top_n=5)
    
    # 3. 生成
    print(f"\n🤖 基于精选的 {len(top_chunks)} 个 chunks 生成回答...")
    user_prompt = build_prompt(query, top_chunks)
    
    response = llm_client.chat.completions.create(
        model="qwen2.5:14b",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    
    print(f"\n{'='*70}")
    print(f"📝 回答:")
    print(f"{'='*70}")
    print(answer)
    
    print(f"\n📚 引用来源:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"  [{i}] {chunk['source']} (rerank: {chunk.get('rerank_score', 'N/A')}/10)")
    
    return {"query": query, "chunks": top_chunks, "answer": answer}


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # 昨天失败的问题!
    advanced_rag("React 推荐用哪种方式管理表单输入?")