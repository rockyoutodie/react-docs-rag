"""
Day 4 实验 2: 用 bge-m3 替换默认 Embedding
任务: 让 Chroma 使用我们的 Ollama bge-m3 模型
"""

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from openai import OpenAI


# ===== 自定义 Embedding 函数 =====
class OllamaBgeM3Embedding(EmbeddingFunction):
    """让 Chroma 用我们的 bge-m3 模型"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model = "bge-m3"
    
    def __call__(self, input: list[str]) -> Embeddings:
        """Chroma 会调用这个方法把文本转向量"""
        embeddings = []
        for text in input:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings


# ===== 使用 =====
embedding_fn = OllamaBgeM3Embedding()

client = chromadb.PersistentClient(path="./chroma_bge")

# 删除旧 collection (重新开始)
try:
    client.delete_collection("demo_bge")
except:
    pass

collection = client.create_collection(
    name="demo_bge",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # 用余弦相似度,与我们 Day 2 实验一致
)

print("📝 插入数据 (用 bge-m3 向量化)...")

collection.add(
    documents=[
        "React 是一个用于构建用户界面的 JavaScript 库",
        "Vue 是一个渐进式 JavaScript 框架",
        "Python 是一种通用编程语言",
        "useState 是 React 的状态管理 Hook",
        "useEffect 用于处理副作用",
        "组件化是 React 的核心思想",
        "苹果是一种水果",
        "今天天气真好"
    ],
    ids=[f"doc_{i}" for i in range(1, 9)]
)

print(f"✅ 已插入 {collection.count()} 条\n")

# ===== 测试不同查询 =====
test_queries = [
    "前端开发框架",
    "怎么管理 React 的状态",
    "处理副作用的方法",
    "什么是组件",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"🔍 查询: 「{query}」")
    print(f"{'='*60}")
    
    results = collection.query(query_texts=[query], n_results=3)
    
    for i, (doc, dist) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    ), 1):
        # cosine: 距离 0=完全相同, 2=完全相反
        # 转换成相似度: similarity = 1 - distance / 2
        similarity = 1 - dist / 2
        print(f"\n{i}. {doc}")
        print(f"   余弦距离: {dist:.4f}  相似度: {similarity:.4f}")