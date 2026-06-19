"""
Day 4 实验 1: Chroma 基础 API
任务: 跑一个最小可用例子,理解 Chroma 怎么工作
"""

import chromadb

# 创建一个本地客户端,数据存在 ./chroma_demo 目录
client = chromadb.PersistentClient(path="./chroma_demo")

# 创建一个 collection (相当于一张表)
# 如果已存在则获取,不存在则创建
collection = client.get_or_create_collection(
    name="demo",
    metadata={"description": "我的第一个 Chroma 集合"}
)

# ===== 插入数据 =====
print("📝 插入 5 条测试数据...")

collection.add(
    documents=[
        "React 是一个用于构建用户界面的 JavaScript 库",
        "Vue 是一个渐进式 JavaScript 框架",
        "Python 是一种通用编程语言",
        "苹果是一种水果",
        "今天天气真好"
    ],
    metadatas=[
        {"category": "frontend", "lang": "zh"},
        {"category": "frontend", "lang": "zh"},
        {"category": "programming", "lang": "zh"},
        {"category": "food", "lang": "zh"},
        {"category": "weather", "lang": "zh"}
    ],
    ids=["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
)

print(f"✅ 已插入 {collection.count()} 条数据\n")

# ===== 查询 =====
print("🔍 查询: '前端开发框架'\n")

results = collection.query(
    query_texts=["前端开发框架"],
    n_results=3
)

# 看返回结构
print("返回结构:")
print(f"  documents: {results['documents'][0]}")
print(f"  metadatas: {results['metadatas'][0]}")
print(f"  distances: {results['distances'][0]}\n")

# 友好显示
print("=" * 60)
print("📊 检索结果:")
print("=" * 60)
for i, (doc, meta, dist) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
), 1):
    # Chroma 默认用 L2 距离,越小越相似
    print(f"\n{i}. {doc}")
    print(f"   分类: {meta['category']}")
    print(f"   距离: {dist:.4f} (越小越相似)")

print("\n" + "=" * 60)
print("💡 观察: 你应该看到 React 和 Vue 排在前面,")
print("   因为它们都和'前端框架'语义相关")
print("=" * 60)