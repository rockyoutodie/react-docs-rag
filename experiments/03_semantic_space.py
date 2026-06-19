"""
Day 2 实验 3: 可视化语义空间
任务: 把一组词的向量画在 2D 平面上,看看它们如何聚类
"""

from openai import OpenAI
import math

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model="bge-m3", input=text)
    return response.data[0].embedding


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2)


# ===== 一组语义相关的词 =====
words = [
    # 编程语言
    "Python", "JavaScript", "TypeScript", "Java",
    # 前端框架
    "React", "Vue", "Angular",
    # 水果
    "苹果", "香蕉", "橙子",
    # 动物
    "狗", "猫", "鸟",
]

print("正在计算所有词的向量...")
embeddings = {word: get_embedding(word) for word in words}
print("✅ 完成\n")

# ===== 计算两两相似度矩阵 =====
print("=" * 80)
print("🗺️  语义相似度矩阵 (越高越相似)")
print("=" * 80)

# 表头
print(f"\n{'':<14}", end="")
for word in words:
    print(f"{word:<10}", end="")
print()

# 矩阵
for w1 in words:
    print(f"{w1:<14}", end="")
    for w2 in words:
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        # 用颜色/符号表示相似度
        if w1 == w2:
            symbol = "─"  # 对角线
        elif sim > 0.7:
            symbol = "🟩"  # 高相似
        elif sim > 0.5:
            symbol = "🟨"  # 中相似
        else:
            symbol = "🟥"  # 低相似
        print(f"{symbol} {sim:.2f}    ", end="")
    print()

print("\n" + "=" * 80)
print("💡 你应该看到:")
print("  • 同类词之间(如 Python 和 JavaScript)相似度 > 0.7 (绿色)")
print("  • 不同类(如 Python 和 苹果)相似度 < 0.5 (红色)")
print("  • 这就是'语义空间'——意思相近的词,在向量空间里靠近")
print("=" * 80)