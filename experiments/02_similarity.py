"""
Day 2 实验 2: 验证向量相似度
观察: 相似文本的向量是否真的相似?
"""

from openai import OpenAI
import math

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="bge-m3",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    返回值在 -1 到 1 之间:
    - 1: 完全相同
    - 0: 完全无关
    - -1: 完全相反
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2)


# ===== 测试用例 =====
test_groups = [
    {
        "name": "近义词测试",
        "texts": [
            "我喜欢吃苹果",
            "我爱苹果",
            "苹果是我的最爱",
            "我讨厌苹果",  # ← 反义
        ]
    },
    {
        "name": "技术领域测试",
        "texts": [
            "React 是一个前端框架",
            "React.js 是用来构建用户界面的库",
            "React 让我们能做 SPA 单页应用",
            "今天天气很好",  # ← 完全无关
        ]
    },
    {
        "name": "中英文测试",
        "texts": [
            "JavaScript is a programming language",
            "JavaScript 是一种编程语言",  # ← 同义中英文
            "Python is also a programming language",
            "I love eating apples",  # ← 完全无关
        ]
    },
]

# ===== 跑测试 =====
for group in test_groups:
    print("\n" + "=" * 70)
    print(f"📋 {group['name']}")
    print("=" * 70)
    
    texts = group["texts"]
    base_text = texts[0]
    base_vec = get_embedding(base_text)
    
    print(f"\n基准文本: 「{base_text}」")
    print(f"\n与其他文本的相似度:")
    
    for other in texts[1:]:
        other_vec = get_embedding(other)
        sim = cosine_similarity(base_vec, other_vec)
        
        # 用条形图可视化
        bar_length = int(sim * 30) if sim > 0 else 0
        bar = "█" * bar_length
        
        print(f"  vs 「{other}」")
        print(f"     相似度: {sim:.4f}  {bar}")

print("\n" + "=" * 70)
print("💡 观察:")
print("  1. 近义句的相似度应该都 > 0.7")
print("  2. 反义句相似度可能仍然较高(都在讨论同一对象)")
print("  3. 跨语言的相似度: bge-m3 中英都强")
print("=" * 70)