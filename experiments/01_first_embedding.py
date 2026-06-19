"""
Day 2 实验 1: 你的第一个向量
任务: 把一句话变成向量,看看长啥样
"""

from openai import OpenAI 

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="bge-m3",
        input=text
    )
    return response.data[0].embedding 

# 测试
text = "我喜欢吃苹果"
print(f"原文: {text}")

vector = get_embedding(text)

print(f"\n向量类型: {type(vector)}")
print(f"向量维度: {len(vector)}")
print(f"\n前 10 个维度: {vector[:10]}")
print(f"\n后 10 个维度: {vector[-10:]}")

# 看看向量"分布"
import statistics
print(f"\n向量统计:")
print(f"  最大值: {max(vector):.4f}")
print(f"  最小值: {min(vector):.4f}")
print(f"  平均值: {statistics.mean(vector):.4f}")
print(f"  标准差: {statistics.stdev(vector):.4f}")