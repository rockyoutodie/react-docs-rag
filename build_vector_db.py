"""
把所有 React 文档 chunks 入库
"""

import json
import time
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from openai import OpenAI


# ============================================================
# 配置
# ============================================================

CHUNKS_FILE = Path("data/processed/chunks.json")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "react_docs"
BATCH_SIZE = 50  # 每批处理 50 条


# ============================================================
# Embedding 函数
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
# 加载 chunks
# ============================================================

def load_chunks():
    print(f"📂 加载 chunks: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   原始 chunk 数: {len(chunks)}")
    return chunks


# ============================================================
# 清洗 chunks (重要!)
# ============================================================

def clean_chunks(chunks):
    """过滤掉垃圾 chunks"""
    cleaned = []
    skipped_short = 0
    skipped_symbol = 0
    
    for chunk in chunks:
        content = chunk["content"].strip()
        
        # 跳过太短的
        if len(content) < 50:
            skipped_short += 1
            continue
        
        # 跳过几乎全是符号的
        import re
        # 计算字母数字字符比例
        alphanumeric = sum(1 for c in content if c.isalnum())
        if alphanumeric / len(content) < 0.3:
            skipped_symbol += 1
            continue
        
        cleaned.append(chunk)
    
    print(f"\n🧹 清洗结果:")
    print(f"   过滤 < 50 字符: {skipped_short} 个")
    print(f"   过滤 符号过多: {skipped_symbol} 个")
    print(f"   保留有效 chunks: {len(cleaned)} 个 ({len(cleaned)/len(chunks)*100:.1f}%)")
    
    return cleaned


# ============================================================
# 入库
# ============================================================

def build_db(chunks):
    print(f"\n🗄️  连接 Chroma...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 删除旧 collection,重新开始
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   删除旧 collection: {COLLECTION_NAME}")
    except:
        pass
    
    # 创建新 collection
    embedding_fn = OllamaBgeM3Embedding()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # 余弦相似度
    )
    print(f"   创建 collection: {COLLECTION_NAME}")
    
    # 分批入库
    print(f"\n📥 开始入库,共 {len(chunks)} 个 chunks,每批 {BATCH_SIZE} 个...")
    start_time = time.time()
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        
        # 准备数据
        documents = [c["content"] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "source": c.get("source", ""),
                "h1": c.get("h1", ""),
                "h2": c.get("h2", ""),
                "h3": c.get("h3", ""),
                "chunk_index": c.get("chunk_index", 0),
            }
            for c in batch
        ]
        
        # 加入 collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # 进度
        progress = (i + len(batch)) / len(chunks) * 100
        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed
        eta = (len(chunks) - i - len(batch)) / rate if rate > 0 else 0
        
        print(f"   [{progress:5.1f}%] {i + len(batch):>5}/{len(chunks)}  "
              f"速率: {rate:.1f} chunks/s  剩余: {eta:.0f}s")
    
    total_time = time.time() - start_time
    print(f"\n✅ 入库完成!")
    print(f"   总耗时: {total_time:.0f}s ({total_time/60:.1f} 分钟)")
    print(f"   平均速率: {len(chunks)/total_time:.1f} chunks/s")
    print(f"   数据库位置: {DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   总条数: {collection.count()}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("🚀 构建向量数据库")
    print("=" * 70)
    
    # 1. 加载
    chunks = load_chunks()
    
    # 2. 清洗
    chunks = clean_chunks(chunks)
    
    # 3. 入库
    build_db(chunks)


if __name__ == "__main__":
    main()