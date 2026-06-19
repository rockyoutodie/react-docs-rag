"""
处理全部 React 文档,切分后保存为 JSON
"""

import json
from pathlib import Path
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# ===== 配置 =====
DATA_DIR = Path("data/raw/react.dev/src/content")  # 改成你的实际路径
OUTPUT_FILE = Path("data/processed/chunks.json")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def process_markdown_file(file_path: Path) -> list[dict]:
    """处理单个 markdown 文件,返回切好的 chunks"""
    
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    
    # 跳过空文件或太短的文件
    if len(content) < 100:
        return []
    
    # 第一步: 按 Markdown 标题切
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    
    try:
        md_chunks = md_splitter.split_text(content)
    except Exception as e:
        # 没有标题的文件,直接进第二步
        md_chunks = [type("obj", (), {"page_content": content, "metadata": {}})()]
    
    # 第二步: 大块再细切
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "!", "?", ".", " ", ""],
    )
    
    final_chunks = []
    
    for md_chunk in md_chunks:
        # 提取内容和元信息
        if hasattr(md_chunk, "page_content"):
            text = md_chunk.page_content
            md_meta = md_chunk.metadata
        else:
            text = md_chunk
            md_meta = {}
        
        # 细切
        sub_chunks = text_splitter.split_text(text)
        
        for i, sub in enumerate(sub_chunks):
            chunk_data = {
                "content": sub,
                "source": str(file_path.relative_to(DATA_DIR)),
                "h1": md_meta.get("h1", ""),
                "h2": md_meta.get("h2", ""),
                "h3": md_meta.get("h3", ""),
                "chunk_index": i,
            }
            final_chunks.append(chunk_data)
    
    return final_chunks


def main():
    print("=" * 70)
    print("🚀 处理 React 文档")
    print("=" * 70)
    
    # 检查数据目录
    if not DATA_DIR.exists():
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        print("请确认你下载的文档路径")
        return
    
    # 收集所有 .md 文件
    md_files = list(DATA_DIR.rglob("*.md"))
    print(f"📚 找到 {len(md_files)} 个 markdown 文件\n")
    
    # 处理每个文件
    all_chunks = []
    
    for i, file_path in enumerate(md_files, 1):
        chunks = process_markdown_file(file_path)
        all_chunks.extend(chunks)
        
        if i % 20 == 0 or i == len(md_files):
            print(f"  处理进度: {i}/{len(md_files)} 文件,累计 {len(all_chunks)} 个 chunks")
    
    print(f"\n✅ 处理完成")
    print(f"  总文件数: {len(md_files)}")
    print(f"  总 chunk 数: {len(all_chunks)}")
    print(f"  平均每文件: {len(all_chunks) / len(md_files):.1f} chunks")
    
    # chunk 长度分布
    chunk_lengths = [len(c["content"]) for c in all_chunks]
    print(f"\n📊 Chunk 长度分布:")
    print(f"  最短: {min(chunk_lengths)} 字符")
    print(f"  最长: {max(chunk_lengths)} 字符")
    print(f"  平均: {sum(chunk_lengths) / len(chunk_lengths):.0f} 字符")
    
    # 保存
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n💾 已保存到: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    
    # 抽样几个 chunks
    print(f"\n📝 抽样 chunks (前 3 个):")
    for chunk in all_chunks[:3]:
        print(f"\n--- Source: {chunk['source']} ---")
        if chunk["h1"]:
            print(f"路径: {chunk['h1']} > {chunk.get('h2', '')} > {chunk.get('h3', '')}")
        print(f"内容: {chunk['content'][:200]}...")


if __name__ == "__main__":
    main()