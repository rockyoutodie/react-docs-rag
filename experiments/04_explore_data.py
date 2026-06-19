"""
Day 2 实验 4: 探索你的数据
任务: 看看你下载的文档长啥样,有多少
"""

import os
from pathlib import Path

# 改成你的实际路径
DATA_DIR = Path("data/raw/react.dev/src/content")

if not DATA_DIR.exists():
    print(f"⚠️ 路径不存在: {DATA_DIR}")
    print("请改成你下载文档的实际路径")
    exit()

# 统计 markdown 文件
md_files = list(DATA_DIR.rglob("*.md"))

print(f"📊 数据统计")
print(f"=" * 60)
print(f"目录: {DATA_DIR}")
print(f"总文件数: {len(md_files)}")

total_chars = 0
total_lines = 0

for f in md_files[:5]:  # 只显示前 5 个
    content = f.read_text(encoding="utf-8", errors="ignore")
    chars = len(content)
    lines = len(content.split("\n"))
    total_chars += chars
    total_lines += lines
    
    print(f"\n📄 {f.relative_to(DATA_DIR)}")
    print(f"   字数: {chars}")
    print(f"   行数: {lines}")
    print(f"   预览: {content[:100]}...")

# 全部统计
all_chars = sum(
    len(f.read_text(encoding="utf-8", errors="ignore"))
    for f in md_files
)

print(f"\n{'=' * 60}")
print(f"📈 总计")
print(f"   文件数: {len(md_files)}")
print(f"   总字符数: {all_chars:,}")
print(f"   平均文件大小: {all_chars // len(md_files):,} 字符")