"""
Day 3 实验 2: LangChain 工业级切分器
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


SAMPLE_DOC = """# useState Hook

useState 是 React 提供的 Hook,用于在函数组件中添加 state(状态)。它返回一个数组,包含两个元素:当前的 state 值,以及更新这个 state 的函数。

## 基础用法

最简单的用法是传入一个初始值:

```javascript
const [count, setCount] = useState(0);
```

这里 count 的初始值是 0,setCount 用来更新 count。

## 更新 State

调用 setCount 时,React 会重新渲染组件:

```javascript
function increment() {
  setCount(count + 1);
}
```

注意:setCount 是异步的,不会立即更新。

## 函数式更新

如果新值依赖于旧值,推荐用函数式更新:

```javascript
setCount(prev => prev + 1);
```

这能避免闭包问题。
"""


# ============================================================
# 1. RecursiveCharacterTextSplitter (最常用)
# ============================================================

print("=" * 80)
print("【LangChain】 RecursiveCharacterTextSplitter")
print("=" * 80)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # 每块最大 300 字
    chunk_overlap=50,      # 重叠 50 字
    separators=["\n\n", "\n", "。", " ", ""],  # 分隔符优先级
)

chunks = splitter.split_text(SAMPLE_DOC)
print(f"切出 {len(chunks)} 块\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} 字符) ---")
    print(chunk)
    print()


# ============================================================
# 2. MarkdownHeaderTextSplitter (针对 Markdown)
# ============================================================

print("\n" + "=" * 80)
print("【LangChain】 MarkdownHeaderTextSplitter")
print("=" * 80)

# 定义要切的标题层级
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # 保留标题内容
)

md_chunks = md_splitter.split_text(SAMPLE_DOC)
print(f"切出 {len(md_chunks)} 块\n")

for i, chunk in enumerate(md_chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"元信息: {chunk.metadata}")
    print(f"内容: {chunk.page_content[:200]}")
    print()


# ============================================================
# 3. 组合用法 (生产级最佳实践)
# ============================================================

print("\n" + "=" * 80)
print("【最佳实践】 Markdown 标题切分 + 递归二次切分")
print("=" * 80)
print("先按标题切大块,大块还太大再用递归切")
print()

# 第一步: 按标题切
md_chunks = md_splitter.split_text(SAMPLE_DOC)

# 第二步: 大块再细切
final_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

final_chunks = []
for chunk in md_chunks:
    sub_chunks = final_splitter.split_text(chunk.page_content)
    for sub in sub_chunks:
        final_chunks.append({
            "content": sub,
            "metadata": chunk.metadata
        })

print(f"最终切出 {len(final_chunks)} 块\n")
for i, chunk in enumerate(final_chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"路径: {chunk['metadata']}")
    print(f"内容: {chunk['content'][:150]}...")
    print()