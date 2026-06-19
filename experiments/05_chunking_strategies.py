"""
Day 3 实验 1: 4 种切分策略对比
"""

import re
from pathlib import Path


# 测试文档(模拟一篇 React 文档)
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

## 注意事项

useState 必须在组件顶层调用,不能放在循环、条件或嵌套函数中。这是因为 React 依赖 Hook 的调用顺序来追踪 state。
"""


# ============================================================
# 策略 1: 固定长度切分
# ============================================================

def chunk_by_size(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    """按字数切,可选重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # 重叠处理
    return chunks


# ============================================================
# 策略 2: 按段落切
# ============================================================

def chunk_by_paragraph(text: str) -> list[str]:
    """按双换行切段落"""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


# ============================================================
# 策略 3: 递归切分
# ============================================================

def chunk_recursive(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    递归切分: 按 \n\n -> \n -> 句号 -> 字数 层层切
    这是 LangChain RecursiveCharacterTextSplitter 的简化版
    """
    separators = ["\n\n", "\n", "。", "!", "?", ".", " ", ""]
    
    def _split(text, separators):
        # 如果短于 chunk_size,直接返回
        if len(text) <= chunk_size:
            return [text]
        
        # 找第一个能用的分隔符
        for sep in separators:
            if sep == "":
                # 最后兜底:硬切
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            if sep in text:
                # 用这个分隔符切
                parts = text.split(sep)
                
                # 重新组合,使每段尽量接近 chunk_size
                chunks = []
                current = ""
                for part in parts:
                    if len(current) + len(part) + len(sep) <= chunk_size:
                        current = current + sep + part if current else part
                    else:
                        if current:
                            chunks.append(current)
                        # 如果单个 part 还太大,递归切
                        if len(part) > chunk_size:
                            chunks.extend(_split(part, separators[separators.index(sep)+1:]))
                            current = ""
                        else:
                            current = part
                
                if current:
                    chunks.append(current)
                
                return chunks
        
        return [text]
    
    chunks = _split(text, separators)
    
    # 加 overlap (相邻 chunk 重叠)
    if overlap > 0 and len(chunks) > 1:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i-1][-overlap:]
            result.append(prev_tail + chunks[i])
        return result
    
    return chunks


# ============================================================
# 策略 4: Markdown 结构化切分
# ============================================================

def chunk_by_markdown(text: str) -> list[dict]:
    """
    按 Markdown 标题切,每段返回 dict 包含层级信息
    """
    chunks = []
    current_chunk = {"title": "", "level": 0, "content": []}
    current_title_path = []  # 标题路径,如 ["useState", "基础用法"]
    
    for line in text.split("\n"):
        # 匹配标题
        title_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        
        if title_match:
            # 保存当前 chunk
            if current_chunk["content"]:
                content = "\n".join(current_chunk["content"]).strip()
                if content:
                    chunks.append({
                        "title": " > ".join(current_title_path),
                        "level": current_chunk["level"],
                        "content": content
                    })
            
            # 开始新 chunk
            level = len(title_match.group(1))
            title = title_match.group(2)
            
            # 更新标题路径
            current_title_path = current_title_path[:level-1] + [title]
            
            current_chunk = {
                "title": title,
                "level": level,
                "content": [line]  # 包含标题本身
            }
        else:
            current_chunk["content"].append(line)
    
    # 保存最后一个
    if current_chunk["content"]:
        content = "\n".join(current_chunk["content"]).strip()
        if content:
            chunks.append({
                "title": " > ".join(current_title_path),
                "level": current_chunk["level"],
                "content": content
            })
    
    return chunks


# ============================================================
# 对比 4 种策略
# ============================================================

print("=" * 80)
print("📋 测试文档")
print("=" * 80)
print(f"原文长度: {len(SAMPLE_DOC)} 字符")
print()

# 策略 1
print("=" * 80)
print("【策略 1】 固定长度切分 (chunk_size=200)")
print("=" * 80)
chunks_1 = chunk_by_size(SAMPLE_DOC, chunk_size=200)
print(f"切出 {len(chunks_1)} 块")
for i, c in enumerate(chunks_1):
    print(f"\n--- Chunk {i+1} ({len(c)} 字符) ---")
    print(c[:150] + "..." if len(c) > 150 else c)

# 策略 2
print("\n" + "=" * 80)
print("【策略 2】 按段落切")
print("=" * 80)
chunks_2 = chunk_by_paragraph(SAMPLE_DOC)
print(f"切出 {len(chunks_2)} 块")
for i, c in enumerate(chunks_2):
    print(f"\n--- Chunk {i+1} ({len(c)} 字符) ---")
    print(c[:150] + "..." if len(c) > 150 else c)

# 策略 3
print("\n" + "=" * 80)
print("【策略 3】 递归切分 (chunk_size=300)")
print("=" * 80)
chunks_3 = chunk_recursive(SAMPLE_DOC, chunk_size=300, overlap=0)
print(f"切出 {len(chunks_3)} 块")
for i, c in enumerate(chunks_3):
    print(f"\n--- Chunk {i+1} ({len(c)} 字符) ---")
    print(c[:150] + "..." if len(c) > 150 else c)

# 策略 4
print("\n" + "=" * 80)
print("【策略 4】 Markdown 结构化切分")
print("=" * 80)
chunks_4 = chunk_by_markdown(SAMPLE_DOC)
print(f"切出 {len(chunks_4)} 块")
for i, c in enumerate(chunks_4):
    print(f"\n--- Chunk {i+1} ---")
    print(f"标题路径: {c['title']}")
    print(f"层级: H{c['level']}")
    print(f"内容长度: {len(c['content'])} 字符")
    print(f"内容预览: {c['content'][:150]}...")

print("\n" + "=" * 80)
print("💡 观察:")
print("  策略 1: 简单但可能切断语义")
print("  策略 2: 保留段落但有的太短")
print("  策略 3: 平衡(LangChain 的默认策略)")
print("  策略 4: 适合 Markdown,带元信息")
print("=" * 80)