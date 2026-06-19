# 📚 React Docs RAG — React 文档智能问答系统

> 基于 RAG (检索增强生成) 的 React 官方文档问答助手。完全本地运行,支持中英文提问,内置 Multi-Query 与 Rerank 两阶段检索优化。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Chroma](https://img.shields.io/badge/ChromaDB-VectorStore-green)
![Ollama](https://img.shields.io/badge/Ollama-Local-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ 项目亮点

- 🔍 **语义搜索**: 理解问题含义而非关键词匹配,支持中文提问检索英文文档
- 🔀 **Multi-Query 改写**: 自动从多角度改写问题,大幅提升检索召回率
- 🎯 **Rerank 重排序**: 两阶段检索 (召回 → 精排),显著提升准确率
- 📚 **引用溯源**: 每个回答标注来源文档与章节,可验证不瞎编
- 🏠 **完全本地**: bge-m3 + qwen2.5:14b 全本地运行,数据不出本机

---

## 🎯 解决的核心问题

LLM 不知道你的私有/最新数据,且容易"幻觉"。本项目通过 RAG 让 LLM:
- 基于真实 React 文档回答,而非训练记忆
- 对文档外的问题诚实回答"我没找到相关信息"
- 每个事实都能追溯到具体文档来源

---

## 🏗️ 技术架构

\`\`\`
┌─────────────────── 离线建库 ───────────────────┐
│  222 个 MD 文件                                  │
│      ↓ MarkdownHeaderSplitter + Recursive       │
│  10430 个文档片段 (清洗后)                       │
│      ↓ bge-m3 向量化 (1024 维)                   │
│  ChromaDB 向量数据库 (cosine, HNSW 索引)         │
└─────────────────────────────────────────────────┘

┌─────────────────── 在线查询 ───────────────────┐
│  用户问题                                        │
│      ↓ Multi-Query (LLM 改写成 4+ 个查询)        │
│  多路 Embedding 召回 → 合并去重 (~15-20 候选)    │
│      ↓ LLM Rerank (相关性打分精排)               │
│  Top-5 精选片段                                  │
│      ↓ 拼装 Prompt + 防幻觉约束                  │
│  qwen2.5:14b 生成带引用的回答                     │
└─────────────────────────────────────────────────┘
\`\`\`

### 两阶段检索

| 阶段 | 处理量     | 方法            | 目标        |
| ---- | ---------- | --------------- | ----------- |
| 召回 | 全库 10430 | Embedding (快)  | 高召回,不漏 |
| 精排 | 召回的 ~20 | LLM Rerank (准) | 高精度,选优 |

---

## 🚀 快速开始

### 前置要求

- Python 3.10+
- [Ollama](https://ollama.com)
- 模型: \`ollama pull bge-m3\` 和 \`ollama pull qwen2.5:14b\`

### 安装

\`\`\`bash
git clone https://github.com/你的用户名/react-docs-rag.git
cd react-docs-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 构建向量库

\`\`\`bash
# 1. 准备数据 (React 文档)
git clone --depth 1 https://github.com/reactjs/react.dev.git data/raw/react.dev

# 2. 切分文档
python process_documents.py

# 3. 构建向量库 (约 15 分钟)
python build_vector_db.py
\`\`\`

### 运行

\`\`\`bash
python app.py   # Web 界面
# 或
python rag_advanced.py   # 命令行测试
\`\`\`

---

## 📂 项目结构

\`\`\`
react-docs-rag/
├── app.py                  # Web 界面 (Gradio)
├── rag.py                  # 基础 RAG (单查询)
├── rag_advanced.py         # 进阶 RAG (Multi-Query + Rerank)
├── process_documents.py    # 文档切分
├── build_vector_db.py      # 向量库构建
├── search.py               # 语义搜索测试
├── experiments/            # 学习实验脚本
├── requirements.txt
└── README.md
\`\`\`

---

## 📊 效果对比

以"React 推荐用哪种方式管理表单输入?"为例:

**基础 RAG**: 检索到 3 个不相关文档,回答跑偏到 ref 操作  
**进阶 RAG**: Multi-Query 找回 \`reacting-to-input-with-state.md\` 等正确文档,回答准确

Multi-Query + Rerank 显著改善了"问题措辞与文档不匹配"导致的检索失败。

---

## 🛠️ 技术栈

- **ChromaDB** — 向量数据库 (HNSW 索引, cosine 距离)
- **bge-m3** — 多语言 Embedding 模型 (中英双语,1024 维)
- **qwen2.5:14b** — 生成模型 + 查询改写 + Rerank
- **LangChain** — 文档切分 (RecursiveCharacterTextSplitter)
- **Gradio** — Web 界面
- **Ollama** — 本地模型推理

---

## 📝 学到的核心知识

- ✅ RAG 完整流程: 解析 → 切分 → 向量化 → 检索 → 增强 → 生成
- ✅ Embedding 与语义空间,余弦相似度,跨语言对齐
- ✅ 文档切分策略与 chunk_size 权衡
- ✅ 向量数据库原理 (HNSW, 召回-精排两阶段架构)
- ✅ Multi-Query 用多样性对抗检索不确定性
- ✅ Rerank 重排序与 LLM-as-Reranker 的局限
- ✅ 防幻觉 Prompt 设计与引用溯源

---

## 🔮 未来改进

- [ ] 用专用 reranker (bge-reranker-v2) 替代 LLM Rerank,提速并提升区分度
- [ ] 批量 Embedding 加速入库
- [ ] 混合检索 (向量 + 关键词 BM25)
- [ ] 多轮对话 + 查询改写 (指代消解)
- [ ] 支持上传自定义文档

---

## 📝 License

MIT

---

## 🙋 关于作者

前端开发者转型 AI 应用工程师,这是我的第三个 AI 项目。

**项目 1**: AI 会议纪要生成器  
**项目 2**: AI 文本工具箱 (Prompt 工程 + 评估系统)  
**项目 3**: 当前项目 (RAG 检索增强)

欢迎 ⭐ Star!