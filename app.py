"""
React 文档智能问答系统 - Web 界面
"""

import gradio as gr
import time
from rag import rag
from rag_advanced import advanced_rag, multi_query_retrieve, llm_rerank, generate_queries


# ============================================================
# 基础模式
# ============================================================

def ui_basic_rag(query, top_k, progress=gr.Progress()):
    if not query.strip():
        return "❌ 请输入问题", ""
    
    progress(0.3, desc="🔍 检索文档...")
    start = time.time()
    
    try:
        result = rag(query, top_k=int(top_k), verbose=False)
        elapsed = time.time() - start
        
        progress(1.0, desc="✅ 完成")
        
        # 格式化来源
        sources_md = "### 📚 引用来源\n\n"
        for i, c in enumerate(result["chunks"], 1):
            path_parts = []
            if c.get("h1"): path_parts.append(c["h1"])
            if c.get("h2"): path_parts.append(c["h2"])
            path = " > ".join(path_parts) if path_parts else "(无标题)"
            sources_md += f"**[{i}]** `{c['source']}`\n"
            sources_md += f"- 章节: {path}\n"
            sources_md += f"- 相似度: {c['similarity']:.3f}\n\n"
        
        answer = f"{result['answer']}\n\n---\n⏱️ 耗时: {elapsed:.1f}s"
        return answer, sources_md
    
    except Exception as e:
        return f"❌ 出错: {str(e)}", ""


# ============================================================
# 进阶模式 (Multi-Query + Rerank)
# ============================================================

def ui_advanced_rag(query, progress=gr.Progress()):
    if not query.strip():
        return "❌ 请输入问题", "", ""
    
    start = time.time()
    
    try:
        # 1. Multi-Query
        progress(0.2, desc="🔀 生成多个查询...")
        queries = generate_queries(query, n=4)
        process_md = "### 🔀 Multi-Query 改写\n\n"
        for i, q in enumerate(queries, 1):
            process_md += f"{i}. {q}\n"
        
        # 2. 检索
        progress(0.4, desc="🔍 多路检索...")
        candidates = multi_query_retrieve(query, top_k_per_query=5)
        process_md += f"\n### 📊 召回\n\n合并去重后 **{len(candidates)}** 个候选\n"
        
        # 3. Rerank
        progress(0.7, desc="🎯 重排序精选...")
        top_chunks = llm_rerank(query, candidates, top_n=5)
        process_md += f"\n### 🎯 Rerank 精选\n\n"
        for c in top_chunks:
            process_md += f"- [{c.get('rerank_score', 'N/A')}/10] `{c['source']}`\n"
        
        # 4. 生成
        progress(0.9, desc="🤖 生成回答...")
        from rag_advanced import build_prompt, RAG_SYSTEM_PROMPT
        from openai import OpenAI
        llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        user_prompt = build_prompt(query, top_chunks)
        response = llm_client.chat.completions.create(
            model="qwen2.5:14b",
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        answer = response.choices[0].message.content
        elapsed = time.time() - start
        
        # 来源
        sources_md = "### 📚 引用来源\n\n"
        for i, c in enumerate(top_chunks, 1):
            sources_md += f"**[{i}]** `{c['source']}` (rerank: {c.get('rerank_score', 'N/A')}/10)\n\n"
        
        progress(1.0, desc="✅ 完成")
        answer = f"{answer}\n\n---\n⏱️ 耗时: {elapsed:.1f}s"
        return answer, sources_md, process_md
    
    except Exception as e:
        return f"❌ 出错: {str(e)}", "", ""


# ============================================================
# 界面
# ============================================================

with gr.Blocks(title="React 文档智能问答", css="footer {display: none !important}") as demo:
    
    gr.Markdown("""
    # 📚 React 文档智能问答系统
    
    > 基于 RAG 的 React 官方文档问答助手 · 完全本地运行 · 支持中英文提问
    
    内置 **Multi-Query 多查询改写** + **Rerank 重排序**,检索准确率远超基础 RAG
    """)
    
    with gr.Tab("⚡ 进阶模式 (推荐)"):
        gr.Markdown("**Multi-Query + Rerank** · 准确率高 · 耗时较长 (~5-8s)")
        
        with gr.Row():
            with gr.Column(scale=2):
                adv_input = gr.Textbox(
                    label="你的问题",
                    placeholder="例如: React 推荐用哪种方式管理表单输入?",
                    lines=2
                )
                adv_btn = gr.Button("🚀 提问", variant="primary")
                
                gr.Examples(
                    examples=[
                        "React 推荐用哪种方式管理表单输入?",
                        "useMemo 和 useCallback 的区别是什么?",
                        "如何在 React 中发起网络请求?",
                        "what is useTransition?",
                    ],
                    inputs=adv_input
                )
            
            with gr.Column(scale=3):
                adv_answer = gr.Markdown(label="回答")
        
        with gr.Row():
            with gr.Column():
                adv_sources = gr.Markdown()
            with gr.Column():
                with gr.Accordion("🔬 检索过程详情", open=False):
                    adv_process = gr.Markdown()
        
        adv_btn.click(
            fn=ui_advanced_rag,
            inputs=[adv_input],
            outputs=[adv_answer, adv_sources, adv_process]
        )
    
    with gr.Tab("🏃 基础模式 (快速)"):
        gr.Markdown("**单查询检索** · 速度快 (~2-3s) · 准确率一般")
        
        with gr.Row():
            with gr.Column(scale=2):
                basic_input = gr.Textbox(
                    label="你的问题",
                    placeholder="输入 React 相关问题...",
                    lines=2
                )
                basic_topk = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Top-K (检索数量)"
                )
                basic_btn = gr.Button("🔍 提问", variant="primary")
            
            with gr.Column(scale=3):
                basic_answer = gr.Markdown(label="回答")
        
        basic_sources = gr.Markdown()
        
        basic_btn.click(
            fn=ui_basic_rag,
            inputs=[basic_input, basic_topk],
            outputs=[basic_answer, basic_sources]
        )
    
    gr.Markdown("""
    ---
    ### 🏗️ 技术架构
    
    **两阶段检索**: Embedding 召回 (bge-m3) → LLM Rerank 精排  
    **数据**: React 官方文档 222 个 MD 文件 → 10430 个向量化片段  
    **模型**: bge-m3 (Embedding) + qwen2.5:14b (生成) · 全部本地运行
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
        inbrowser=True,
    )