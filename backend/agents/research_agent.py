from __future__ import annotations

from typing import Dict, Any

import autogen

from backend.agents.base_agent import create_assistant_agent
from backend.tools.arxiv_search import search_arxiv_literature
from backend.tools.openalex_search import search_openalex_graph
from backend.tools.pdf_rag import query_paper_rag
from backend.utils.logger import sys_logger

PROMPT_MAP = {
    "landscape": "backend/prompts/researcher_landscape.yaml",
    "deep_dive": "backend/prompts/researcher_deepdive.yaml",
}


def get_researcher(
    llm_config: Dict[str, Any],
    executor: autogen.UserProxyAgent,
    mode: str = "landscape",
) -> autogen.AssistantAgent:
    """
    初始化研究员代理。

    Args:
        llm_config: 大模型配置
        executor: 工具执行代理 (Admin)
        mode: "landscape"（全景扫描）或 "deep_dive"（定向深挖）
    """
    prompt_path = PROMPT_MAP.get(mode, PROMPT_MAP["landscape"])
    sys_logger.info(f"🕵️ 正在组装 Researcher (模式: {mode})...")

    researcher = create_assistant_agent(prompt_path, llm_config)

    autogen.agentchat.register_function(
        search_openalex_graph,
        caller=researcher,
        executor=executor,
        name="search_openalex_graph",
        description="检索高被引基石论文与核心概念（OpenAlex 学术图谱）",
    )

    autogen.agentchat.register_function(
        search_arxiv_literature,
        caller=researcher,
        executor=executor,
        name="search_arxiv_literature",
        description="检索最新 ArXiv 预印本，洞察前沿风向",
    )

    autogen.agentchat.register_function(
        query_paper_rag,
        caller=researcher,
        executor=executor,
        name="query_paper_rag",
        description=(
            "使用 RAG 向量技术向超长 PDF 论文提问。"
            "不要用它获取全文，而是用来查询具体的细节"
            "（如：使用了什么数据集？局限性是什么？）"
        ),
    )

    sys_logger.success(f"🛠️ Researcher [{researcher.name}] ({mode}) 工具已装载。")
    return researcher
