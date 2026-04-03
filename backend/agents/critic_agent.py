from __future__ import annotations

from typing import Dict, Any

import autogen

from backend.agents.base_agent import create_assistant_agent
from backend.tools.arxiv_search import search_arxiv_literature
from backend.utils.logger import sys_logger


def get_synthesizer(llm_config: Dict[str, Any]) -> autogen.AssistantAgent:
    """初始化课题合成器，纯逻辑推演节点，不需要外部工具。"""
    sys_logger.info("🧠 正在组装 Synthesizer 实例...")
    synthesizer = create_assistant_agent("backend/prompts/synthesizer.yaml", llm_config)
    sys_logger.success(f"✨ Synthesizer [{synthesizer.name}] 准备就绪。")
    return synthesizer


def get_critic(llm_config: Dict[str, Any]) -> autogen.AssistantAgent:
    """
    初始化评审员 (Critic)。
    使用 register_for_llm + register_for_execution 双注册模式，
    确保 Critic 既能提议调用工具，也能自己执行。
    """
    sys_logger.info("🛡️ 正在组装 Critic 实例...")
    critic = create_assistant_agent("backend/prompts/critic.yaml", llm_config)

    critic.register_for_llm(
        name="search_arxiv_literature",
        description="检索 ArXiv 以验证候选选题是否已被抢发（撞车检测）",
    )(search_arxiv_literature)

    sys_logger.success(f"⚔️ Critic [{critic.name}] 已武装撞车检测工具。")
    return critic
