"""
工具层证据注册钩子

提供装饰器和辅助函数，将工具函数的返回值自动注册到 EvidenceStore。
由于 EvidenceStore 实例在运行时创建，采用模块级全局引用 + 延迟绑定。
"""

from __future__ import annotations

import json
import functools
from typing import Callable, Optional

from backend.utils.logger import sys_logger

_active_store = None


def bind_store(store):
    """绑定当前活跃的 EvidenceStore 实例（由 Pipeline 在启动时调用）。"""
    global _active_store
    _active_store = store
    sys_logger.debug("工具层已绑定 EvidenceStore 实例")


def get_bound_store():
    return _active_store


def auto_register(source_type: str) -> Callable:
    """
    装饰器：在工具函数执行后，自动将结果注册到已绑定的 EvidenceStore。

    仅在 _active_store 已绑定时生效，否则静默跳过。
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            store = _active_store
            if store is None:
                return result
            try:
                store.register_sources_from_tool_result(result, source_type)
            except Exception as e:
                sys_logger.warning(f"自动注册 Source 失败 ({func.__name__}): {e}")
            return result
        return wrapper
    return decorator


def register_tool_result_post_hoc(
    chat_history: list[dict], source_type_map: Optional[dict] = None
):
    """
    从 chat_history 中提取所有工具调用的返回值，
    将其注册到已绑定的 EvidenceStore。

    Args:
        chat_history: AutoGen 聊天记录
        source_type_map: 工具名 → source_type 的映射，例如
            {"search_openalex_graph": "openalex", "search_arxiv_literature": "arxiv"}
    """
    store = _active_store
    if store is None:
        return

    if source_type_map is None:
        source_type_map = {
            "search_openalex_graph": "openalex",
            "search_arxiv_literature": "arxiv",
            "query_paper_rag": "pdf_rag",
        }

    for msg in chat_history:
        content = msg.get("content", "")
        tool_name = msg.get("name", "")

        if tool_name in source_type_map and content:
            try:
                store.register_sources_from_tool_result(
                    content, source_type_map[tool_name]
                )
            except Exception as e:
                sys_logger.debug(f"Post-hoc 注册跳过 ({tool_name}): {e}")
