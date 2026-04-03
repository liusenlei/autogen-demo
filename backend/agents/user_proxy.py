import autogen

from backend.agents.base_agent import _logging_hook
from backend.utils.logger import sys_logger


def _is_termination(msg: dict) -> bool:
    """检测终止消息：同时支持 TERMINATE 和 [READY_FOR_HUMAN] 两种终止标记。"""
    content = msg.get("content", "") or ""
    return "TERMINATE" in content or "[READY_FOR_HUMAN]" in content


def get_user_proxy(human_input_mode: str = "NEVER") -> autogen.UserProxyAgent:
    """
    创建人类代理人
    :param human_input_mode:
        "ALWAYS" - 每次发言都等待人类在控制台输入指令（适合调试和深度干预）。
        "NEVER"  - 完全自动流转，只在遇到 TERMINATE 时退出（适合批处理）。
        "TERMINATE" - 自动流转，当其他 Agent 给出最终结论时才让人类确认。
    """
    sys_logger.info(f"👤 初始化 UserProxy (模式: {human_input_mode})")

    proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="你是整个系统的管理员。你负责执行各个 Agent 调用的工具代码，并在最终环节审查课题。",
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=15,
        is_termination_msg=_is_termination,
        code_execution_config={
            "use_docker": False,
            "last_n_messages": 3
        }
    )

    proxy.register_reply(
        [autogen.Agent, None],
        _logging_hook,
        position=1
    )

    return proxy
