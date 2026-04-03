from typing import Any, Tuple

import autogen

from backend.utils.file_io import read_yaml
from backend.utils.logger import sys_logger


def _logging_hook(
        recipient: autogen.ConversableAgent,
        messages: list,
        sender: autogen.ConversableAgent,
        config: Any
) -> Tuple[bool, None]:
    """
    内部拦截器：自动捕获并记录 Agent 之间的每一次消息传递。
    """

    if not messages:
        return False, None

    last_msg = messages[-1]
    content = last_msg.get("content", "")

    if "tool_calls" in last_msg:
        sys_logger.info(f"🤖 [{sender.name} ➡️ {recipient.name}] 发起了工具调用请求...")
    elif content:
        snippet = content.replace('\n', ' ')[:100]
        suffix = "..." if len(content) > 100 else ""
        sys_logger.info(f"💬 [{sender.name} ➡️ {recipient.name}] {snippet}{suffix}")

    return False, None


def create_assistant_agent(yaml_path: str, llm_config: dict) -> autogen.AssistantAgent:
    """
    Agent 工厂函数：读取 YAML 配置，创建并返回注入了日志追踪的 AssistantAgent
    :param yaml_path: YAML 配置
    :param llm_config: 模型配置
    :return: 注入了日志追踪的 AssistantAgent
    """

    prompt_data = read_yaml(yaml_path)

    agent = autogen.AssistantAgent(
        name=prompt_data.get("name", "Unknown_Agent"),
        system_message=prompt_data.get("system_message", ""),
        description=prompt_data.get("description", ""),
        llm_config=llm_config,
    )

    agent.register_reply(
        [autogen.Agent, None],
        reply_func=_logging_hook,
        config={"callback": None},
        ignore_async_in_sync_chat=True,
        position=1
    )

    sys_logger.debug(f"✅ 成功实例化 Agent: {agent.name}")
    return agent
