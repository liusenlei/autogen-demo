from autogen import OpenAIWrapper
from backend.utils.logger import sys_logger


def check_query_specificity(user_input: str, chat_history: list, llm_config: dict) -> str:
    """
    【意图澄清拦截器】评估用户的科研想法是否足够具体。
    结合之前的聊天历史（如果有的话），判断当前意图。
    """
    sys_logger.info("🛡️ Gatekeeper 正在评估用户输入的具体程度...")

    # 初始化 AutoGen 底层的高效客户端
    client = OpenAIWrapper(**llm_config)

    # 构建上下文（把用户之前被追问的历史也带上，方便用户连续回答）
    history_text = ""
    if len(chat_history) > 1:
        history_text = "【之前的对话上下文】\n"
        for msg in chat_history[-3:-1]:  # 取最近的两条上下文
            role = "用户" if msg["role"] == "user" else "系统"
            history_text += f"{role}: {msg['content']}\n"

    system_prompt = f"""
    你是一位顶尖的大学教授，正在听取新生的开题想法。
    你的任务是判断用户的想法是否足够“具体且具备研究价值”。

    【评判标准】
    - ❌ **太模糊 (Vague)**：只有一个大方向或名词。例如：“大模型在医疗的应用”、“图像识别”、“推荐系统”。
    - ✅ **足够具体 (Specific)**：包含特定的场景、痛点或技术结合点。例如：“大模型在医疗问答中的幻觉控制”、“对比学习在长尾推荐系统中的应用”。

    {history_text}

    【输出规则】
    1. 如果想法已经足够具体，**只能**输出精确的字眼：`[SPECIFIC]`（不要带任何其他废话）。
    2. 如果想法太模糊，请以引导者的语气，输出 `[VAGUE]`，换行，然后给出 2-3 个启发式的追问，帮助用户缩小范围。（例如：你想关注医疗的哪个环节？是诊断还是病历生成？你倾向于解决精度问题还是推理速度问题？）
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"我的想法是：{user_input}"}
    ]

    try:
        response = client.create(messages=messages)
        result = response.choices[0].message.content.strip()
        sys_logger.debug(f"🛡️ Gatekeeper 判定结果: {result[:50]}...")
        return result
    except Exception as e:
        sys_logger.error(f"Gatekeeper 评估失败: {e}")
        # 如果判断层崩溃，安全起见直接放行
        return "[SPECIFIC]"