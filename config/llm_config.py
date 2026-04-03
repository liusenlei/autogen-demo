import os

from dotenv import load_dotenv

from backend.utils.logger import sys_logger

load_dotenv()


def _get_api_key(provider: str = "OPENAI") -> str:
    """内部辅助函数：安全地获取 API Key"""
    env_var_name = f"{provider}_API_KEY"
    api_key = os.environ.get(env_var_name)

    if not api_key:
        error_msg = f"❌ 严重错误：未在环境变量或 .env 文件中找到 {env_var_name}！"
        sys_logger.critical(error_msg)
        raise ValueError(error_msg)

    return api_key


def _get_base_url() -> str:
    """获取 API base_url，优先读 OPENAI_BASE_URL 环境变量。"""
    return os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def get_gpt5_config(temperature: float = 0.1, cache_seed: int = 42) -> dict:
    """
    获取顶配推理模型配置 (适用于复杂逻辑推演，如 Researcher 和 Critic)

    参数:
        temperature: 温度越低越严谨，学术科研场景建议保持在 0.1 - 0.3 之间。
        cache_seed: AutoGen 特有的缓存种子。相同 seed 下的重复请求直接读本地缓存，不仅秒回，而且**不扣费**！开发调试时极其有用。设置为 None 可关闭缓存。
    """
    sys_logger.debug(f"加载 LLM 配置: gpt-5.4-pro | Temp: {temperature}")
    return {
        "config_list": [
            {
                "model": "qwen/qwen3.6-plus-preview:free",
                "api_key": _get_api_key("OPENAI"),
                "base_url": _get_base_url(),
            }
        ],
        "temperature": temperature,
        "cache_seed": cache_seed,
        "timeout": 120,
    }


def get_cheap_config(temperature: float = 0.3, cache_seed: int = 42) -> dict:
    """
    获取经济型模型配置 (适用于日常代码调试、简单的文本格式化)
    """
    sys_logger.debug(f"加载 LLM 配置: gpt-5.4-mini | Temp: {temperature}")
    return {
        "config_list": [
            {
                "model": "qwen/qwen3.6-plus-preview:free",
                "api_key": _get_api_key("OPENAI"),
                "base_url": _get_base_url(),
            }
        ],
        "temperature": temperature,
        "cache_seed": cache_seed,
        "timeout": 60,
    }


def get_local_vllm_config(temperature: float = 0.1) -> dict:
    """
    示例：如果你在本地部署了 vLLM / Ollama (例如 Llama-3 或 DeepSeek)
    """
    sys_logger.debug(f"加载 LLM 配置: 本地开源模型 | Temp: {temperature}")
    return {
        "config_list": [
            {
                "model": "llama-3-70b-instruct",
                "api_key": "EMPTY",
                "base_url": "http://localhost:8000/v1",
            }
        ],
        "temperature": temperature,
        "cache_seed": 42,
    }
