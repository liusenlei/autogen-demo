import os
import sys
from loguru import logger


def setup_logger():
    """初始化并配置全局 Loguru 日志"""

    # 移除 Loguru 默认的处理器，防止重复输出
    logger.remove()

    # 确立日志存储目录
    log_dir = os.path.join(os.path.dirname(__file__), "../../logs")
    os.makedirs(log_dir, exist_ok=True)

    # 1. 控制台输出 (面向开发者测试：高亮、彩色、INFO级别)
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )

    # 2. 本地文件存档 (面向运维：详细排查、DEBUG级别、日志轮转)
    logger.add(
        f"{log_dir}/system_{{time:YYYY-MM-DD}}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    logger.add(
        f"{log_dir}/audit.json",
        level="INFO",
        serialize=True,
        rotation="50 MB"
    )


# 在模块导入时自动初始化
setup_logger()

# 导出 logger 供其他模块使用
sys_logger = logger