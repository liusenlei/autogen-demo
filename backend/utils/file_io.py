import json
import os
from typing import Union, Dict, Any, List

import yaml

from backend.utils.logger import sys_logger


def get_project_root() -> str:
    """
    获取项目根目录的绝对路径。
    假设当前文件位于 src/utils/file_io.py，则向上退两级到达根目录。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))


def resolve_path(relative_path: str) -> str:
    """
    将基于项目根目录的相对路径转换为绝对路径。
    例如: resolve_path("src/prompts/researcher.yaml")
    """
    # 如果已经是绝对路径，则直接返回
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(get_project_root(), relative_path)


def ensure_dir(file_path: str):
    """确保目标文件所在的目录存在，如果不存在则自动创建"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        sys_logger.debug(f"📁 自动创建目录: {directory}")


# ==========================================
# YAML 读写 (主要用于 Prompt 和 系统配置)
# ==========================================
def read_yaml(file_path: str) -> Dict[str, Any]:
    """读取 YAML 文件并返回字典"""
    abs_path = resolve_path(file_path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            sys_logger.debug(f"📄 成功读取 YAML: {abs_path}")
            return data or {}
    except FileNotFoundError:
        sys_logger.error(f"❌ 找不到 YAML 文件: {abs_path}")
        raise
    except yaml.YAMLError as e:
        sys_logger.error(f"❌ 解析 YAML 文件失败: {abs_path} | 错误: {e}")
        raise


# ==========================================
# JSON 读写 (主要用于 Agent 状态缓存、工具结果落盘)
# ==========================================
def read_json(file_path: str) -> Union[Dict, List, None]:
    """读取 JSON 文件"""
    abs_path = resolve_path(file_path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        sys_logger.warning(f"⚠️ 找不到 JSON 文件 (返回 None): {abs_path}")
        return None
    except json.JSONDecodeError as e:
        sys_logger.error(f"❌ 解析 JSON 文件失败: {abs_path} | 错误: {e}")
        raise


def write_json(data: Union[Dict, List], file_path: str, indent: int = 2):
    """将数据写入 JSON 文件"""
    abs_path = resolve_path(file_path)
    ensure_dir(abs_path)
    try:
        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        sys_logger.debug(f"💾 成功写入 JSON: {abs_path}")
    except Exception as e:
        sys_logger.error(f"❌ 写入 JSON 文件失败: {abs_path} | 错误: {e}")
        raise


# ==========================================
# 纯文本读写 (主要用于生成最终的 Markdown 报告)
# ==========================================
def read_text(file_path: str) -> str:
    """读取纯文本文件"""
    abs_path = resolve_path(file_path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        sys_logger.error(f"❌ 找不到文本文件: {abs_path}")
        raise


def write_text(content: str, file_path: str):
    """将字符串写入文本文件"""
    abs_path = resolve_path(file_path)
    ensure_dir(abs_path)
    try:
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        sys_logger.info(f"📝 成功生成文本文件: {abs_path}")
    except Exception as e:
        sys_logger.error(f"❌ 写入文本文件失败: {abs_path} | 错误: {e}")
        raise
