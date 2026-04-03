import os
import json
import time
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from backend.utils.logger import sys_logger
from backend.tools.evidence_hooks import auto_register

# ==========================================
# 1. 缓存配置
# ==========================================
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/arxiv_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_EXPIRY_SECONDS = 86400


def _get_cache_path(query: str, max_results: int) -> str:
    unique_string = f"{query}_{max_results}".lower()
    cache_key = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{cache_key}.json")


def _format_arxiv_query(raw_query: str) -> str:
    """将大白话翻译为 ArXiv 严格的布尔语法"""
    if ":" in raw_query:
        return raw_query
    clean_words = raw_query.replace('"', '').replace("'", "").split()
    if not clean_words:
        return "all:AI"
    return " AND ".join([f"all:{word}" for word in clean_words])


# ==========================================
# 2. 核心工具 (绕过 buggy 的第三方库，纯原生实现)
# ==========================================
@auto_register("arxiv")
def search_arxiv_literature(query: str, max_results: int = 5) -> str:
    """
    【Agent 工具】检索最新 ArXiv 预印本洞察前沿风向。
    强制使用 HTTPS 直接调用底层 API，完美避开 HTTP 301 重定向 Bug。
    """
    safe_query = _format_arxiv_query(query)
    cache_path = _get_cache_path(safe_query, max_results)

    # --- 拦截器：读缓存 ---
    if os.path.exists(cache_path):
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age < CACHE_EXPIRY_SECONDS:
            sys_logger.debug(f"⚡ 命中 ArXiv 缓存: '{safe_query}'")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()

    sys_logger.info(f"📡 [ArXiv Tool] 正在通过原生 HTTPS 检索: '{safe_query}'")

    try:
        # 1. 强制构造 HTTPS 的安全 URL
        encoded_query = urllib.parse.quote(safe_query)
        url = f"https://export.arxiv.org/api/query?search_query={encoded_query}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

        # 2. 发起原生网络请求 (伪装 User-Agent 防止被盾)
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchCopilot/1.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()

        # 3. 使用 Python 内置的 ElementTree 解析 XML
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        paper_list = []
        for entry in root.findall('atom:entry', ns):
            # 提取标题
            title_node = entry.find('atom:title', ns)
            title = title_node.text.strip().replace('\n', ' ') if title_node is not None else "Unknown Title"

            # 提取摘要
            summary_node = entry.find('atom:summary', ns)
            summary = summary_node.text.strip().replace('\n', ' ') if summary_node is not None else ""

            # 提取发布时间
            published_node = entry.find('atom:published', ns)
            published = published_node.text[:10] if published_node is not None else "Unknown Date"

            # 提取作者组
            authors = []
            for author_node in entry.findall('atom:author', ns):
                name_node = author_node.find('atom:name', ns)
                if name_node is not None and name_node.text:
                    authors.append(name_node.text.strip())

            # 提取 PDF 下载链接
            pdf_url = ""
            for link_node in entry.findall('atom:link', ns):
                if link_node.attrib.get('title') == 'pdf':
                    pdf_url = link_node.attrib.get('href')
                    break

            paper_list.append({
                "title": title,
                "authors": authors,
                "published_date": published,
                "pdf_url": pdf_url,
                "summary": summary
            })

        # --- 结果打包与写缓存 ---
        if not paper_list:
            sys_logger.warning(f"⚠️ 未找到文献: '{safe_query}'")
            result_str = json.dumps({
                "status": "success",
                "message": "未检索到相关预印本，该领域可能暂未出现最新研究。"
            }, ensure_ascii=False)
        else:
            sys_logger.success(f"✅ 成功获取 {len(paper_list)} 篇文献。")
            result_str = json.dumps({
                "status": "success",
                "query_used": safe_query,
                "data": paper_list
            }, ensure_ascii=False, indent=2)

        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(result_str)

        return result_str

    except Exception as e:
        error_msg = f"原生 HTTPS 检索失败: {str(e)}"
        sys_logger.error(f"❌ {error_msg}")
        return json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False)