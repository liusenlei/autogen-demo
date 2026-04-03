import json

import requests

from backend.utils.logger import sys_logger
from backend.tools.evidence_hooks import auto_register


def reconstruct_abstract(inverted_index: dict) -> str:
    """内部辅助函数：将 OpenAlex 的倒排索引还原为完整的摘要文本"""
    if not inverted_index:
        return "No abstract available."

    # 找到最大的索引位置，确定摘要的总词数
    max_index = max([pos for positions in inverted_index.values() for pos in positions])
    words = [""] * (max_index + 1)

    # 将单词填入对应的位置
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word

    return " ".join(words)


@auto_register("openalex")
def search_openalex_graph(query: str, limit: int = 5) -> str:
    """
    【Agent 工具】使用 OpenAlex 学术图谱检索高被引文献与核心概念。

    使用场景：
    当你需要评估某个领域的热度、寻找基石论文，或者需要提取该领域的关联核心概念(Concepts)时使用。

    参数:
        query (str): 检索关键词，必须是英文。
        limit (int, optional): 返回数量，默认 5 篇。

    返回:
        str: 包含高引论文、引用量、摘要以及核心概念标签的 JSON 字符串。
    """
    sys_logger.info(f"⚙️ [Tool Call] OpenAlex 检索 | Query: '{query}'")

    # 构建 OpenAlex API URL
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "sort": "cited_by_count:desc",  # 按总被引次数倒序，寻找最具影响力的基石论文
        "per-page": limit,
        # 选择需要的字段以减小 payload
        "select": "id,title,publication_year,authorships,cited_by_count,abstract_inverted_index,concepts,primary_location"
    }

    headers = {
        "User-Agent": "ResearchCopilotAgent/1.0 (mailto:jamessenleiaicom@gmail.com)"
    }

    try:
        sys_logger.debug("正在请求 OpenAlex API...")
        response = requests.get(base_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            sys_logger.warning(f"OpenAlex 检索无结果 | Query: '{query}'")
            return json.dumps({
                "status": "no_results",
                "message": f"未找到与 '{query}' 相关的高被引论文。"
            }, ensure_ascii=False)

        papers = []
        for item in results:
            # 提取第一作者
            authorships = item.get("authorships", [])
            first_author = authorships[0]["author"]["display_name"] if authorships else "Unknown"

            # 提取高权重概念 (Score > 0.5)
            concepts = [
                c["display_name"] for c in item.get("concepts", [])
                if c.get("score", 0) > 0.5
            ][:3]  # 只取前 3 个最核心的概念

            # 还原摘要
            abstract_text = reconstruct_abstract(item.get("abstract_inverted_index", {}))

            # 提取原文 PDF 链接 (如果有开源版本)
            pdf_url = ""
            location = item.get("primary_location")
            if location and location.get("pdf_url"):
                pdf_url = location.get("pdf_url")

            papers.append({
                "title": item.get("title", "No Title"),
                "year": item.get("publication_year", "Unknown"),
                "first_author": first_author,
                "citation_count": item.get("cited_by_count", 0),
                "core_concepts": concepts,
                "abstract": abstract_text[:1500],
                "pdf_url": pdf_url
            })

        sys_logger.success(f"OpenAlex 检索成功 | 获取到 {len(papers)} 篇高价值文献")

        return json.dumps({
            "status": "success",
            "query_used": query,
            "insights": "已按总被引次数(citation_count)为您降序排列。请利用 core_concepts 字段发现该领域的交叉学科关联。",
            "papers": papers
        }, ensure_ascii=False, indent=2)

    except requests.exceptions.RequestException as e:
        sys_logger.exception(f"OpenAlex 请求失败 | Query: {query}")
        return json.dumps({
            "status": "error",
            "message": f"访问 OpenAlex 图谱时发生错误: {str(e)}"
        }, ensure_ascii=False)
