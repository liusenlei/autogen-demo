import json
import re

import fitz
import requests

from backend.utils.logger import sys_logger


def extract_pdf_content(pdf_url: str, max_pages: int = 3) -> str:
    """
    【Agent 工具】读取并提取在线 PDF 论文的纯文本内容。

    使用场景：
    当你通过 ArXiv 检索发现某篇论文的 Abstract（摘要）不足以评估其具体的方法论细节、
    实验数据集或具体的 Research Gap 时，使用此工具读取该论文的 PDF 原文。

    参数:
        pdf_url (str): PDF 文件的完整下载链接 (如 "https://arxiv.org/pdf/2401.12345").
        max_pages (int, optional): 最多读取的页数。默认 3 页（通常覆盖 Introduction 和 Related Work）。
                                   【极其重要】为了防止超出你的阅读记忆上限(Token限制)，禁止一次性读取超过 5 页。

    返回:
        str: 包含论文提取文本的 JSON 格式字符串。
    """
    sys_logger.info(f"⚙️ [Tool Call] 解析 PDF | URL: {pdf_url}, 最大页数: {max_pages}")

    # 修复 ArXiv 链接：大模型有时会传入网页版链接(abs)，需自动替换为 pdf 链接
    if "arxiv.org/abs/" in pdf_url:
        pdf_url = pdf_url.replace("/abs/", "/pdf/")
        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"

    try:
        # 1. 发起请求并下载 PDF 数据到内存 (避免磁盘 I/O，提升 Agent 响应速度)
        sys_logger.debug("正在下载 PDF 文件流...")
        response = requests.get(pdf_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        response.raise_for_status()  # 检查 HTTP 错误

        pdf_bytes = response.content

    except requests.exceptions.RequestException as e:
        sys_logger.error(f"PDF 下载失败 | URL: {pdf_url} | 错误: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"无法下载该 PDF 文件，网络错误: {str(e)}。"
        }, ensure_ascii=False)

    try:
        # 2. 使用 PyMuPDF 在内存中打开文档
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # 获取实际页数与请求页数的最小值
        actual_pages = len(doc)
        pages_to_read = min(actual_pages, max_pages)

        extracted_text = ""

        # 3. 遍历并提取文本
        for page_num in range(pages_to_read):
            page = doc[page_num]
            # 采用 "blocks" 模式提取，它能较好地按物理块（段落）读取，天然缓解双栏混排问题
            blocks = page.get_text("blocks")

            # 按垂直位置 (y0) 简单排序，保证阅读顺序基本自上而下
            blocks.sort(key=lambda b: (b[1], b[0]))

            for block in blocks:
                # 过滤掉非文本块 (如图片、无意义的空行)
                text = block[4].strip()
                if text:
                    extracted_text += text + "\n\n"

        doc.close()

        # 4. 文本清洗工程 (Data Cleaning)
        # 修复断字换行 (例如 "hal-\nlucination" -> "hallucination")
        extracted_text = re.sub(r'-\n\s*', '', extracted_text)
        # 将多个连续换行替换为单个换行，压缩 Token
        extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text)

        sys_logger.success(f"PDF 解析成功 | URL: {pdf_url}, 共提取 {pages_to_read} 页文本。")

        return json.dumps({
            "status": "success",
            "source_url": pdf_url,
            "pages_read": f"1 to {pages_to_read} (Total pages: {actual_pages})",
            "content": extracted_text[:15000]
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        sys_logger.exception(f"PDF 文本解析失败 | URL: {pdf_url}")
        return json.dumps({
            "status": "error",
            "message": f"解析 PDF 文件结构时发生错误: {str(e)}。可能该 PDF 是扫描版或已加密。"
        }, ensure_ascii=False)
