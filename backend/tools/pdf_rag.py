import os
import hashlib
import requests
import tempfile
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.utils.logger import sys_logger
from backend.tools.evidence_hooks import auto_register


# 1. 初始化本地持久化向量数据库
# 数据会保存在项目根目录的 data/vector_db 下，关机不丢失
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/vector_db")
os.makedirs(DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=DB_PATH)

def _get_collection_name(pdf_url: str) -> str:
    """用 URL 的 MD5 哈希作为数据库集合(Collection)的名称，确保唯一且合法"""
    return "pdf_" + hashlib.md5(pdf_url.encode('utf-8')).hexdigest()

@auto_register("pdf_rag")
def query_paper_rag(pdf_url: str, query: str, top_k: int = 3) -> str:
    """
    【Agent 工具】使用 RAG 向量技术，针对长篇 PDF 论文提出具体问题并获取精准段落。
    参数:
        pdf_url (str): PDF 文件的在线下载链接。
        query (str): 你想向这篇论文提出的具体问题。例如："What are the limitations mentioned in the conclusion?" 或 "What dataset was used?"
        top_k (int): 检索返回的最相关段落数量，默认 3。
    返回:
        str: 论文中最相关的原文段落摘录。
    """

    sys_logger.info(f"📚 [RAG Tool] 检索论文 | 目标: {pdf_url} | 问题: '{query}'")
    collection_name = _get_collection_name(pdf_url)

    try:
        # 2. 检查数据库中是否已经有这篇论文了 (命中缓存)
        try:
            collection = chroma_client.get_collection(name=collection_name)
            sys_logger.debug(f"⚡ 命中本地向量数据库缓存，直接进行语义检索...")
        except ValueError:
            # 3. 缓存未命中，开始下载并向量化新论文
            sys_logger.info(f"📥 未找到该论文的向量缓存，正在下载并构建 RAG 索引...")

            # 下载 PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_pdf_path = tmp_file.name

            # 解析 PDF 文本
            reader = PdfReader(tmp_pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text: full_text += text + "\n"

            os.remove(tmp_pdf_path)

            if not full_text.strip():
                return "错误：未能从该 PDF 链接中提取到有效文本（可能是扫描版或加了密的 PDF）。"

            # 4. 文本切块 (Text Chunking)
            # 使用 LangChain 的切分器，按段落和句子切分，保持语义完整
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(full_text)

            # 5. 存入 ChromaDB (自动调用默认模型进行向量化 Embeddings)
            collection = chroma_client.create_collection(name=collection_name)

            # 构建 ChromaDB 需要的数据结构
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_url, "chunk_index": i} for i in range(len(chunks))]

            sys_logger.debug(f"🧠 正在将 {len(chunks)} 个文本块向量化并写入数据库 (首次运行需下载轻量级模型)...")
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            sys_logger.success(f"✅ RAG 索引构建完成！")

        # 6. 执行向量检索 (Vector Search)
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # 7. 组装返回给 Agent 的结果
        retrieved_docs = results['documents'][0]
        if not retrieved_docs:
            return "未能检索到与您的问题相关的段落。"

        formatted_result = f"针对您的问题 '{query}'，从该论文中检索到以下最相关的段落：\n\n"
        for i, doc in enumerate(retrieved_docs):
            formatted_result += f"--- 相关段落 {i + 1} ---\n{doc}\n\n"

        return formatted_result

    except Exception as e:
        sys_logger.error(f"RAG 检索失败: {str(e)}")
        return f"读取或检索 PDF 时发生错误: {str(e)}"