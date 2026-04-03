# AI 科研选题副驾 (Research Topic Copilot)

帮助研究者从一个模糊的种子想法出发，经过**分阶段文献调研**，最终收敛到有据可依的选题方向。

## 目标用户

- **硕博新生**：正在寻找开题方向，需要快速摸底一个陌生领域的全景。
- **资深研究员 / PI**：寻找跨学科交叉灵感，或快速评估某个新方向的可行性。
- **企业 R&D 人员**：需要跟进前沿技术并判断哪些方向值得投入。

## 核心能力

| 能力 | 说明 |
|------|------|
| 意图澄清 | Gatekeeper 评估种子想法是否足够具体，模糊则引导追问 |
| 全景扫描 | 同时检索 OpenAlex 高被引基石论文 + ArXiv 最新预印本，构建领域地图 |
| 方向深挖 | 针对用户选定的方向，通过 PDF RAG 穿透论文原文，提取精确 Gap |
| 候选生成 | Synthesizer 基于证据链生成 2-4 个候选选题，附带四维度评分 |
| 撞车检测 | Critic 亲自调用 ArXiv 检索验证选题是否已被抢发 |
| 证据溯源 | 每条断言关联到具体的 Source ID，前端可逐条验证 |

**不做的事**：不生成完整实验设计、不写文献综述、不做论文写作辅助、不保证引用论文 100% 存在（但提供验证机制）。

---

## 系统架构

```
┌──────────────────────────────────────────────────┐
│                表现层 (Streamlit UI)               │
│  多步骤向导 / 证据面板 / 比较矩阵 / 健康度报告     │
├──────────────────────────────────────────────────┤
│                编排层 (PhasedPipeline)             │
│  Phase 0→1→2→3→4 / HumanGate / MetricsCollector  │
├──────────────────────────────────────────────────┤
│                智能层 (AutoGen Agents)             │
│  Researcher(双模式) / Synthesizer / Critic         │
├──────────────────────────────────────────────────┤
│                证据层 (EvidenceStore)              │
│  Source 注册 / Claim 管理 / Candidate 管理 / 快照   │
├──────────────────────────────────────────────────┤
│                工具层 (Tools + Hooks)              │
│  OpenAlex / ArXiv / PDF RAG / @auto_register      │
└──────────────────────────────────────────────────┘
```

### 工作流程

```
用户输入种子想法
      │
      ▼
┌─ Phase 0: 意图澄清 ─────────────────────┐
│  Gatekeeper 判断是否足够具体              │
│  模糊 → 追问引导 → 用户补充 → 再判断      │
└──────────────────────────────────────────┘
      │ 明确
      ▼
┌─ Phase 1: 全景扫描 ─────────────────────┐
│  Researcher (全景模式)                    │
│  → OpenAlex 高被引基石论文               │
│  → ArXiv 前沿预印本                      │
│  → 输出: 领域全景图 + 初步 Gap 列表       │
└──────────────────────────────────────────┘
      │
      ▼
  ╔══ HumanGate 1 ═══════════════════════╗
  ║  用户从 Gap 列表中选择 2-3 个方向      ║
  ║  可补充自定义方向                       ║
  ╚════════════════════════════════════════╝
      │
      ▼
┌─ Phase 2: 方向深挖 ─────────────────────┐
│  Researcher (深挖模式)                    │
│  → PDF RAG 穿透论文原文                  │
│  → 精确 Gap / SOTA / 可用数据集           │
│  → 竞争风险评估                           │
└──────────────────────────────────────────┘
      │
      ▼
┌─ Phase 3: 候选生成与评审 ───────────────┐
│  Synthesizer → 生成 2-4 个候选选题       │
│  Critic → 撞车检测 + 可行性 + 价值判断    │
│  → 输出: 多维度比较矩阵                   │
└──────────────────────────────────────────┘
      │
      ▼
  ╔══ HumanGate 2 ═══════════════════════╗
  ║  用户评审候选 / 提出修改意见            ║
  ║  可要求重新生成                         ║
  ╚════════════════════════════════════════╝
      │
      ▼
┌─ Phase 4: 收敛报告 ─────────────────────┐
│  生成最终选题决议报告                     │
│  附带完整证据溯源 + L1 健康度报告          │
└──────────────────────────────────────────┘
```

---

## 项目结构

```
autogen-demo/
├── app.py                              # Streamlit 前端入口（多步骤向导 UI）
├── pyproject.toml                      # 项目依赖声明（uv 管理）
├── config/
│   └── llm_config.py                   # LLM 模型配置（API Key、温度、缓存）
│
├── backend/
│   ├── agents/                         # Agent 定义层
│   │   ├── base_agent.py               #   YAML 工厂 + 消息日志拦截钩子
│   │   ├── research_agent.py           #   Researcher（支持 landscape / deep_dive 双模式）
│   │   ├── critic_agent.py             #   Synthesizer + Critic 工厂
│   │   └── user_proxy.py              #   Admin UserProxy（工具执行代理）
│   │
│   ├── evidence/                       # 证据管理层（核心新增）
│   │   ├── models.py                   #   Source / Claim / Candidate 数据模型
│   │   ├── store.py                    #   EvidenceStore 证据注册中枢
│   │   ├── comparison.py               #   多维度比较矩阵构建 + Markdown 渲染
│   │   └── metrics.py                  #   L1 自动化评估指标收集器
│   │
│   ├── prompts/                        # Agent 提示词（YAML 配置）
│   │   ├── researcher_landscape.yaml   #   Researcher 全景扫描模式
│   │   ├── researcher_deepdive.yaml    #   Researcher 定向深挖模式
│   │   ├── synthesizer.yaml            #   Synthesizer（结构化 JSON 输出）
│   │   └── critic.yaml                #   Critic（结构化评审结论）
│   │
│   ├── tools/                          # 外部工具层
│   │   ├── openalex_search.py          #   OpenAlex 学术图谱检索（Tier 1）
│   │   ├── arxiv_search.py             #   ArXiv 预印本检索（Tier 2）
│   │   ├── pdf_rag.py                  #   PDF RAG 向量检索（Tier 3，ChromaDB）
│   │   ├── pdf_parser.py               #   PDF 文本提取（PyMuPDF）
│   │   └── evidence_hooks.py           #   @auto_register 装饰器 + post-hoc 注册
│   │
│   ├── workflows/                      # 工作流编排层
│   │   ├── gatekeeper.py               #   Phase 0 意图澄清拦截器
│   │   └── topic_generation.py         #   PhasedPipeline 四阶段编排引擎
│   │
│   └── utils/
│       ├── logger.py                   #   Loguru 日志（控制台 + 文件 + JSON 审计）
│       └── file_io.py                  #   YAML/JSON/文本 读写工具
│
├── data/
│   ├── arxiv_cache/                    # ArXiv 查询 JSON 缓存（24h 过期）
│   └── vector_db/                      # ChromaDB 持久化向量数据库
│
└── logs/
    ├── system_YYYY-MM-DD.log           # 按日滚动系统日志
    └── audit.json                      # JSON 序列化审计日志
```

---

## 模块详解

### 证据管理层 (`backend/evidence/`)

整个系统的可信度核心。所有从工具层检索到的文献数据，都必须先注册到 EvidenceStore，再被 Agent 引用。

**Source**：一条可溯源的文献来源，包含 `source_id`、类型（openalex/arxiv/pdf_rag）、标题、作者、URL 等。每条 Source 按数据源类型分配信任权重（OpenAlex 1.0 > PDF RAG 0.9 > ArXiv 0.8）。

**Claim**：一条断言（如"该领域在冷启动场景下缺乏有效方法"），必须关联至少一个 Source ID。置信度基于关联来源的数量和等级自动计算：多源交叉验证的 Claim 置信度更高。

**Candidate**：一个候选选题方向，包含四维评分（新颖性/可行性/影响力/风险）、完整证据链、Critic 评审结论。综合得分 = novelty\*0.3 + feasibility\*0.3 + impact\*0.3 - risk\*0.1。

### 编排层 (`backend/workflows/topic_generation.py`)

`PhasedPipeline` 是系统的编排核心，替代了原来的 AutoGen GroupChat。每个 Phase 是一个独立的 Agent 协作单元：

- `run_landscape_scan()` — Phase 1，Researcher 全景检索
- `run_deep_dive()` — Phase 2，Researcher 定向深挖
- `run_candidate_generation()` — Phase 3，Synthesizer 生成 + Critic 评审
- `run_convergence()` — Phase 4，最终报告生成

Phase 之间的流转由前端 Streamlit 的 `session_state` 控制，两个 HumanGate（Phase 1→2、Phase 3→4）确保用户可以介入方向选择和候选评审。

### 工具层 (`backend/tools/`)

三个数据源按信任等级分层：

| 层级 | 工具 | 数据源 | 用途 |
|------|------|--------|------|
| Tier 1 | `search_openalex_graph` | OpenAlex 知识图谱 | 高被引基石论文、核心概念 |
| Tier 2 | `search_arxiv_literature` | ArXiv API | 最新预印本、前沿动向、撞车检测 |
| Tier 3 | `query_paper_rag` | ChromaDB + PyPDF | PDF 原文深度语义检索 |

所有工具函数都通过 `@auto_register` 装饰器自动将返回结果注册到 EvidenceStore。

**冲突处理规则**：
- SOTA 信息以 ArXiv 为准，经典 Baseline 以 OpenAlex 为准
- 摘要与 RAG 原文冲突时，以 RAG 原文为准
- 同一 Gap 被多源描述时，置信度自动提升

### 评估层 (`backend/evidence/metrics.py`)

L1 自动化健康度指标，每次运行自动收集：

| 指标 | 健康阈值 | 说明 |
|------|----------|------|
| 工具调用成功率 | >= 90% | API 稳定性 |
| 引用溯源率 | 100% | 每条 Claim 是否都有 source_id |
| Critic 驳回率 | 30%-50% | 过低说明方案太平庸，过高说明生成质量差 |
| 单 Phase 耗时 | Phase 1 < 60s | 用户体验 |

---

## 快速开始

### 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器（推荐）或 pip

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd autogen-demo

# 方式 1：使用 uv（推荐）
uv sync

# 方式 2：使用 pip
pip install -e .
```

### 配置

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your-api-key-here
```

系统默认使用 OpenRouter 兼容接口。如需切换模型，修改 `config/llm_config.py` 中的 `model` 字段。

### 运行

```bash
# 方式 1：使用 uv
uv run streamlit run app.py

# 方式 2：直接运行（确保虚拟环境已激活）
streamlit run app.py
```

启动后浏览器会自动打开 `http://localhost:8501`。

---

## 使用指南

### Step 1: 配置 API Key

在左侧控制台输入你的 OpenAI / OpenRouter API Key。可选调节「创新温度」（低 = 保守稳健，高 = 发散创新）。

### Step 2: 输入种子想法（Phase 0）

在输入框输入你的研究方向，例如：

> 大模型在医疗问答中的幻觉控制

如果系统认为想法太模糊，会引导你通过 2-3 个追问缩小范围。

### Step 3: 查看领域全景（Phase 1）

系统自动检索文献，展示：
- 左栏：该领域的高被引基石论文
- 右栏：最新的前沿论文
- 下方：初步识别的 Research Gap 列表

**你需要做的**：从 Gap 列表中勾选 2-3 个你感兴趣的方向（也可以手动输入自定义方向），然后点击「确认方向」。

### Step 4: 审阅深度调研（Phase 2）

系统针对你选定的方向进行 PDF 级的深度调研，展示每个方向的：
- 精确 Research Gap 及其文献证据
- 当前 SOTA 方法和性能
- 可用的公开数据集
- 竞争风险评估

确认后点击「生成候选选题」。

### Step 5: 评审候选选题（Phase 3）

系统生成 2-4 个候选选题，以比较矩阵形式展示四维评分（新颖性/可行性/影响力/风险）。Critic 会对每个候选执行撞车检测。

**你可以做的**：
- 直接点击「生成最终报告」
- 在反馈框输入修改意见后再生成
- 点击「重新生成候选」要求重跑

### Step 6: 获取最终报告（Phase 4）

系统输出 Markdown 格式的《科研选题决议报告》，包含：
- 领域概述
- 每个推荐选题的详细分析（含 source_id 引用）
- 多维度评分表格
- 证据溯源附录（所有引用的论文信息）

底部可展开查看：
- **L1 健康度报告**：工具调用成功率、溯源率、驳回率
- **证据溯源面板**：所有注册的 Source 和 Claim，按置信度标注

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| 多智能体编排 | Microsoft AutoGen (pyautogen) |
| 大模型 | OpenAI 兼容接口 (支持 OpenRouter) |
| 文献检索 | OpenAlex API, ArXiv API |
| 向量检索 / RAG | ChromaDB + LangChain Text Splitters |
| PDF 解析 | PyPDF + PyMuPDF |
| 日志 | Loguru |
| 依赖管理 | uv + pyproject.toml |

---

## 数据源说明

| 数据源 | 信任等级 | 覆盖范围 | 限制 |
|--------|----------|----------|------|
| OpenAlex | Tier 1 (权威基石) | 已发表的同行评审论文，按被引量排序 | 不含预印本 |
| ArXiv | Tier 2 (前沿雷达) | 近几个月的最新预印本 | 未经同行评审 |
| PDF RAG | Tier 3 (深度真理) | 具体论文的方法论细节 | 仅限 Open Access PDF |

---

## 开发

### 项目结构约定

- Agent 的 system_message 通过 YAML 文件管理（`backend/prompts/`）
- 所有工具函数返回 JSON 字符串，状态码为 `"status": "success"` 或 `"error"`
- 工具函数通过 `@auto_register` 装饰器自动注册到 EvidenceStore

### 添加新的数据源

1. 在 `backend/tools/` 下创建新的工具函数文件
2. 使用 `@auto_register("your_source_type")` 装饰器
3. 在 `Source.TIER_WEIGHTS` 中添加该类型的信任权重
4. 在 `evidence_hooks.py` 的 `source_type_map` 中注册映射

### 调试技巧

- 设置 `cache_seed=42` 可以启用 AutoGen 的 LLM 缓存，相同请求秒回且不扣费
- 运行日志位于 `logs/` 目录，`system_*.log` 包含全量 DEBUG 日志
- `logs/audit.json` 包含 JSON 格式的结构化审计记录
