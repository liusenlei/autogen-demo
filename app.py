"""
AI 科研选题副驾 — 多步骤向导式交互界面

基于 Streamlit session_state 实现分阶段流水线 UI：
  Phase 0  意图澄清（Gatekeeper）
  Phase 1  全景扫描 → HumanGate 1: 用户选择方向
  Phase 2  方向深挖
  Phase 3  候选生成与评审 → HumanGate 2: 用户确认/修改
  Phase 4  收敛报告
"""

import os
import sys
import json
import streamlit as st

from backend.workflows.gatekeeper import check_query_specificity
from backend.workflows.topic_generation import PhasedPipeline
from backend.evidence.store import EvidenceStore
from backend.evidence.comparison import (
    build_comparison_matrix,
    render_comparison_markdown,
    VERDICT_ICONS,
)
from config.llm_config import get_gpt5_config


# ============================================================
# 页面配置与样式
# ============================================================

st.set_page_config(
    page_title="AI 科研选题副驾 | Research Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    footer {visibility: hidden;}
    [data-testid="stToolbar"] > * {
        visibility: hidden;
    }
    [data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
    }
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 960px;}
    .phase-badge {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        font-size: 0.85em; font-weight: 600; margin-right: 8px;
    }
    .phase-active {background: #e8f5e9; color: #2e7d32;}
    .phase-done {background: #e3f2fd; color: #1565c0;}
    .phase-pending {background: #f5f5f5; color: #9e9e9e;}
    pre {border-radius: 8px;}
    table {width: 100%;}
</style>
""", unsafe_allow_html=True)

PHASE_NAMES = {
    0: "意图澄清",
    1: "全景扫描",
    2: "方向深挖",
    3: "候选评审",
    4: "最终报告",
}


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("控制台")
    st.markdown("---")

    api_key = st.text_input("🔑 API Key", type="password", help="OpenAI / OpenRouter 密钥")
    temperature = st.slider("🌡️ 创新温度", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.caption("**工作流阶段**")
    current_phase = st.session_state.get("current_phase", 0)
    for idx, name in PHASE_NAMES.items():
        if idx < current_phase:
            st.caption(f"✅ Phase {idx}: {name}")
        elif idx == current_phase:
            st.caption(f"▶️ **Phase {idx}: {name}**")
        else:
            st.caption(f"⬜ Phase {idx}: {name}")

    st.markdown("---")
    if st.button("🔄 重新开始", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ============================================================
# Session State 初始化
# ============================================================

def init_state():
    defaults = {
        "current_phase": 0,
        "seed_idea": "",
        "messages": [],
        "phase1_result": None,
        "phase2_result": None,
        "phase3_result": None,
        "phase4_result": None,
        "selected_directions": [],
        "evidence_store": None,
        "pipeline": None,
        "user_feedback": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# 辅助：stdout 缓冲器（执行期间只写终端，不动 Streamlit widget 树）
# ============================================================

class _StdoutBuffer:
    """将 stdout 写入内存缓冲区，同时转发到真实终端。"""

    def __init__(self):
        self.buffer = ""

    def write(self, text):
        sys.__stdout__.write(text)
        self.buffer += text

    def flush(self):
        sys.__stdout__.flush()

    def get_log(self, max_chars: int = 5000) -> str:
        return self.buffer[-max_chars:] if self.buffer else ""


def run_pipeline_phase(func, *args, **kwargs):
    """
    执行 Pipeline Phase 函数，捕获 stdout 日志。
    返回 (result, log_text)。不在执行期间更新任何 Streamlit 组件。
    """
    original_stdout = sys.stdout
    buf = _StdoutBuffer()
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = original_stdout
    return result, buf.get_log()


def get_pipeline() -> tuple[PhasedPipeline, EvidenceStore]:
    """获取或创建 Pipeline 和 EvidenceStore。"""
    if st.session_state["evidence_store"] is None:
        st.session_state["evidence_store"] = EvidenceStore()
    if st.session_state["pipeline"] is None:
        llm_config = get_gpt5_config(temperature=temperature, cache_seed=42)
        st.session_state["pipeline"] = PhasedPipeline(
            llm_config, st.session_state["evidence_store"]
        )
    return st.session_state["pipeline"], st.session_state["evidence_store"]


# ============================================================
# 主界面
# ============================================================

st.title("🔬 AI 科研选题副驾")
st.markdown("从一个种子想法出发，经过分阶段调研，最终收敛到有据可依的选题方向。")

if not api_key:
    st.info("👈 请先在左侧控制台填写 API Key 激活系统。")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
current_phase = st.session_state["current_phase"]


# ============================================================
# Phase 0: 意图澄清
# ============================================================

if current_phase == 0:
    st.header("Phase 0: 意图澄清")
    st.markdown("请输入你的研究种子想法。系统会判断是否足够具体。")

    for msg in st.session_state["messages"]:
        role = msg["role"]
        icon = "💡" if role == "user" else "🤖"
        with st.chat_message(role, avatar=icon):
            st.markdown(msg["content"])

    if user_input := st.chat_input("输入研究方向（如：大模型在医疗问答中的幻觉控制）"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="💡"):
            st.markdown(user_input)

        llm_config = get_gpt5_config(temperature=temperature, cache_seed=42)

        with st.spinner("🤔 正在评估课题框架是否明确..."):
            result = check_query_specificity(
                user_input, st.session_state["messages"], llm_config
            )

        if "[VAGUE]" in result:
            clarify = result.replace("[VAGUE]", "").strip()
            st.session_state["messages"].append({"role": "assistant", "content": clarify})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(clarify)
            st.rerun()
        else:
            combined = user_input
            if len(st.session_state["messages"]) >= 3:
                combined = "综合上下文，用户的最终意图是：\n" + "\n".join(
                    f"{m['role']}: {m['content']}" for m in st.session_state["messages"][-4:]
                )
            st.session_state["seed_idea"] = combined
            st.session_state["current_phase"] = 1
            st.success("✅ 意图已明确，进入全景扫描阶段。")
            st.rerun()


# ============================================================
# Phase 1: 全景扫描
# ============================================================

elif current_phase == 1:
    st.header("Phase 1: 全景扫描")
    seed = st.session_state["seed_idea"]
    st.info(f"**种子想法**: {seed[:200]}")

    if st.session_state["phase1_result"] is None:
        try:
            pipeline, store = get_pipeline()
            with st.status("🔍 Researcher 正在检索文献并构建领域全景图...", expanded=False):
                st.write("正在调用 OpenAlex + ArXiv 检索文献...")
                result, log_text = run_pipeline_phase(
                    pipeline.run_landscape_scan, seed
                )
                st.write("检索完成，正在解析结果...")
            st.session_state["phase1_result"] = {
                "data": result.data,
                "raw_chat_history": result.raw_chat_history,
                "agent_log": log_text,
            }
            st.rerun()
        except Exception as e:
            st.session_state["phase1_result"] = {
                "data": {"landscape": {}, "sources": [], "error": f"{type(e).__name__}: {e}"},
                "raw_chat_history": [],
                "agent_log": "",
            }
            st.rerun()

    result_data = st.session_state["phase1_result"]["data"]
    _agent_log = st.session_state["phase1_result"].get("agent_log", "")
    if _agent_log:
        with st.expander("🤖 Agent 推演日志", expanded=False):
            st.code(_agent_log[-5000:], language=None)

    if result_data.get("error"):
        st.error(f"Phase 1 执行出错: {result_data['error']}")
        if st.button("🔄 重试全景扫描", type="primary"):
            st.session_state["phase1_result"] = None
            st.rerun()
        st.stop()

    landscape = result_data.get("landscape", {})

    has_structured = bool(
        landscape.get("foundation_papers")
        or landscape.get("frontier_papers")
        or landscape.get("identified_gaps")
    )

    st.subheader("📊 领域全景图")

    if has_structured:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏛️ 基石论文")
            for paper in landscape.get("foundation_papers", []):
                with st.container():
                    st.markdown(
                        f"**{paper.get('title', 'N/A')}** "
                        f"({paper.get('year', 'N/A')}, 引用: {paper.get('citations', 'N/A')})"
                    )
                    st.caption(paper.get("core_contribution", ""))

        with col2:
            st.markdown("#### 🚀 前沿论文")
            for paper in landscape.get("frontier_papers", []):
                with st.container():
                    st.markdown(
                        f"**{paper.get('title', 'N/A')}** "
                        f"({paper.get('date', 'N/A')})"
                    )
                    st.caption(paper.get("key_innovation", ""))
    else:
        st.warning("未能提取结构化的领域全景数据，展示 Agent 原始分析结果：")
        raw_history = st.session_state["phase1_result"].get("raw_chat_history", [])
        for msg in raw_history:
            content = msg.get("content", "") or ""
            name = msg.get("name", msg.get("role", ""))
            if content.strip() and name != "Admin" and len(content) > 50:
                st.markdown(content[:5000])
                break

    st.markdown("#### 🔍 初步识别的 Research Gaps")
    gaps = landscape.get("identified_gaps", [])
    if gaps:
        gap_options = [g.get("gap_description", f"Gap {i+1}") for i, g in enumerate(gaps)]

        selected = st.multiselect(
            "请选择 2-3 个你感兴趣的方向进行深入调研：",
            options=gap_options,
            default=gap_options[:min(2, len(gap_options))],
            max_selections=3,
        )

        custom_direction = st.text_input(
            "或者输入你自己的调研方向（可选）：",
            placeholder="例如：关注冷启动场景下的数据增强方法",
        )

        if st.button("✅ 确认方向，进入深度调研", type="primary", use_container_width=True):
            directions = list(selected)
            if custom_direction:
                directions.append(custom_direction)
            if not directions:
                st.warning("请至少选择一个方向。")
            else:
                st.session_state["selected_directions"] = directions
                st.session_state["current_phase"] = 2
                st.rerun()
    else:
        st.warning("未能自动识别 Gap。请手动输入调研方向。")
        custom = st.text_input("输入你希望深入调研的方向：")
        if st.button("✅ 确认方向", type="primary") and custom:
            st.session_state["selected_directions"] = [custom]
            st.session_state["current_phase"] = 2
            st.rerun()

    if landscape.get("landscape_summary"):
        st.markdown("---")
        st.markdown(f"**领域概述**: {landscape['landscape_summary']}")

    with st.expander("📋 已注册的文献来源"):
        sources = result_data.get("sources", [])
        for s in sources:
            st.caption(
                f"[{s['source_id']}] {s['title']} | {s['source_type']} | {s.get('url', '')}"
            )


# ============================================================
# Phase 2: 方向深挖
# ============================================================

elif current_phase == 2:
    st.header("Phase 2: 方向深挖")
    directions = st.session_state["selected_directions"]
    st.info("选定方向: " + " / ".join(directions))

    if st.session_state["phase2_result"] is None:
        try:
            pipeline, store = get_pipeline()
            with st.status("🔬 Researcher 正在对选定方向进行深度调研...", expanded=False):
                st.write("正在执行 PDF RAG 和追加检索...")
                result, log_text = run_pipeline_phase(
                    pipeline.run_deep_dive,
                    st.session_state["seed_idea"],
                    directions,
                )
                st.write("深度调研完成。")
            st.session_state["phase2_result"] = {
                "data": result.data,
                "raw_chat_history": result.raw_chat_history,
                "agent_log": log_text,
            }
            st.rerun()
        except Exception as e:
            st.session_state["phase2_result"] = {
                "data": {"deep_dive": {}, "claims": [], "error": f"{type(e).__name__}: {e}"},
                "raw_chat_history": [],
                "agent_log": "",
            }
            st.rerun()

    phase2_data = st.session_state["phase2_result"]["data"]
    _agent_log = st.session_state["phase2_result"].get("agent_log", "")
    if _agent_log:
        with st.expander("🤖 Agent 推演日志", expanded=False):
            st.code(_agent_log[-5000:], language=None)

    if phase2_data.get("error"):
        st.error(f"Phase 2 执行出错: {phase2_data['error']}")
        if st.button("🔄 重试深度调研", type="primary"):
            st.session_state["phase2_result"] = None
            st.rerun()
        st.stop()

    deep_dive = phase2_data.get("deep_dive", {})

    st.subheader("🔬 深度调研结果")
    for d in deep_dive.get("directions", []):
        with st.expander(f"📌 {d.get('direction_name', '方向')}", expanded=True):
            st.markdown("**精确 Gaps:**")
            for gap in d.get("precise_gaps", []):
                st.markdown(f"- {gap.get('gap', '')}")
                st.caption(f"  证据: {gap.get('supporting_evidence', '')}")

            sota = d.get("current_sota", {})
            if sota:
                st.markdown(
                    f"**当前 SOTA**: {sota.get('method', 'N/A')} "
                    f"(性能: {sota.get('performance', 'N/A')})"
                )

            datasets = d.get("available_datasets", [])
            if datasets:
                st.markdown(f"**可用数据集**: {', '.join(datasets)}")

            risk = d.get("competition_risk", "")
            if risk:
                st.markdown(f"**竞争风险**: {risk}")

    st.markdown("---")
    if st.button("🚀 生成候选选题并提交评审", type="primary", use_container_width=True):
        st.session_state["current_phase"] = 3
        st.rerun()


# ============================================================
# Phase 3: 候选生成与评审
# ============================================================

elif current_phase == 3:
    st.header("Phase 3: 候选选题生成与评审")

    if st.session_state["phase3_result"] is None:
        try:
            pipeline, store = get_pipeline()
            deep_dive_data = (
                st.session_state["phase2_result"]["data"].get("deep_dive", {})
                if st.session_state["phase2_result"]
                else {}
            )
            with st.status("🧠 Synthesizer 正在生成候选选题，Critic 正在评审...", expanded=False):
                st.write("正在生成候选选题并进行评审...")
                result, log_text = run_pipeline_phase(
                    pipeline.run_candidate_generation,
                    st.session_state["seed_idea"],
                    deep_dive_data,
                )
                st.write("候选生成与评审完成。")
            st.session_state["phase3_result"] = {
                "data": result.data,
                "raw_chat_history": result.raw_chat_history,
                "agent_log": log_text,
            }
            st.rerun()
        except Exception as e:
            st.session_state["phase3_result"] = {
                "data": {"candidates": [], "snapshot": {}, "error": f"{type(e).__name__}: {e}"},
                "raw_chat_history": [],
                "agent_log": "",
            }
            st.rerun()

    phase3_data = st.session_state["phase3_result"]["data"]
    _agent_log = st.session_state["phase3_result"].get("agent_log", "")
    if _agent_log:
        with st.expander("🤖 Agent 推演日志", expanded=False):
            st.code(_agent_log[-5000:], language=None)

    if phase3_data.get("error"):
        st.error(f"Phase 3 执行出错: {phase3_data['error']}")
        if st.button("🔄 重试候选生成", type="primary"):
            st.session_state["phase3_result"] = None
            st.rerun()
        st.stop()

    candidates_data = phase3_data.get("candidates", [])

    st.subheader("📊 候选选题比较矩阵")

    _, store = get_pipeline()
    all_candidates = store.get_all_candidates()

    if all_candidates:
        matrix_md = render_comparison_markdown(all_candidates)
        st.markdown(matrix_md)
    elif candidates_data:
        for c in candidates_data:
            verdict_icon = VERDICT_ICONS.get(c.get("critic_verdict", ""), "⏳")
            st.markdown(f"### {verdict_icon} {c.get('title', 'N/A')}")
            scores = c.get("scores", {})
            cols = st.columns(4)
            cols[0].metric("新颖性", f"{scores.get('novelty', 0)}/10")
            cols[1].metric("可行性", f"{scores.get('feasibility', 0)}/10")
            cols[2].metric("影响力", f"{scores.get('impact', 0)}/10")
            cols[3].metric("风险", f"{scores.get('risk', 0)}/10")
            if c.get("critic_notes"):
                st.caption(f"评审意见: {c['critic_notes'][:200]}")
            st.markdown("---")
    else:
        st.warning("未能生成候选选题数据。")

    with st.expander("📋 完整证据快照"):
        snapshot = st.session_state["phase3_result"]["data"].get("snapshot", {})
        stats = snapshot.get("stats", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("文献来源", stats.get("total_sources", 0))
        c2.metric("证据断言", stats.get("total_claims", 0))
        c3.metric("平均置信度", stats.get("avg_claim_confidence", 0))

    st.markdown("---")
    st.markdown("### 您的评审意见")
    user_feedback = st.text_area(
        "对候选选题的反馈（可选）：",
        placeholder="例如：方案 1 不错，但希望聚焦在轻量化方向；方案 3 的数据集不太合适...",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ 生成最终报告", type="primary", use_container_width=True):
            st.session_state["user_feedback"] = user_feedback
            st.session_state["current_phase"] = 4
            st.rerun()
    with col_b:
        if st.button("🔄 重新生成候选", use_container_width=True):
            st.session_state["phase3_result"] = None
            st.rerun()


# ============================================================
# Phase 4: 最终报告
# ============================================================

elif current_phase == 4:
    st.header("Phase 4: 最终选题报告")

    if st.session_state["phase4_result"] is None:
        try:
            pipeline, store = get_pipeline()
            with st.status("📝 正在生成最终选题报告...", expanded=False):
                st.write("Synthesizer 正在综合所有证据生成报告...")
                result, log_text = run_pipeline_phase(
                    pipeline.run_convergence,
                    st.session_state["seed_idea"],
                    st.session_state.get("user_feedback", ""),
                )
                st.write("报告生成完成。")
            st.session_state["phase4_result"] = {
                "data": result.data,
                "raw_chat_history": result.raw_chat_history,
                "agent_log": log_text,
            }
            st.rerun()
        except Exception as e:
            st.session_state["phase4_result"] = {
                "data": {
                    "final_report": "",
                    "snapshot": {},
                    "metrics_report": {},
                    "error": f"{type(e).__name__}: {e}",
                },
                "raw_chat_history": [],
                "agent_log": "",
            }
            st.rerun()

    phase4_data = st.session_state["phase4_result"]["data"]
    _agent_log = st.session_state["phase4_result"].get("agent_log", "")
    if _agent_log:
        with st.expander("🤖 Agent 推演日志", expanded=False):
            st.code(_agent_log[-5000:], language=None)

    if phase4_data.get("error"):
        st.error(f"Phase 4 执行出错: {phase4_data['error']}")
        if st.button("🔄 重试报告生成", type="primary"):
            st.session_state["phase4_result"] = None
            st.rerun()

    final_report = phase4_data.get("final_report", "")

    st.success("🎉 选题调研完成！以下是最终报告：")
    st.markdown("---")

    if final_report and len(final_report) > 100:
        report_parts = final_report.split("\n\n")
        CHUNK_SIZE = 20
        for i in range(0, len(report_parts), CHUNK_SIZE):
            chunk = "\n\n".join(report_parts[i:i + CHUNK_SIZE])
            st.markdown(chunk, unsafe_allow_html=True)
    else:
        st.warning("LLM 生成的报告内容不完整，以下展示 Agent 原始输出：")
        raw_history = st.session_state["phase4_result"].get("raw_chat_history", [])
        shown = False
        for msg in raw_history:
            content = msg.get("content", "") or ""
            name = msg.get("name", msg.get("role", ""))
            if name == "Admin" or not content.strip():
                continue
            if len(content) > 100:
                st.markdown(content[:8000])
                shown = True
        if not shown:
            st.info("未能提取到有效输出，请查看下方的结构化数据。")

    snapshot = st.session_state["phase4_result"]["data"].get("snapshot", {})
    candidates_data = snapshot.get("candidates", [])
    if candidates_data:
        with st.expander("📊 候选选题结构化数据（完整）", expanded=not bool(final_report)):
            for idx, c in enumerate(candidates_data, 1):
                verdict_icon = VERDICT_ICONS.get(c.get("critic_verdict", ""), "⏳")
                st.markdown(f"#### {verdict_icon} {idx}. {c.get('title', 'N/A')}")

                scores = c.get("scores", {})
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("新颖性", f"{scores.get('novelty', 0)}/10")
                sc2.metric("可行性", f"{scores.get('feasibility', 0)}/10")
                sc3.metric("影响力", f"{scores.get('impact', 0)}/10")
                sc4.metric("风险", f"{scores.get('risk', 0)}/10")
                sc5.metric("综合分", c.get("composite_score", "N/A"))

                gap = c.get("gap")
                if gap:
                    st.markdown(f"**Research Gap**: {gap.get('content', 'N/A')}")
                    conf = gap.get("confidence", 0)
                    bar = "🟢" if conf >= 0.7 else ("🟡" if conf >= 0.4 else "🔴")
                    st.caption(f"{bar} 置信度: {conf} | 来源: {', '.join(gap.get('source_ids', []))}")

                if c.get("method"):
                    st.markdown(f"**建议方法**: {c['method']}")
                if c.get("suggested_datasets"):
                    st.markdown(f"**数据集**: {', '.join(c['suggested_datasets'])}")
                if c.get("suggested_baselines"):
                    st.markdown(f"**Baseline**: {', '.join(c['suggested_baselines'])}")
                if c.get("critic_notes"):
                    st.info(f"**评审意见**: {c['critic_notes'][:500]}")
                st.markdown("---")

    st.markdown("---")

    with st.expander("📈 L1 系统健康度报告"):
        metrics_report = st.session_state["phase4_result"]["data"].get("metrics_report", {})
        if metrics_report:
            summary = metrics_report.get("summary", {})
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "工具调用成功率",
                f"{summary.get('overall_tool_success_rate', 0) * 100:.0f}%",
            )
            mc2.metric(
                "引用溯源率",
                f"{summary.get('citation_traceability_rate', 0) * 100:.0f}%",
            )
            mc3.metric(
                "Critic 驳回率",
                f"{summary.get('critic_rejection_rate', 0) * 100:.0f}%",
            )

            health = metrics_report.get("health_checks", {})
            for check_name, check_data in health.items():
                status = "✅" if check_data.get("healthy") else "⚠️"
                st.caption(
                    f"{status} **{check_name}**: {check_data.get('value', 'N/A')} "
                    f"(阈值: {check_data.get('threshold', 'N/A')})"
                )

            st.caption(
                f"总耗时: {summary.get('total_duration_seconds', 0):.1f}s | "
                f"完成阶段: {summary.get('total_phases_completed', 0)}"
            )

    with st.expander("📋 证据溯源面板"):
        snapshot = st.session_state["phase4_result"]["data"].get("snapshot", {})

        st.markdown("#### 文献来源")
        for s in snapshot.get("sources", []):
            url_link = f" [📄 PDF]({s['url']})" if s.get("url") else ""
            st.caption(
                f"**[{s['source_id']}]** {s['title']} "
                f"({s.get('source_type', '')}){url_link}"
            )

        st.markdown("#### 证据断言")
        for c in snapshot.get("claims", []):
            conf = c.get("confidence", 0)
            bar = "🟢" if conf >= 0.7 else ("🟡" if conf >= 0.4 else "🔴")
            st.caption(
                f"{bar} [{c['claim_id']}] ({c['claim_type']}, 置信度={conf}) "
                f"{c['content'][:100]}"
            )

        st.markdown("#### 审计统计")
        stats = snapshot.get("stats", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("文献来源", stats.get("total_sources", 0))
        c2.metric("证据断言", stats.get("total_claims", 0))
        c3.metric("候选选题", stats.get("total_candidates", 0))
        c4.metric("通过评审", stats.get("accepted_candidates", 0))

    st.markdown("---")
    st.markdown("### 📊 您对这份报告的评价")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 极具启发", use_container_width=True):
            st.toast("✅ 已记录正向反馈！")
    with col2:
        if st.button("🤔 方向不错但需调整", use_container_width=True):
            st.toast("📝 建议点击「重新开始」输入更精确的约束条件。")
    with col3:
        if st.button("👎 不满意", use_container_width=True):
            st.toast("⚠️ 已记录。建议调整「创新温度」或尝试更具体的种子想法。")
