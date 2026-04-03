from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from backend.agents.critic_agent import get_synthesizer, get_critic
from backend.agents.research_agent import get_researcher
from backend.agents.user_proxy import get_user_proxy
from backend.evidence.metrics import MetricsCollector
from backend.evidence.models import Candidate
from backend.evidence.store import EvidenceStore
from backend.tools.evidence_hooks import bind_store, register_tool_result_post_hoc
from backend.utils.logger import sys_logger

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def _retry_chat(admin, agent, message: str, max_turns: int, retries: int = MAX_RETRIES):
    """带重试的 initiate_chat 包装器，处理 API 瞬时故障。"""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return admin.initiate_chat(
                agent, message=message, max_turns=max_turns, summary_method="last_msg"
            )
        except (json.JSONDecodeError, Exception) as e:
            last_error = e
            err_name = type(e).__name__
            sys_logger.warning(
                f"API 调用失败 (尝试 {attempt}/{retries}): {err_name}: {e}"
            )
            if attempt < retries:
                sys_logger.info(f"等待 {RETRY_DELAY_SECONDS}s 后重试...")
                time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(
        f"API 调用在 {retries} 次重试后仍然失败: {last_error}"
    ) from last_error


# ------------------------------------------------------------------ #
#  Phase 结果容器
# ------------------------------------------------------------------ #


@dataclass
class PhaseResult:
    phase_name: str
    data: dict = field(default_factory=dict)
    needs_human_input: bool = False
    input_hint: str = ""
    raw_chat_history: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  PhasedPipeline 主类
# ------------------------------------------------------------------ #


class PhasedPipeline:
    """
    分阶段编排引擎。

    每个 Phase 由前端通过 run_phase() 调用，Phase 之间由前端
    （Streamlit session_state）负责流转和 HumanGate 交互。
    """

    def __init__(self, llm_config: dict[str, Any], evidence_store: EvidenceStore):
        self.llm_config = llm_config
        self.store = evidence_store
        self.metrics = MetricsCollector()

        bind_store(evidence_store)
        self._admin = get_user_proxy(human_input_mode="NEVER")

    # ================================================================ #
    #  Phase 1: 全景扫描
    # ================================================================ #

    def run_landscape_scan(self, seed_idea: str) -> PhaseResult:
        """
        全景扫描阶段：Researcher 调用 OpenAlex + ArXiv，
        输出结构化领域地图，将所有来源注册到 EvidenceStore。
        """
        sys_logger.info(">>> Phase 1: 全景扫描 启动")
        self.metrics.start_phase("landscape_scan")

        researcher = get_researcher(
            self.llm_config, executor=self._admin, mode="landscape"
        )

        sources_summary = self.store.get_sources_summary()

        prompt = f"""
            【Phase 1 — 全景扫描任务】
            
            用户的种子想法: "{seed_idea}"
            
            请你执行以下操作：
            1. 调用 search_openalex_graph 检索该方向的高被引基石论文（至少 5 篇）。
            2. 调用 search_arxiv_literature 检索最近的前沿预印本（至少 5 篇）。
            3. 综合检索结果，用以下 JSON 格式输出你的分析：
            
            ```json
            {{
              "foundation_papers": [
                {{"title": "...", "year": ..., "citations": ..., "core_contribution": "一句话概括"}}
              ],
              "frontier_papers": [
                {{"title": "...", "date": "...", "key_innovation": "一句话概括"}}
              ],
              "identified_gaps": [
                {{"gap_description": "...", "evidence": "基于哪些论文发现的，引用标题"}}
              ],
              "landscape_summary": "2-3 句话总结当前领域状态"
            }}
            ```
            
            已有注册来源：
            {sources_summary}
            
            严禁捏造论文。如果检索不到，如实报告。完成后在最后一行输出 TERMINATE
        """

        try:
            chat_result = _retry_chat(
                self._admin, researcher, message=prompt, max_turns=8
            )
        except RuntimeError as e:
            sys_logger.error(f"Phase 1 失败: {e}")
            self.metrics.end_phase()
            return PhaseResult(
                phase_name="landscape_scan",
                data={"landscape": {}, "sources": [], "error": str(e)},
                needs_human_input=True,
                input_hint="API 调用失败，请检查网络和 API Key 后重试。",
            )

        if chat_result and chat_result.chat_history:
            register_tool_result_post_hoc(chat_result.chat_history)
            self.metrics.record_tool_calls_from_chat(chat_result.chat_history)

        landscape_data = self._extract_json_from_chat(chat_result)
        self._register_claims_from_landscape(landscape_data)

        self.metrics.end_phase()
        sys_logger.info(">>> Phase 1: 全景扫描 完成")
        return PhaseResult(
            phase_name="landscape_scan",
            data={
                "landscape": landscape_data,
                "sources": [s.to_dict() for s in self.store.get_all_sources()],
            },
            needs_human_input=True,
            input_hint="请从上方的研究方向和 Gap 中选择 2-3 个你感兴趣的方向进行深入调研。",
            raw_chat_history=chat_result.chat_history if chat_result else [],
        )

    # ================================================================ #
    #  Phase 2: 方向深挖
    # ================================================================ #

    def run_deep_dive(self, seed_idea: str, selected_directions: list[str]) -> PhaseResult:
        """
        方向深挖阶段：针对用户选定的方向，通过 RAG 进行深度验证。
        """
        sys_logger.info(">>> Phase 2: 方向深挖 启动")
        self.metrics.start_phase("deep_dive")

        researcher = get_researcher(
            self.llm_config, executor=self._admin, mode="deep_dive"
        )

        directions_text = "\n".join(
            f"  {i + 1}. {d}" for i, d in enumerate(selected_directions)
        )
        sources_summary = self.store.get_sources_summary()

        prompt = f"""
            【Phase 2 — 方向深挖任务】
            
            用户的种子想法: "{seed_idea}"
            用户选定的调研方向:
            {directions_text}
            
            已注册的文献来源（可用 source_id 引用）：
            {sources_summary}
            
            请对每个选定方向执行深度调研：
            1. 如果已有论文有 PDF 链接，使用 query_paper_rag 查询：
               - 该方向现有方法的具体局限性是什么？
               - 有哪些公开数据集可以使用？
               - 当前 SOTA 方法是什么，性能如何？
            2. 如果需要更多前沿论文，可以追加调用 search_arxiv_literature。
            
            请用以下 JSON 格式输出分析结果：
            
            ```json
            {{
              "directions": [
                {{
                  "direction_name": "...",
                  "precise_gaps": [
                    {{"gap": "...", "supporting_evidence": "引用具体论文标题和发现"}}
                  ],
                  "available_datasets": ["..."],
                  "current_sota": {{"method": "...", "performance": "...", "source": "论文标题"}},
                  "competition_risk": "高/中/低，说明理由"
                }}
              ]
            }}
            ```
            
            严禁捏造论文。完成后在最后一行输出 TERMINATE
        """

        try:
            chat_result = _retry_chat(
                self._admin, researcher, message=prompt, max_turns=10
            )
        except RuntimeError as e:
            sys_logger.error(f"Phase 2 失败: {e}")
            self.metrics.end_phase()
            return PhaseResult(
                phase_name="deep_dive",
                data={"deep_dive": {}, "claims": [], "error": str(e)},
                needs_human_input=False,
            )

        if chat_result and chat_result.chat_history:
            register_tool_result_post_hoc(chat_result.chat_history)
            self.metrics.record_tool_calls_from_chat(chat_result.chat_history)

        deep_dive_data = self._extract_json_from_chat(chat_result)
        self._register_claims_from_deep_dive(deep_dive_data)

        self.metrics.end_phase()
        sys_logger.info(">>> Phase 2: 方向深挖 完成")
        return PhaseResult(
            phase_name="deep_dive",
            data={
                "deep_dive": deep_dive_data,
                "claims": [c.to_dict() for c in self.store.get_all_claims()],
            },
            needs_human_input=False,
            raw_chat_history=chat_result.chat_history if chat_result else [],
        )

    # ================================================================ #
    #  Phase 3: 候选生成与评审
    # ================================================================ #

    def run_candidate_generation(
            self, seed_idea: str, deep_dive_data: dict
    ) -> PhaseResult:
        """
        候选生成阶段：Synthesizer 生成候选选题 → Critic 评审。
        """
        sys_logger.info(">>> Phase 3: 候选生成与评审 启动")
        self.metrics.start_phase("candidate_generation")

        synthesizer = get_synthesizer(self.llm_config)
        critic = get_critic(self.llm_config)

        sources_summary = self.store.get_sources_summary()
        claims_text = "\n".join(
            f"[{c.claim_id}] ({c.claim_type}, 置信度={c.confidence}) {c.content}"
            for c in self.store.get_all_claims()
        )

        synth_prompt = f"""
            【Phase 3 — 候选选题生成任务】
            
            用户的种子想法: "{seed_idea}"
            
            已注册的证据 Claims:
            {claims_text}
            
            已注册的文献来源:
            {sources_summary}
            
            深度调研数据:
            {json.dumps(deep_dive_data, ensure_ascii=False, indent=2)[:4000]}
            
            请基于以上证据生成 2-4 个候选选题方向。每个候选必须：
            - 明确引用至少一个已有的 Claim ID 或 Source ID 作为依据
            - 提供四维评分（1-10 分）
            
            请用以下 JSON 格式输出：
            
            ```json
            {{
              "candidates": [
                {{
                  "title": "学术化选题标题",
                  "gap_description": "基于哪个 Research Gap",
                  "gap_evidence_ids": ["C_xxx", "S_xxx"],
                  "proposed_method": "建议的核心方法/创新点",
                  "scores": {{
                    "novelty": 8,
                    "feasibility": 7,
                    "impact": 8,
                    "risk": 4
                  }},
                  "suggested_datasets": ["数据集1"],
                  "suggested_baselines": ["方法1"],
                  "rationale": "为什么这个选题值得做，2-3句话"
                }}
              ]
            }}
            ```
            
            完成后在最后一行输出 TERMINATE
        """

        try:
            synth_result = _retry_chat(
                self._admin, synthesizer, message=synth_prompt, max_turns=3
            )
        except RuntimeError as e:
            sys_logger.error(f"Phase 3 Synthesizer 失败: {e}")
            self.metrics.end_phase()
            return PhaseResult(
                phase_name="candidate_generation",
                data={"candidates": [], "snapshot": self.store.snapshot(), "error": str(e)},
                needs_human_input=True,
                input_hint="候选生成失败，请检查 API 后重试。",
            )

        candidates_data = self._extract_json_from_chat(synth_result)
        candidates = self._register_candidates(candidates_data)

        critic_prompt = self._build_critic_prompt(candidates, sources_summary)
        critic_result = None
        try:
            critic_result = _retry_chat(
                self._admin, critic, message=critic_prompt, max_turns=3
            )
            self._apply_critic_verdicts(critic_result)
        except RuntimeError as e:
            sys_logger.warning(f"Phase 3 Critic 失败，跳过评审: {e}")

        if critic_result and critic_result.chat_history:
            self.metrics.record_tool_calls_from_chat(critic_result.chat_history)

        self.metrics.end_phase()
        sys_logger.info(">>> Phase 3: 候选生成与评审 完成")
        return PhaseResult(
            phase_name="candidate_generation",
            data={
                "candidates": [c.to_dict() for c in self.store.get_all_candidates()],
                "snapshot": self.store.snapshot(),
            },
            needs_human_input=True,
            input_hint="请评审以上候选选题。您可以采纳、修改或要求重新调研。",
            raw_chat_history=(
                    (synth_result.chat_history if synth_result else [])
                    + (critic_result.chat_history if critic_result else [])
            ),
        )

    # ================================================================ #
    #  Phase 4: 收敛出最终报告
    # ================================================================ #

    def run_convergence(self, seed_idea: str, user_feedback: str = "") -> PhaseResult:
        """
        收敛阶段：基于用户反馈和已有证据，生成最终选题报告。
        """
        sys_logger.info(">>> Phase 4: 收敛报告 启动")
        self.metrics.start_phase("convergence")

        synthesizer = get_synthesizer(self.llm_config)

        accepted = self.store.get_accepted_candidates()
        all_candidates = self.store.get_all_candidates()
        target = accepted if accepted else all_candidates

        candidates_text = json.dumps(
            [c.to_dict() for c in target], ensure_ascii=False, indent=2
        )
        sources_summary = self.store.get_sources_summary()

        prompt = f"""
            【Phase 4 — 最终选题报告生成】
            
            用户的种子想法: "{seed_idea}"
            用户反馈: "{user_feedback if user_feedback else '无特殊要求'}"
            
            已通过评审的候选选题:
            {candidates_text[:8000]}
            
            文献来源:
            {sources_summary}
            
            请生成一份格式精美的 Markdown 最终选题报告，要求：
            
            1. 报告标题为 "# 科研选题决议报告"
            2. 包含 "## 领域风向标" 概述（2-3段，概述研究领域的现状与趋势）
            3. 包含 "## 推荐选题" 部分，对每个推荐选题用 "### 选题N: 标题" 详细展开：
               - **核心 Research Gap**（引用 [source_id]）
               - **建议方法与创新点**
               - **多维度评分**（Markdown 表格：新颖性 / 可行性 / 影响力 / 风险 / 综合分）
               - **潜在风险与缓解策略**
               - **建议数据集与 Baseline**
            4. 包含 "## 证据溯源附录" 部分，列出报告中引用的每个 source_id 对应的论文信息（标题、作者、URL）
            
            每一个断言都必须标注 [source_id] 以便溯源验证。
            请务必一次性输出完整报告，不要分多次输出。
            输出完成后在最后一行单独写 TERMINATE
        """

        try:
            chat_result = _retry_chat(
                self._admin, synthesizer, message=prompt, max_turns=2
            )
        except RuntimeError as e:
            sys_logger.error(f"Phase 4 失败: {e}")
            self.metrics.end_phase()
            return PhaseResult(
                phase_name="convergence",
                data={
                    "final_report": "",
                    "snapshot": self.store.snapshot(),
                    "metrics_report": self.metrics.generate_report(self.store),
                    "error": str(e),
                },
                needs_human_input=False,
            )

        final_report = self._extract_last_content(chat_result)

        self.metrics.end_phase()
        sys_logger.info(">>> Phase 4: 收敛报告 完成")
        return PhaseResult(
            phase_name="convergence",
            data={
                "final_report": final_report,
                "snapshot": self.store.snapshot(),
                "metrics_report": self.metrics.generate_report(self.store),
            },
            needs_human_input=False,
            raw_chat_history=chat_result.chat_history if chat_result else [],
        )

    # ================================================================ #
    #  辅助方法
    # ================================================================ #

    def _extract_json_from_chat(self, chat_result) -> dict:
        """从聊天记录中提取最后一个有效 JSON 块。"""
        if not chat_result or not chat_result.chat_history:
            return {}
        for msg in reversed(chat_result.chat_history):
            content = msg.get("content", "") or ""
            if not content.strip():
                continue
            candidates = self._find_all_json_blocks(content)
            for json_str in candidates:
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
        return {}

    @staticmethod
    def _find_all_json_blocks(text: str) -> list[str]:
        """
        在文本中查找所有可能的 JSON 块，按优先级排序返回。
        优先级: ```json 代码块 > 裸 JSON 对象
        """
        import re
        results = []

        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
            candidate = match.group(1).strip()
            if candidate.startswith("{"):
                results.append(candidate)

        if not results:
            depth = 0
            start_idx = -1
            for i, ch in enumerate(text):
                if ch == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start_idx != -1:
                        results.append(text[start_idx:i + 1])
                        start_idx = -1

        results.sort(key=len, reverse=True)
        return results

    @staticmethod
    def _extract_last_content(chat_result) -> str:
        """
        从聊天记录中提取最终报告内容。

        策略:
        1. 优先找包含「科研选题决议报告」标题的消息（精确匹配报告格式）
        2. 其次找最长的非工具、非 Admin 消息（报告通常是最长的）
        3. 兜底取最后一条有内容的消息
        对找到的内容，只清理末尾的 TERMINATE 标记，不做全局替换。
        """
        if not chat_result or not chat_result.chat_history:
            return "未能生成报告。"

        import re

        def _clean_terminate(text: str) -> str:
            """只移除末尾的 TERMINATE 标记，保留正文中可能出现的同名词。"""
            return re.sub(r'\s*TERMINATE\s*$', '', text).strip()

        messages = chat_result.chat_history

        for msg in reversed(messages):
            content = msg.get("content", "") or ""
            if "科研选题决议报告" in content and len(content) > 200:
                return _clean_terminate(content)

        best_msg = ""
        for msg in messages:
            content = msg.get("content", "") or ""
            role = msg.get("role", "")
            name = msg.get("name", "")
            if role == "user" or name == "Admin":
                continue
            if len(content) > len(best_msg):
                best_msg = content

        if best_msg:
            return _clean_terminate(best_msg)

        for msg in reversed(messages):
            content = msg.get("content", "") or ""
            if content.strip():
                return _clean_terminate(content)

        return "未能提取报告内容。"

    def _register_claims_from_landscape(self, data: dict):
        """从全景扫描结果中提取并注册 Claims。"""
        gaps = data.get("identified_gaps", [])
        for gap_item in gaps:
            desc = gap_item.get("gap_description", "")
            if not desc:
                continue
            evidence_text = gap_item.get("evidence", "")
            matching_sids = self._match_sources_by_text(evidence_text)
            self.store.register_claim(
                content=desc, claim_type="gap", source_ids=matching_sids
            )

    def _register_claims_from_deep_dive(self, data: dict):
        """从深挖结果中提取并注册 Claims。"""
        directions = data.get("directions", [])
        for d in directions:
            for gap_item in d.get("precise_gaps", []):
                gap_text = gap_item.get("gap", "")
                evidence_text = gap_item.get("supporting_evidence", "")
                if not gap_text:
                    continue
                matching_sids = self._match_sources_by_text(evidence_text)
                self.store.register_claim(
                    content=gap_text, claim_type="gap", source_ids=matching_sids
                )
            sota = d.get("current_sota", {})
            if sota.get("method"):
                matching_sids = self._match_sources_by_text(sota.get("source", ""))
                self.store.register_claim(
                    content=f"当前 SOTA: {sota['method']} ({sota.get('performance', 'N/A')})",
                    claim_type="method",
                    source_ids=matching_sids,
                )

    def _register_candidates(self, data: dict) -> list[Candidate]:
        """从 Synthesizer 输出中注册候选选题。"""
        registered = []
        for item in data.get("candidates", []):
            evidence_ids = item.get("gap_evidence_ids", [])
            gap_claim = self.store.register_claim(
                content=item.get("gap_description", ""),
                claim_type="gap",
                source_ids=[sid for sid in evidence_ids if sid.startswith("S")],
            )
            rationale_claim = self.store.register_claim(
                content=item.get("rationale", ""),
                claim_type="trend",
                source_ids=[sid for sid in evidence_ids if sid.startswith("S")],
            )
            candidate = Candidate(
                title=item.get("title", ""),
                gap=gap_claim,
                method=item.get("proposed_method", ""),
                scores=item.get("scores", {}),
                evidence_chain=[gap_claim, rationale_claim],
                suggested_datasets=item.get("suggested_datasets", []),
                suggested_baselines=item.get("suggested_baselines", []),
            )
            self.store.register_candidate(candidate)
            registered.append(candidate)
        return registered

    @staticmethod
    def _build_critic_prompt(candidates: list[Candidate], sources_summary: str) -> str:
        candidates_text = json.dumps(
            [c.to_dict() for c in candidates], ensure_ascii=False, indent=2
        )
        return f"""
            【Phase 3 — 候选选题评审任务】
            
            以下是 Synthesizer 生成的候选选题：
            {candidates_text[:5000]}
            
            已注册的文献来源：
            {sources_summary}
            
            请对每个候选选题执行以下检查：
            1. **撞车检测**: 调用 search_arxiv_literature 检查该选题的核心创新点是否已有高度相似论文。
            2. **可行性检查**: 数据集是否公开？方法是否需要不合理的算力？
            3. **价值判断**: 这个 Gap 是否真的值得解决？
            
            对每个候选给出评审结论，使用以下 JSON 格式：
            
            ```json
            {{
              "reviews": [
                {{
                  "candidate_id": "T_xxx",
                  "verdict": "ACCEPT 或 REJECT 或 REVISE",
                  "scoop_check": "是否撞车，引用找到的论文",
                  "feasibility_check": "可行性评估",
                  "impact_check": "价值评估",
                  "suggestions": "改进建议（如有）"
                }}
              ]
            }}
            ```
            
            完成后在最后一行输出 TERMINATE
        """

    def _apply_critic_verdicts(self, chat_result):
        """从 Critic 输出中提取评审结论并更新 Candidate。"""
        data = self._extract_json_from_chat(chat_result)
        for review in data.get("reviews", []):
            cid = review.get("candidate_id", "")
            verdict = review.get("verdict", "REVISE")
            notes_parts = []
            for key in ("scoop_check", "feasibility_check", "impact_check", "suggestions"):
                val = review.get(key, "")
                if val:
                    notes_parts.append(f"**{key}**: {val}")
            notes = "\n".join(notes_parts)
            self.store.update_candidate_verdict(cid, verdict, notes)

    def _match_sources_by_text(self, text: str) -> list[str]:
        """通过文本模糊匹配已注册的 Source。"""
        if not text:
            return []
        matched = []
        text_lower = text.lower()
        for source in self.store.get_all_sources():
            title_words = source.title.lower().split()
            if len(title_words) >= 3:
                hit_count = sum(1 for w in title_words if len(w) > 3 and w in text_lower)
                if hit_count >= 2:
                    matched.append(source.source_id)
        return matched


# ------------------------------------------------------------------ #
#  向后兼容：保留旧入口签名（供可能的外部调用）
# ------------------------------------------------------------------ #

def run_topic_copilot_workflow(seed_idea: str, llm_config: dict):
    """
    向后兼容入口：执行全流程（无人工介入模式）。
    新前端应直接使用 PhasedPipeline 分阶段调用。
    """
    store = EvidenceStore()
    pipeline = PhasedPipeline(llm_config, store)

    phase1 = pipeline.run_landscape_scan(seed_idea)

    gaps = phase1.data.get("landscape", {}).get("identified_gaps", [])
    directions = [g.get("gap_description", "") for g in gaps[:3]] or [seed_idea]

    phase2 = pipeline.run_deep_dive(seed_idea, directions)

    phase3 = pipeline.run_candidate_generation(
        seed_idea, phase2.data.get("deep_dive", {})
    )

    phase4 = pipeline.run_convergence(seed_idea)

    return phase4
