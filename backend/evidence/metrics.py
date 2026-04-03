"""
L1 自动化评估指标

持续监控系统的工具链路健康度和输出质量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from backend.evidence.store import EvidenceStore
from backend.utils.logger import sys_logger


@dataclass
class PhaseMetrics:
    """单个 Phase 的运行指标。"""
    phase_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    tool_calls_total: int = 0
    tool_calls_success: int = 0
    tool_calls_failed: int = 0

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def tool_success_rate(self) -> float:
        if self.tool_calls_total == 0:
            return 1.0
        return round(self.tool_calls_success / self.tool_calls_total, 2)


class MetricsCollector:
    """
    L1 自动化评估指标收集器。

    职责:
      - 跟踪每个 Phase 的工具调用成功率
      - 统计引用溯源率（Claim 是否都有 source_id）
      - 统计 Critic 驳回率
      - 监控 Phase 耗时
    """

    def __init__(self):
        self._phase_metrics: list[PhaseMetrics] = []
        self._current_phase: PhaseMetrics | None = None

    def start_phase(self, phase_name: str):
        self._current_phase = PhaseMetrics(phase_name=phase_name)
        sys_logger.debug(f"[Metrics] Phase 开始: {phase_name}")

    def end_phase(self):
        if self._current_phase:
            self._current_phase.end_time = datetime.now()
            self._phase_metrics.append(self._current_phase)
            sys_logger.debug(
                f"[Metrics] Phase 结束: {self._current_phase.phase_name} "
                f"({self._current_phase.duration_seconds:.1f}s)"
            )
            self._current_phase = None

    def record_tool_call(self, success: bool):
        if self._current_phase:
            self._current_phase.tool_calls_total += 1
            if success:
                self._current_phase.tool_calls_success += 1
            else:
                self._current_phase.tool_calls_failed += 1

    def record_tool_calls_from_chat(self, chat_history: list[dict]):
        """从聊天历史中提取工具调用结果并统计。"""
        tool_call_names = {
            "search_openalex_graph", "search_arxiv_literature",
            "query_paper_rag", "search_arxiv_literature",
        }
        for msg in chat_history:
            name = msg.get("name", "")
            content = msg.get("content", "")
            if name in tool_call_names and content:
                try:
                    data = json.loads(content)
                    success = data.get("status") != "error"
                except (json.JSONDecodeError, AttributeError):
                    success = bool(content.strip())
                self.record_tool_call(success)

    # ------------------------------------------------------------------ #
    #  聚合指标
    # ------------------------------------------------------------------ #

    def compute_citation_traceability(self, store: EvidenceStore) -> float:
        """引用溯源率：有 source_id 的 Claim 占比。"""
        claims = store.get_all_claims()
        if not claims:
            return 1.0
        traced = sum(1 for c in claims if c.source_ids)
        return round(traced / len(claims), 2)

    def compute_critic_rejection_rate(self, store: EvidenceStore) -> float:
        """Critic 驳回率。健康范围 30%-50%。"""
        candidates = store.get_all_candidates()
        if not candidates:
            return 0.0
        reviewed = [c for c in candidates if c.critic_verdict]
        if not reviewed:
            return 0.0
        rejected = sum(1 for c in reviewed if c.critic_verdict == "REJECT")
        return round(rejected / len(reviewed), 2)

    def compute_overall_tool_success_rate(self) -> float:
        """整体工具调用成功率。"""
        total = sum(p.tool_calls_total for p in self._phase_metrics)
        success = sum(p.tool_calls_success for p in self._phase_metrics)
        if total == 0:
            return 1.0
        return round(success / total, 2)

    def generate_report(self, store: EvidenceStore) -> dict:
        """生成完整的 L1 评估报告。"""
        phase_details = []
        for pm in self._phase_metrics:
            phase_details.append({
                "phase": pm.phase_name,
                "duration_seconds": pm.duration_seconds,
                "tool_calls_total": pm.tool_calls_total,
                "tool_success_rate": pm.tool_success_rate,
            })

        overall_tool_rate = self.compute_overall_tool_success_rate()
        citation_rate = self.compute_citation_traceability(store)
        critic_rejection = self.compute_critic_rejection_rate(store)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_tool_success_rate": overall_tool_rate,
                "citation_traceability_rate": citation_rate,
                "critic_rejection_rate": critic_rejection,
                "total_phases_completed": len(self._phase_metrics),
                "total_duration_seconds": sum(
                    p.duration_seconds for p in self._phase_metrics
                ),
            },
            "health_checks": {
                "tool_success_rate": _health_check(overall_tool_rate, 0.9, "≥ 90%"),
                "citation_traceability": _health_check(citation_rate, 1.0, "100%"),
                "critic_rejection_rate": _health_check_range(
                    critic_rejection, 0.3, 0.5, "30%-50%"
                ),
            },
            "phase_details": phase_details,
            "evidence_stats": store.snapshot().get("stats", {}),
        }

        sys_logger.info(
            f"[Metrics] L1 报告: 工具成功率={overall_tool_rate}, "
            f"溯源率={citation_rate}, 驳回率={critic_rejection}"
        )
        return report


def _health_check(value: float, threshold: float, label: str) -> dict:
    return {
        "value": value,
        "threshold": label,
        "healthy": value >= threshold,
    }


def _health_check_range(
    value: float, low: float, high: float, label: str
) -> dict:
    return {
        "value": value,
        "threshold": label,
        "healthy": low <= value <= high if value > 0 else True,
    }
