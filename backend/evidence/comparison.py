"""
多维度候选选题比较矩阵

提供将 EvidenceStore 中的 Candidate 列表转换为
前端可渲染的比较矩阵数据结构和 Markdown 表格。
"""

from __future__ import annotations

from backend.evidence.models import Candidate


DIMENSION_LABELS = {
    "novelty": "新颖性",
    "feasibility": "可行性",
    "impact": "影响力",
    "risk": "风险",
}

VERDICT_LABELS = {
    "ACCEPT": "通过",
    "REJECT": "驳回",
    "REVISE": "需修改",
    "": "待评审",
}

VERDICT_ICONS = {
    "ACCEPT": "✅",
    "REJECT": "❌",
    "REVISE": "🔄",
    "": "⏳",
}


def build_comparison_matrix(candidates: list[Candidate]) -> dict:
    """
    将候选列表转换为结构化比较矩阵。

    返回格式:
    {
        "dimensions": ["novelty", "feasibility", "impact", "risk"],
        "candidates": [
            {
                "id": "T_xxx",
                "title": "...",
                "scores": {"novelty": 8, ...},
                "composite_score": 7.2,
                "verdict": "ACCEPT",
                "gap_summary": "...",
                "method_summary": "...",
            }
        ],
        "ranking": ["T_xxx", "T_yyy", ...]
    }
    """
    dimensions = list(DIMENSION_LABELS.keys())

    items = []
    for c in candidates:
        items.append({
            "id": c.candidate_id,
            "title": c.title,
            "scores": {d: c.scores.get(d, 0) for d in dimensions},
            "composite_score": c.composite_score,
            "verdict": c.critic_verdict,
            "verdict_label": VERDICT_LABELS.get(c.critic_verdict, c.critic_verdict),
            "gap_summary": c.gap.content[:120] if c.gap else "",
            "method_summary": c.method[:120],
            "critic_notes": c.critic_notes[:200],
            "datasets": c.suggested_datasets,
            "baselines": c.suggested_baselines,
        })

    items.sort(key=lambda x: x["composite_score"], reverse=True)

    return {
        "dimensions": dimensions,
        "dimension_labels": DIMENSION_LABELS,
        "candidates": items,
        "ranking": [it["id"] for it in items],
    }


def render_comparison_markdown(candidates: list[Candidate]) -> str:
    """将候选选题渲染为 Markdown 比较表格。"""
    if not candidates:
        return "*暂无候选选题。*"

    matrix = build_comparison_matrix(candidates)
    lines = []

    lines.append("## 候选选题比较矩阵\n")

    header = "| 排名 | 选题 | 新颖性 | 可行性 | 影响力 | 风险 | 综合分 | 评审 |"
    sep = "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines.append(header)
    lines.append(sep)

    for rank, item in enumerate(matrix["candidates"], 1):
        s = item["scores"]
        icon = VERDICT_ICONS.get(item["verdict"], "")
        lines.append(
            f"| {rank} "
            f"| **{item['title'][:50]}** "
            f"| {_score_bar(s['novelty'])} "
            f"| {_score_bar(s['feasibility'])} "
            f"| {_score_bar(s['impact'])} "
            f"| {_score_bar(s['risk'])} "
            f"| **{item['composite_score']}** "
            f"| {icon} {item['verdict_label']} |"
        )

    lines.append("")

    for rank, item in enumerate(matrix["candidates"], 1):
        lines.append(f"### {rank}. {item['title']}")
        lines.append(f"- **Research Gap**: {item['gap_summary']}")
        lines.append(f"- **建议方法**: {item['method_summary']}")
        if item["datasets"]:
            lines.append(f"- **推荐数据集**: {', '.join(item['datasets'])}")
        if item["baselines"]:
            lines.append(f"- **推荐 Baseline**: {', '.join(item['baselines'])}")
        if item["critic_notes"]:
            lines.append(f"- **评审意见**: {item['critic_notes']}")
        lines.append("")

    return "\n".join(lines)


def _score_bar(score: float) -> str:
    """将 1-10 分转换为可视化的条形。"""
    score = max(0, min(10, score))
    filled = round(score)
    return "█" * filled + "░" * (10 - filled) + f" {score}"
