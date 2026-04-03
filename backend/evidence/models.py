from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Source:
    """一条可溯源的文献/数据来源。"""

    source_id: str = field(default_factory=lambda: _gen_id("S"))
    source_type: str = ""          # "openalex" | "arxiv" | "pdf_rag"
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    url: str = ""
    citation_count: Optional[int] = None
    abstract: str = ""
    retrieved_at: datetime = field(default_factory=datetime.now)
    raw_data: dict = field(default_factory=dict)

    # ---- 数据源信任权重 ----
    TIER_WEIGHTS = {"openalex": 1.0, "arxiv": 0.8, "pdf_rag": 0.9}

    @property
    def tier_weight(self) -> float:
        return self.TIER_WEIGHTS.get(self.source_type, 0.5)

    def to_citation(self) -> str:
        author_str = self.authors[0] if self.authors else "Unknown"
        if len(self.authors) > 1:
            author_str += " et al."
        year_str = str(self.year) if self.year else "n.d."
        return f"{author_str}, {year_str}"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "url": self.url,
            "citation_count": self.citation_count,
            "abstract": self.abstract[:500],
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass
class Claim:
    """一条断言，必须由至少一个 Source 支撑。"""

    claim_id: str = field(default_factory=lambda: _gen_id("C"))
    content: str = ""
    claim_type: str = ""           # "gap" | "trend" | "method" | "risk"
    source_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def compute_confidence(self, sources: list[Source]) -> float:
        """基于来源数量和来源等级计算置信度（0‑1）。"""
        if not self.source_ids:
            self.confidence = 0.0
            return self.confidence
        matched = [s for s in sources if s.source_id in self.source_ids]
        if not matched:
            self.confidence = 0.0
            return self.confidence
        avg_weight = sum(s.tier_weight for s in matched) / len(matched)
        count_bonus = min(len(matched) * 0.15, 0.3)
        self.confidence = round(min(avg_weight + count_bonus, 1.0), 2)
        return self.confidence

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "content": self.content,
            "claim_type": self.claim_type,
            "source_ids": self.source_ids,
            "confidence": self.confidence,
        }


@dataclass
class Candidate:
    """一个候选选题方向，附带证据链和多维评分。"""

    candidate_id: str = field(default_factory=lambda: _gen_id("T"))
    title: str = ""
    gap: Optional[Claim] = None
    method: str = ""
    scores: dict = field(default_factory=lambda: {
        "novelty": 0.0,
        "feasibility": 0.0,
        "impact": 0.0,
        "risk": 0.0,
    })
    evidence_chain: list[Claim] = field(default_factory=list)
    critic_verdict: str = ""       # "ACCEPT" | "REJECT" | "REVISE"
    critic_notes: str = ""
    suggested_datasets: list[str] = field(default_factory=list)
    suggested_baselines: list[str] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """加权综合得分（risk 反向计入）。"""
        s = self.scores
        return round(
            s.get("novelty", 0) * 0.3
            + s.get("feasibility", 0) * 0.3
            + s.get("impact", 0) * 0.3
            - s.get("risk", 0) * 0.1,
            2,
        )

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "title": self.title,
            "gap": self.gap.to_dict() if self.gap else None,
            "method": self.method,
            "scores": self.scores,
            "composite_score": self.composite_score,
            "evidence_chain": [c.to_dict() for c in self.evidence_chain],
            "critic_verdict": self.critic_verdict,
            "critic_notes": self.critic_notes,
            "suggested_datasets": self.suggested_datasets,
            "suggested_baselines": self.suggested_baselines,
        }
