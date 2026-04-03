from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from backend.evidence.models import Source, Claim, Candidate, _gen_id
from backend.utils.logger import sys_logger


class EvidenceStore:
    """
    证据管理中枢 —— 所有检索结果、断言、候选选题的唯一注册中心。

    职责:
      1. 接收工具层返回的文献数据，注册为 Source
      2. 管理 Agent 生成的 Claim，并自动计算置信度
      3. 存储和管理候选选题 Candidate
      4. 提供审计快照，用于前端证据面板渲染
    """

    def __init__(self):
        self._sources: dict[str, Source] = {}
        self._claims: dict[str, Claim] = {}
        self._candidates: dict[str, Candidate] = {}
        self._audit_log: list[dict] = []
        sys_logger.debug("EvidenceStore 初始化完成")

    # ------------------------------------------------------------------ #
    #  Source 管理
    # ------------------------------------------------------------------ #

    def register_source(
        self,
        source_type: str,
        title: str,
        url: str = "",
        authors: list[str] | None = None,
        year: int | None = None,
        citation_count: int | None = None,
        abstract: str = "",
        raw_data: dict | None = None,
    ) -> Source:
        """注册一条文献来源，返回包含 source_id 的 Source 对象。"""
        for existing in self._sources.values():
            if existing.title and existing.title.lower() == title.lower():
                sys_logger.debug(f"Source 去重命中: {title[:60]}")
                return existing

        source = Source(
            source_id=_gen_id("S"),
            source_type=source_type,
            title=title,
            authors=authors or [],
            year=year,
            url=url,
            citation_count=citation_count,
            abstract=abstract,
            retrieved_at=datetime.now(),
            raw_data=raw_data or {},
        )
        self._sources[source.source_id] = source
        self._log("register_source", source.source_id, title[:80])
        return source

    def register_sources_from_tool_result(
        self, tool_result_json: str, source_type: str
    ) -> list[Source]:
        """从工具函数的 JSON 返回值中批量注册 Source。"""
        registered: list[Source] = []
        try:
            data = json.loads(tool_result_json)
        except json.JSONDecodeError:
            sys_logger.warning("无法解析工具返回的 JSON，跳过 Source 注册")
            return registered

        if data.get("status") != "success":
            return registered

        items = data.get("papers", data.get("data", []))
        for item in items:
            authors_raw = item.get("authors", [])
            if authors_raw and isinstance(authors_raw[0], dict):
                authors_raw = [a.get("display_name", str(a)) for a in authors_raw]
            first_author = item.get("first_author")
            if first_author and first_author not in authors_raw:
                authors_raw = [first_author] + authors_raw

            source = self.register_source(
                source_type=source_type,
                title=item.get("title", "Untitled"),
                url=item.get("pdf_url", ""),
                authors=authors_raw,
                year=item.get("year") or self._parse_year(item.get("published_date")),
                citation_count=item.get("citation_count"),
                abstract=item.get("abstract", item.get("summary", "")),
                raw_data=item,
            )
            registered.append(source)
        return registered

    def get_source(self, source_id: str) -> Optional[Source]:
        return self._sources.get(source_id)

    def get_all_sources(self) -> list[Source]:
        return list(self._sources.values())

    def get_sources_summary(self) -> str:
        """返回 Agent 可消费的文本摘要，包含 source_id 以便引用。"""
        if not self._sources:
            return "当前没有已注册的文献来源。"
        lines = []
        for s in self._sources.values():
            cite = s.to_citation()
            lines.append(
                f"[{s.source_id}] {s.title} ({cite}) "
                f"| type={s.source_type} citations={s.citation_count or 'N/A'} "
                f"| url={s.url}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Claim 管理
    # ------------------------------------------------------------------ #

    def register_claim(
        self,
        content: str,
        claim_type: str,
        source_ids: list[str],
    ) -> Claim:
        """注册一条断言并自动计算置信度。"""
        claim = Claim(
            claim_id=_gen_id("C"),
            content=content,
            claim_type=claim_type,
            source_ids=source_ids,
        )
        matched_sources = [self._sources[sid] for sid in source_ids if sid in self._sources]
        claim.compute_confidence(matched_sources)
        self._claims[claim.claim_id] = claim
        self._log("register_claim", claim.claim_id, content[:80])
        return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)

    def get_all_claims(self) -> list[Claim]:
        return list(self._claims.values())

    # ------------------------------------------------------------------ #
    #  Candidate 管理
    # ------------------------------------------------------------------ #

    def register_candidate(self, candidate: Candidate) -> Candidate:
        self._candidates[candidate.candidate_id] = candidate
        self._log("register_candidate", candidate.candidate_id, candidate.title[:80])
        return candidate

    def update_candidate_verdict(
        self, candidate_id: str, verdict: str, notes: str
    ) -> Optional[Candidate]:
        cand = self._candidates.get(candidate_id)
        if cand:
            cand.critic_verdict = verdict
            cand.critic_notes = notes
            self._log("update_verdict", candidate_id, verdict)
        return cand

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        return self._candidates.get(candidate_id)

    def get_all_candidates(self) -> list[Candidate]:
        return list(self._candidates.values())

    def get_accepted_candidates(self) -> list[Candidate]:
        return [c for c in self._candidates.values() if c.critic_verdict == "ACCEPT"]

    # ------------------------------------------------------------------ #
    #  审计与快照
    # ------------------------------------------------------------------ #

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)

    def snapshot(self) -> dict:
        """生成完整的证据快照，用于前端渲染或持久化。"""
        return {
            "sources": [s.to_dict() for s in self._sources.values()],
            "claims": [c.to_dict() for c in self._claims.values()],
            "candidates": [c.to_dict() for c in self._candidates.values()],
            "stats": {
                "total_sources": len(self._sources),
                "total_claims": len(self._claims),
                "total_candidates": len(self._candidates),
                "accepted_candidates": len(self.get_accepted_candidates()),
                "avg_claim_confidence": self._avg_confidence(),
            },
        }

    # ------------------------------------------------------------------ #
    #  内部工具
    # ------------------------------------------------------------------ #

    def _avg_confidence(self) -> float:
        if not self._claims:
            return 0.0
        return round(
            sum(c.confidence for c in self._claims.values()) / len(self._claims), 2
        )

    @staticmethod
    def _parse_year(date_str: str | None) -> int | None:
        if not date_str:
            return None
        try:
            return int(date_str[:4])
        except (ValueError, TypeError):
            return None

    def _log(self, action: str, entity_id: str, detail: str):
        entry = {
            "action": action,
            "entity_id": entity_id,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        }
        self._audit_log.append(entry)
        sys_logger.debug(f"[EvidenceStore] {action}: {entity_id} | {detail}")
