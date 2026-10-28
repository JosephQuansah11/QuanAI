"""Memory: what the agent has seen, tried and learned.

Two stores back the agent's ability to improve between runs:

* an append-only item log (observations, decisions, artifact notes) that is
  keyword-searchable, so the agent can recall relevant prior context;
* lessons, which pair a failure signature with the strategy that fixed it.
  The planner and the repair loop both consult these before acting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..util import (
    append_jsonl,
    atomic_write_json,
    digest,
    keywords,
    now_iso,
    read_json,
    read_jsonl,
    short_id,
    tokens,
)


@dataclass
class MemoryItem:
    id: str
    kind: str
    text: str
    data: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    at: str = field(default_factory=now_iso)
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "data": self.data,
            "tags": self.tags,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        return cls(
            id=data.get("id", short_id()),
            kind=data.get("kind", "note"),
            text=data.get("text", ""),
            data=dict(data.get("data") or {}),
            tags=list(data.get("tags") or []),
            at=data.get("at") or now_iso(),
        )


class Memory:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.items_path = self.root / "items.jsonl"
        self.lessons_path = self.root / "lessons.json"
        self.plans_path = self.root / "plans.json"

    # -- items ----------------------------------------------------------
    def add(self, kind: str, text: str, data: dict | None = None, tags: list[str] | None = None) -> MemoryItem:
        item = MemoryItem(
            id=f"mem-{short_id('', 8)}",
            kind=kind,
            text=text.strip(),
            data=data or {},
            tags=tags or keywords(text, 6),
        )
        append_jsonl(self.items_path, item.to_dict())
        return item

    def items(self, kind: str | None = None, limit: int = 200) -> list[MemoryItem]:
        rows = [MemoryItem.from_dict(r) for r in read_jsonl(self.items_path)]
        if kind:
            rows = [r for r in rows if r.kind == kind]
        return rows[-limit:]

    def search(self, query: str, kind: str | None = None, limit: int = 5) -> list[MemoryItem]:
        query_tokens = [t for t in tokens(query) if len(t) > 2]
        if not query_tokens:
            return []
        corpus = self.items(kind=kind, limit=2000)
        if not corpus:
            return []
        doc_freq: dict[str, int] = {}
        docs = []
        for item in corpus:
            doc_tokens = set(tokens(f"{item.text} {' '.join(item.tags)}"))
            docs.append((item, doc_tokens))
            for token in doc_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        total = len(docs)
        scored: list[MemoryItem] = []
        for item, doc_tokens in docs:
            score = 0.0
            for token in query_tokens:
                if token in doc_tokens:
                    score += 1.0 + math.log(1.0 + total / (1.0 + doc_freq.get(token, 0)))
            if score > 0:
                item.score = round(score, 3)
                scored.append(item)
        scored.sort(key=lambda entry: (entry.score, entry.at), reverse=True)
        return scored[:limit]

    # -- lessons --------------------------------------------------------
    def lessons(self) -> list[dict]:
        return read_json(self.lessons_path, default=[]) or []

    def learn(self, tool: str, signature: str, advice: str, evidence: str = "") -> dict:
        rows = self.lessons()
        key = digest(tool, signature, length=12)
        record = {
            "key": key,
            "tool": tool,
            "signature": signature[:400],
            "advice": advice[:600],
            "evidence": evidence[:600],
            "hits": 0,
            "learned_at": now_iso(),
        }
        for row in rows:
            if row.get("key") == key:
                row.update(
                    {
                        "advice": record["advice"],
                        "evidence": record["evidence"],
                        "hits": int(row.get("hits", 0)) + 1,
                    }
                )
                atomic_write_json(self.lessons_path, rows)
                return row
        rows.append(record)
        atomic_write_json(self.lessons_path, rows[-200:])
        return record

    def advice_for(self, tool: str, error: str) -> list[str]:
        """Return advice from stored lessons that match this failure."""
        if not error:
            return []
        error_tokens = set(tokens(error))
        out: list[str] = []
        for row in self.lessons():
            if row.get("tool") not in {tool, "*", ""}:
                continue
            signature_tokens = set(tokens(row.get("signature", "")))
            if not signature_tokens:
                continue
            overlap = len(signature_tokens & error_tokens) / len(signature_tokens)
            if overlap >= 0.5:
                out.append(row.get("advice", ""))
        return [a for a in out if a][:3]

    # -- plan reuse -----------------------------------------------------
    def plans(self) -> dict:
        return read_json(self.plans_path, default={}) or {}

    def remember_plan(self, request: str, plan: dict) -> None:
        plans = self.plans()
        key = digest(*keywords(request, 8), length=12)
        plans[key] = {
            "request": request[:300],
            "keywords": keywords(request, 10),
            "plan": plan,
            "at": now_iso(),
            "reused": plans.get(key, {}).get("reused", 0),
        }
        atomic_write_json(self.plans_path, plans)

    def recall_plan(self, request: str, threshold: float = 0.6) -> dict | None:
        wanted = set(keywords(request, 10))
        if not wanted:
            return None
        best: tuple[float, dict] | None = None
        for entry in self.plans().values():
            have = set(entry.get("keywords", []))
            if not have:
                continue
            jaccard = len(wanted & have) / len(wanted | have)
            if jaccard >= threshold and (best is None or jaccard > best[0]):
                best = (jaccard, entry)
        if not best:
            return None
        entry = best[1]
        plans = self.plans()
        key = digest(*entry.get("keywords", []), length=12)
        if key in plans:
            plans[key]["reused"] = int(entry.get("reused", 0)) + 1
            atomic_write_json(self.plans_path, plans)
        return entry.get("plan")

    # -- convenience ----------------------------------------------------
    def summarise(self, request: str, limit: int = 4) -> list[str]:
        return [f"[{i.kind}] {i.text}" for i in self.search(request, limit=limit)]

    def stats(self) -> dict:
        return {
            "items": len(read_jsonl(self.items_path)),
            "lessons": len(self.lessons()),
            "plans": len(self.plans()),
            "path": str(self.root),
        }

        scored.sort(key=lambda i: (i.score, i.at), reverse=True)
        return scored[:limit]
