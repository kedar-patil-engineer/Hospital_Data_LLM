# agent/eval_tracker.py
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


@dataclass
class EvalTracker:
    prompt_version: str = "v1"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    provider: str = ""
    model: str = ""
    events: list[dict] = field(default_factory=list)

    def record_llm(
        self,
        *,
        agent: str,
        latency_s: float,
        prompt_hash: str,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cost_usd_est: Optional[float] = None,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> None:
        self.events.append(
            {
                "type": "llm_call",
                "agent": agent,
                "latency_s": round(float(latency_s), 4),
                "prompt_hash": prompt_hash,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd_est": cost_usd_est,
                "status": status,
                "error": error,
                "ts": time.time(),
            }
        )

    def record_tool(
        self,
        *,
        tool: str,
        latency_s: float,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> None:
        self.events.append(
            {
                "type": "tool_call",
                "tool": tool,
                "latency_s": round(float(latency_s), 4),
                "status": status,
                "error": error,
                "ts": time.time(),
            }
        )

    def summary(self) -> Dict[str, Any]:
        total = time.time() - self.started_at
        llm_calls = [e for e in self.events if e["type"] == "llm_call"]
        tool_calls = [e for e in self.events if e["type"] == "tool_call"]

        tokens_in = sum((e.get("tokens_in") or 0) for e in llm_calls)
        tokens_out = sum((e.get("tokens_out") or 0) for e in llm_calls)
        cost = sum((e.get("cost_usd_est") or 0.0) for e in llm_calls)

        return {
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
            "total_latency_s": round(float(total), 4),
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "tokens_in": tokens_in or None,
            "tokens_out": tokens_out or None,
            "cost_usd_est": round(cost, 6) if cost else None,
            "events": self.events,
        }

    def persist_jsonl(self, path: str = "logs/eval_log.jsonl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.summary(), ensure_ascii=False) + "\n")
