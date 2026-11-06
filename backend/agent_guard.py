"""Spend caps and rate limit for the code-writing agent (docs/coding-agent-plan.md: built in milestone 1).

A ledger of every agent run (one JSON line each: time, cost, tokens, tools, outcome; no IP, no text)
drives daily and monthly USD caps. The per-IP rate limit lives in memory.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class AgentGuard:
    def __init__(self, data_dir: str | os.PathLike, *, per_hour: int = 10,
                 daily_usd: float = 5.0, monthly_usd: float = 50.0):
        self.ledger = Path(data_dir) / "usage.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.per_hour, self.daily_usd, self.monthly_usd = per_hour, daily_usd, monthly_usd
        # ponytail: per-IP windows in memory; a restart resets them, the USD caps (from the ledger) do not
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._day, self._month = self._today()
        self.spent_day = self.spent_month = 0.0
        self.answers_day = 0
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                self._count(row)

    @staticmethod
    def _today() -> tuple[str, str]:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")

    def _roll(self) -> None:
        day, month = self._today()
        if day != self._day:
            self._day, self.spent_day, self.answers_day = day, 0.0, 0
        if month != self._month:
            self._month, self.spent_month = month, 0.0

    def _count(self, row: dict) -> None:
        ts = str(row.get("ts", ""))
        if ts[:7] == self._month:
            self.spent_month += float(row.get("usd", 0))
        if ts[:10] == self._day:
            self.spent_day += float(row.get("usd", 0))
            self.answers_day += 1

    def check(self, client: str) -> Optional[tuple[str, str]]:
        """None when the request may run, else (error code, message for the developer)."""
        with self._lock:
            self._roll()
            if self.spent_month >= self.monthly_usd:
                return "monthly_cap", "The agent has reached this month's budget. The search answers below still work."
            if self.spent_day >= self.daily_usd:
                return "daily_cap", "The agent has reached today's budget; it resets at 00:00 UTC. The search answers below still work."
            window, now = self._hits[client], time.monotonic()
            while window and now - window[0] > 3600:
                window.popleft()
            if len(window) >= self.per_hour:
                return "rate_limited", f"You have asked {self.per_hour} questions in the last hour; please try again later."
            window.append(now)
            return None

    def record(self, *, kind: str, usd: float, outcome: str, seconds: float,
               tokens: Optional[dict] = None, tools: int = 0, models: Optional[list] = None) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, "outcome": outcome,
               "usd": round(usd, 6), "seconds": round(seconds, 2), "tools": tools,
               "tokens": tokens or {}, "models": models or []}
        with self._lock:
            self._roll()
            self._count(row)
            with self.ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def status(self) -> dict:
        with self._lock:
            self._roll()
            return {"today_usd": round(self.spent_day, 4), "month_usd": round(self.spent_month, 4),
                    "answers_today": self.answers_day, "daily_cap_usd": self.daily_usd,
                    "monthly_cap_usd": self.monthly_usd, "per_hour": self.per_hour}
