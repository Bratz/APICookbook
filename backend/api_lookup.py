"""Read-only lookups over the BaNCS Swagger spec: search, one operation, its body schema, an area.

Plain functions over the knowledge base, no model. Build & run's agent reaches them through the
proxy's ``/_kb/*`` routes; the endpoint cards and the eval use them directly. (They began as the
retired code agent's tools: docs/decode-fork-plan.md, F6.)
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

from backend.config import settings

BASE_URL = settings.api_base_url.rstrip("/") + "/Core"  # the Swagger spec's basePath
MAX_TOOL_CHARS = 30_000  # ~8k tokens; decode caps tool output too (tools/truncate.py). 3 of 495 schemas exceed it.
SEARCH_RESULTS = 8

_common_params: dict[int, frozenset] = {}


def _swagger(kb, name: str) -> Optional[dict]:
    ep = kb.endpoints.get(name)
    return ep if ep and ep.get("source") == "swagger" else None


def _common(kb) -> frozenset:
    """Transport headers present on most calls (Accesstoken, ChannelType...): noise for a reader."""
    if id(kb) not in _common_params:
        counts = Counter(k for ep in kb.endpoints.values() if isinstance(ep.get("parameters"), dict) for k in ep["parameters"])
        _common_params[id(kb)] = frozenset(k for k, c in counts.items() if c > 0.3 * max(len(kb.endpoints), 1))
    return _common_params[id(kb)]


def search_endpoints(kb, rag, query: str) -> list[dict]:
    """The top Swagger operations for a task phrase, with a relevance score (semantic when RAG is on)."""
    hits: list[tuple[str, float]] = []
    if rag:
        for doc, score in rag.search(query, top_k=SEARCH_RESULTS * 3, min_score=0):
            name = doc.metadata.get("endpoint_name")
            if _swagger(kb, name):
                hits.append((name, round(score, 3)))
    else:  # RAG disabled: fuzzy name match
        hits = [(n, round(s / 100, 3)) for n, s in kb.search_endpoints(query) if _swagger(kb, n)]
    return [{"endpoint": n, "method": kb.endpoints[n]["method"], "path": kb.endpoints[n]["url"],
             "description": (kb.endpoints[n].get("description") or "")[:160],
             "area": kb.endpoints[n].get("category"), "score": s} for n, s in hits[:SEARCH_RESULTS]]


def get_endpoint(kb, name: str) -> dict:
    """Method, path template, parameters and whether a body is needed, for one Swagger operation."""
    ep = _swagger(kb, name)
    if not ep:
        return {"error": f"{name} is not a Swagger operation; use a name search_endpoints returned"}
    common = _common(kb)
    params = {k: {f: v.get(f) for f in ("in", "type", "required", "description") if v.get(f) is not None}
              for k, v in (ep.get("parameters") or {}).items()
              if isinstance(v, dict) and k not in common and v.get("in") != "body"}
    return {"endpoint": name, "method": ep["method"], "path": ep["url"], "url_template": BASE_URL + ep["url"],
            "description": ep.get("description"), "area": ep.get("category"), "parameters": params,
            "has_body": bool(ep.get("body_schema"))}


def get_body_schema(kb, name: str) -> Any:
    """The JSON Schema of the operation's request body, every definition inlined, cut to the size cap."""
    if not _swagger(kb, name):
        return {"error": f"{name} is not a Swagger operation"}
    schema = kb.resolved_body_schema(name)
    return compact(schema) if schema else "This operation takes no request body."


def compact(schema: dict) -> Any:
    """A schema small enough for a model: as is when it fits, else without prose, else cut with a note."""
    if len(json.dumps(schema)) <= MAX_TOOL_CHARS:
        return schema

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in ("description", "title", "example")}
        return [strip(v) for v in node] if isinstance(node, list) else node

    lean = strip(schema)
    text = json.dumps(lean)
    if len(text) <= MAX_TOOL_CHARS:
        return lean
    cut = MAX_TOOL_CHARS
    while True:  # the cut text is a JSON string, so escaping grows it: shrink until the whole result fits
        result = {"truncated": True, "note": "schema too large; field names and types only, cut at the limit",
                  "schema_start": text[:cut]}
        if len(json.dumps(result)) <= MAX_TOOL_CHARS:
            return result
        cut = int(cut * 0.9)
