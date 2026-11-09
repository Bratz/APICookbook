"""BaNCS API knowledge, read through the run's proxy (APICookbook fork F2a).

Three read-only tools backed by cookbook-api's knowledge base, the same one its code agent uses:
``search_endpoints``, ``get_endpoint`` and ``get_body_schema``. They call the run's proxy
(``BANCS_PROXY_URL``, e.g. ``http://10.0.0.49:7601/r/<token>``) under ``/_kb/``; the proxy answers
only while the run is alive. With no proxy configured each tool says so instead of failing.
"""

from __future__ import annotations

import json
from urllib.parse import quote

import httpx
from pydantic_ai import RunContext

from decode.agent.deps import AgentDeps
from decode.config.settings import settings

SEARCH_TOOL_NAME = "search_endpoints"
ENDPOINT_TOOL_NAME = "get_endpoint"
BODY_SCHEMA_TOOL_NAME = "get_body_schema"
_NOT_CONFIGURED = "No BaNCS proxy is configured for this run (BANCS_PROXY_URL is empty)."


async def _kb(path: str, params: dict[str, str] | None = None) -> str:
    base = settings.bancs_proxy_url.rstrip("/")
    if not base:
        return _NOT_CONFIGURED
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base}/_kb/{path}", params=params)
    except httpx.HTTPError as exc:
        return f"The API knowledge service did not answer ({type(exc).__name__}); try again."
    if response.status_code != 200:
        return f"The API knowledge service answered {response.status_code}: {response.text[:300]}"
    return json.dumps(response.json(), ensure_ascii=False)


async def search_endpoints(ctx: RunContext[AgentDeps], query: str) -> str:
    """Find BaNCS API operations by meaning. Use a short task phrase ("cancel cash block").

    Returns the top 8 with method, path, description and a relevance score; judge relevance yourself.
    """
    return await _kb("search", {"q": query})


async def get_endpoint(ctx: RunContext[AgentDeps], name: str) -> str:
    """Method, path template, parameters and whether a body is needed, for one operation by name.

    Use a name search_endpoints returned. ``url_template`` is already the run's proxy address.
    """
    return await _kb(f"endpoint/{quote(name, safe='')}")


async def get_body_schema(ctx: RunContext[AgentDeps], name: str) -> str:
    """The JSON Schema of an operation's request body, with every definition inlined."""
    return await _kb(f"body-schema/{quote(name, safe='')}")


def instructions() -> str:
    """The BaNCS part of the system prompt; empty when the run has no proxy."""
    base = settings.bancs_proxy_url.rstrip("/")
    if not base:
        return ""
    return f"""You are working for a visitor of the APICookbook developer portal: they describe a task with the
TCS BaNCS banking APIs, and you write code for it, run it in this sandbox, and fix it until it works.

First decide what the visitor wants:
- Only the TCS BaNCS APIs and code that calls them. If the message is about anything else, decline in
  one sentence (say what you can help with) and finish, without calling any tool.
- A QUESTION about the API (which operation to use, its parameters, body fields, what it returns):
  answer from search_endpoints, get_endpoint and get_body_schema alone. Name the operation, its method
  and path, and the fields that matter. Write no code and make no bank calls, then finish.
- A TASK to do or build something with the API: follow everything below.

How to work on a task:
- Find the operations with search_endpoints, then get_endpoint for each one you use, and get_body_schema
  before writing any request body. Use only operations these tools returned.
- Every URL your code calls must be the `url_template` of an operation get_endpoint returned, with only
  its {{placeholders}} filled in. Never type a path yourself or guess a similar one (`/customer/detail`
  is not `/customer/details`). Before you run your code, check each URL in it against those templates.
- Write the code in the language the task asks for (default: Python with `requests`), save it in the
  workspace, and run it. Read the output and fix the code until it works. You are not done until
  your code has really called the bank: a script you did not run is not an answer.
- Never ask the visitor anything (do not call ask_user): nobody can answer during a run. The visitor
  only approves or rejects your writes.
- When the task names no customer, account, loan or other record, find a REAL one yourself with GET
  calls and use it; say which one you picked and how. Good places to find real records: the fund
  transfers list (it carries real account and customer references), the customer list (customer
  IDs), a customer's account list, and the loan product list. Never use placeholders like 12345 or
  TEST.
- For a write, use the real record you found and realistic values, and MAKE the call: you never need
  permission in advance, because the visitor approves or rejects every write before it is sent.
- Finish with: what the code does, the file name(s), how to run it, and what running it returned.

The bank, from this sandbox:
- Your code reaches the bank ONLY through this base URL: {base}/Core . Append each operation's path
  template and fill its {{placeholders}}. No other host is reachable, and there is no internet.
- The proxy adds the entity, userId and languageCode headers the bank needs. Send
  `Content-Type: application/json` with a body. No Authorization header.
- GET requests run at once. Every POST, PUT, PATCH or DELETE changes shared demo data, so it WAITS
  until the visitor approves it (up to 2 minutes). Only make a write the task needs, one at a time.
  If it comes back 403 with "was not sent to the bank", the visitor said no: do not retry it; say
  what it would have done and finish.
- This is a shared demo bank. Never loop over writes, and never write data the task did not ask for."""
