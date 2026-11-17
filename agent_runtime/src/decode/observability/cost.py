"""Run cost in USD — what ``decode run --summary-json`` reports.

pydantic-ai computes the price for anything the genai-prices catalog knows (public Gemini and
OpenRouter models). When the catalog has no row, fall back to the configured per-million-token
rates. No rates configured → no cost, because a wrong number in a cost dashboard is worse than a
blank one.

That fallback is for a PER-TOKEN model the catalog missed (an OpenRouter slug), not for a self-hosted
endpoint: self-hosted serving bills GPU-seconds, so pricing its tokens would report a number nobody
is charged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from decimal import Decimal

from pydantic_ai.messages import ModelResponse

from decode.config.settings import settings

logger = logging.getLogger(__name__)

_TOKENS_PER_MILLION = 1_000_000


def run_cost_usd(responses: Sequence[ModelResponse]) -> float | None:
    """The USD cost of a whole run's model responses, or ``None`` when it cannot be priced honestly.

    This is what ``decode run --summary-json`` reports (ADR-0022 §1). Priority: the catalog price
    pydantic-ai computes
    (``ModelResponse.cost()``, genai-prices), then the configured per-million rates over the run's
    own token totals, then ``None``.

    All-or-nothing on the catalog: if ANY response has no catalog row, the whole run falls back to
    the rates. Summing a catalog price for some responses and zero for the rest would silently
    understate the run, and a wrong number in a cost dashboard is worse than a blank one. A run that
    made no request at all costs ``0.0`` — nothing was spent, and that is not an unknown.
    """
    catalog_total = Decimal(0)
    input_tokens = 0
    output_tokens = 0
    catalog_priced_everything = True
    for response in responses:
        input_tokens += response.usage.input_tokens or 0
        output_tokens += response.usage.output_tokens or 0
        price = _catalog_price(response)
        if price is None:
            catalog_priced_everything = False
        else:
            catalog_total += price
    if catalog_priced_everything:
        return float(catalog_total)

    input_rate = settings.llm_cost_input_usd_per_mtok
    output_rate = settings.llm_cost_output_usd_per_mtok
    if not input_rate and not output_rate:
        return None
    return (input_tokens * input_rate + output_tokens * output_rate) / _TOKENS_PER_MILLION


def _catalog_price(response: ModelResponse) -> Decimal | None:
    """What genai-prices charges for ``response``, or ``None`` when its catalog has no row.

    ``ModelResponse.cost()`` asserts on a missing model name and raises ``LookupError`` for an
    unknown ``(provider, model)`` — both are "the catalog cannot price this", never a run failure,
    so every exception maps to ``None`` and the caller's fallback decides.
    """
    if not response.model_name:
        return None
    try:
        return response.cost().total_price
    except Exception:
        logger.debug("genai-prices could not price model %r", response.model_name, exc_info=True)
        return None
