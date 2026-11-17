"""Unit tests for the run cost helper (:func:`decode.observability.cost.run_cost_usd`)."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

from decode.config.settings import settings
from decode.observability.cost import run_cost_usd


@pytest.fixture
def rates(monkeypatch):
    """Configure $1/Mtok in and $2/Mtok out — round numbers keep the expected cost readable."""
    monkeypatch.setattr(settings, "llm_cost_input_usd_per_mtok", 1.0, raising=False)
    monkeypatch.setattr(settings, "llm_cost_output_usd_per_mtok", 2.0, raising=False)


# --- the run-level cost helper: the same honesty rule over a run's model responses (ADR-0022 §1) --


def _priced_response() -> ModelResponse:
    """A response the genai-prices catalog knows, so pydantic-ai can price it itself."""
    return ModelResponse(
        parts=[TextPart(content="ok")],
        usage=RequestUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        model_name="gemini-2.5-flash",
        provider_name="google-gla",
    )


def _unpriceable_response(*, input_tokens: int = 1_000_000, output_tokens: int = 500_000):
    """A response no catalog row covers — a scripted/self-hosted model id, the fallback's case."""
    return ModelResponse(
        parts=[TextPart(content="ok")],
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model_name="Qwen/Qwen3.6-35B-A3B-FP8",
    )


def test_run_cost_uses_the_catalog_price_when_pydantic_ai_can_price_every_response():
    """Branch 1: the catalog price pydantic-ai already computes, summed — no rates needed."""
    cost = run_cost_usd([_priced_response(), _priced_response()])

    assert cost is not None
    assert cost == pytest.approx(2 * float(_priced_response().cost().total_price))


def test_run_cost_falls_back_to_the_configured_rates_when_the_catalog_misses(rates):
    """Branch 2: one unpriceable response sends the WHOLE run to the rates over its own totals.

    All-or-nothing on purpose: summing a catalog price for some responses and nothing for the rest
    would silently understate the run, which is the one thing this module refuses to do.
    """
    cost = run_cost_usd([_priced_response(), _unpriceable_response()])

    assert cost == pytest.approx((2_000_000 * 1.0 + 1_500_000 * 2.0) / 1_000_000)


def test_run_cost_is_none_when_the_catalog_misses_and_no_rates_are_configured():
    """Branch 3: the Modal endpoint — GPU-seconds bill, so a token price would be a lie."""
    assert run_cost_usd([_unpriceable_response()]) is None


def test_run_cost_of_a_run_that_made_no_request_is_zero():
    """Nothing was requested, so nothing was spent — 0.0, not an unknown."""
    assert run_cost_usd([]) == 0.0
