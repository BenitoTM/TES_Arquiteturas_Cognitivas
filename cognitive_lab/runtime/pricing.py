from __future__ import annotations

import os
from typing import Any


DEFAULT_TOKEN_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cached_input_tokens": 0,
    "reasoning_output_tokens": 0,
}

MODEL_PRICING = {
    "google/gemini-2.5-flash": {
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
        "cached_input_price_per_1m": 0.03,
        "source": "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing",
    },
    "gemini-2.5-flash": {
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
        "cached_input_price_per_1m": 0.03,
        "source": "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing",
    },
}


def zero_token_usage() -> dict[str, int]:
    return dict(DEFAULT_TOKEN_USAGE)


def extract_token_usage(usage_metadata: Any) -> dict[str, int]:
    usage = zero_token_usage()
    if not isinstance(usage_metadata, dict):
        return usage

    input_tokens = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0))
    output_tokens = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0))
    total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)

    input_details = usage_metadata.get("input_token_details") or {}
    output_details = usage_metadata.get("output_token_details") or {}

    usage["input_tokens"] = int(input_tokens or 0)
    usage["output_tokens"] = int(output_tokens or 0)
    usage["total_tokens"] = int(total_tokens or 0)
    usage["cached_input_tokens"] = int(
        input_details.get("cache_read", input_details.get("cached_tokens", 0)) or 0
    )
    usage["reasoning_output_tokens"] = int(
        output_details.get("reasoning", output_details.get("reasoning_tokens", 0)) or 0
    )
    return usage


def add_token_usage(total: dict[str, int], usage_metadata: Any) -> dict[str, int]:
    extracted = extract_token_usage(usage_metadata)
    return {
        key: int(total.get(key, 0)) + int(extracted.get(key, 0))
        for key in DEFAULT_TOKEN_USAGE
    }


def estimate_cost_usd(model: str, token_usage: dict[str, int]) -> dict[str, Any]:
    input_override = os.getenv("MODEL_INPUT_PRICE_PER_1M_TOKENS")
    output_override = os.getenv("MODEL_OUTPUT_PRICE_PER_1M_TOKENS")
    cached_input_override = os.getenv("MODEL_CACHED_INPUT_PRICE_PER_1M_TOKENS")
    source_override = os.getenv("MODEL_PRICING_SOURCE")

    pricing = MODEL_PRICING.get(model)
    pricing_source = None

    if input_override and output_override:
        pricing = {
            "input_price_per_1m": float(input_override),
            "output_price_per_1m": float(output_override),
            "cached_input_price_per_1m": float(cached_input_override or 0),
            "source": source_override or "environment override",
        }
        pricing_source = "env_override"
    elif pricing is not None:
        pricing_source = "builtin_table"

    if pricing is None:
        return {
            "estimated_cost_usd": None,
            "pricing_source": None,
            "input_price_per_1m": None,
            "output_price_per_1m": None,
            "cached_input_price_per_1m": None,
        }

    input_tokens = int(token_usage.get("input_tokens", 0))
    output_tokens = int(token_usage.get("output_tokens", 0))
    cached_input_tokens = int(token_usage.get("cached_input_tokens", 0))
    non_cached_input_tokens = max(0, input_tokens - cached_input_tokens)

    estimated_cost = (
        (non_cached_input_tokens / 1_000_000) * pricing["input_price_per_1m"]
        + (cached_input_tokens / 1_000_000) * pricing.get("cached_input_price_per_1m", 0)
        + (output_tokens / 1_000_000) * pricing["output_price_per_1m"]
    )

    return {
        "estimated_cost_usd": round(estimated_cost, 8),
        "pricing_source": pricing_source,
        "pricing_reference": pricing.get("source"),
        "input_price_per_1m": pricing["input_price_per_1m"],
        "output_price_per_1m": pricing["output_price_per_1m"],
        "cached_input_price_per_1m": pricing.get("cached_input_price_per_1m", 0),
    }
