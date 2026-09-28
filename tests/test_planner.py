from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from semantic_lineage.agents.planner import OpenAIPlanner, PlannerError
from semantic_lineage.config import Settings


def create_planner(response_or_error: object) -> tuple[OpenAIPlanner, MagicMock]:
    planner = OpenAIPlanner(Settings(llm_api_key="test-key"))
    create = MagicMock()
    if isinstance(response_or_error, Exception):
        create.side_effect = response_or_error
    else:
        create.return_value = response_or_error
    planner._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return planner, create


def completed_response(output_text: str) -> SimpleNamespace:
    return SimpleNamespace(status="completed", output_text=output_text)


def test_openai_planner_constructs_strict_structured_intent_request() -> None:
    planner, create = create_planner(
        completed_response('{"intent": "discovery", "rationale": "find an asset"}')
    )

    decision = planner.classify_intent("Where is customer revenue?", "catalog context")

    assert decision.intent == "discovery"
    assert decision.rationale == "find an asset"
    request = create.call_args.kwargs
    assert request["model"] == "gpt-5.4-mini"
    assert request["text"]["format"] == {
        "type": "json_schema",
        "name": "intent_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["discovery", "lineage", "analysis"]},
                "rationale": {"type": "string"},
            },
            "required": ["intent", "rationale"],
            "additionalProperties": False,
        },
    }


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (completed_response("not json"), "invalid structured output"),
        (completed_response("[]"), "not a JSON object"),
        (completed_response('{"intent": "discovery"}'), "required field 'rationale'"),
        (
            completed_response('{"intent": "unrecognized", "rationale": "reason"}'),
            "unsupported intent",
        ),
        (completed_response(""), "no structured output"),
        (SimpleNamespace(status="incomplete", output_text=""), "did not complete"),
    ],
)
def test_openai_planner_converts_malformed_or_incomplete_output_to_planner_error(
    response: SimpleNamespace, message: str
) -> None:
    planner, _ = create_planner(response)

    with pytest.raises(PlannerError, match=message):
        planner.classify_intent("Where is customer revenue?", "catalog context")


def test_openai_planner_converts_connection_error_to_actionable_planner_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    planner, _ = create_planner(openai.APIConnectionError(request=request))

    with pytest.raises(PlannerError, match="Could not connect to the OpenAI API"):
        planner.classify_intent("Where is customer revenue?", "catalog context")


def test_openai_planner_converts_rate_limit_error_to_actionable_planner_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, request=request)
    planner, _ = create_planner(openai.RateLimitError("rate limited", response=response, body=None))

    with pytest.raises(PlannerError, match="rate limit"):
        planner.classify_intent("Where is customer revenue?", "catalog context")


def test_openai_planner_converts_other_api_errors_to_actionable_planner_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(401, request=request)
    planner, _ = create_planner(openai.APIStatusError("unauthorized", response=response, body=None))

    with pytest.raises(PlannerError, match="OpenAI request failed"):
        planner.classify_intent("Where is customer revenue?", "catalog context")
