"""
Expected behavior for two canonical support scenarios (Case 1 strong, Case 2 weak).

Ollama output is non-deterministic; we mock the /api/generate response for the
decision step so these tests are stable. They assert the contract we want when
the evaluator returns appropriate JSON.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.decision import evaluate_response
from agents.orchestrator import process_ticket


def _fake_ollama_client(json_in_response: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"response": json_in_response})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _ollama_model_payload(obj: dict) -> str:
    return json.dumps(obj)


@patch("agents.decision.httpx.AsyncClient")
@pytest.mark.asyncio
async def test_case1_strong_billing_charged_twice(mock_client_cls) -> None:
    """
    Case 1 (Strong): clear billing fact in knowledge — expect high confidence, resolve.
    We simulate the evaluator JSON Ollama should return in the ideal case.
    """
    payload = {
        "confidence": 0.88,
        "decision": "resolve",
        "reason": "Response aligns with company billing / double-charge policy.",
    }
    mock_client_cls.return_value = _fake_ollama_client(_ollama_model_payload(payload))

    out = await evaluate_response(
        message="I was charged twice",
        category="Billing",
        response="Per policy, double charges are refunded in 3-5 business days. ...",
        context="Double charges are refunded within 3-5 business days.",
    )

    assert out["decision"] == "resolve"
    assert out["confidence"] >= 0.8
    assert out["reason"]
    assert out["action"] == "refund"


@patch("agents.decision.httpx.AsyncClient")
@pytest.mark.asyncio
async def test_case2_weak_vague_app_behavior(mock_client_cls) -> None:
    """
    Case 2 (Weak): vague symptom — expect low confidence, escalate.
    """
    payload = {
        "confidence": 0.25,
        "decision": "escalate",
        "reason": "Issue is vague; insufficient detail to self-resolve.",
    }
    mock_client_cls.return_value = _fake_ollama_client(_ollama_model_payload(payload))

    out = await evaluate_response(
        message="My app behaves weird sometimes",
        category="Technical",
        response="Please try updating the app. If that does not help, we can look further.",
        context="(limited relevant context)",
    )

    assert out["decision"] == "escalate"
    assert out["confidence"] < 0.5
    assert out["reason"]
    assert out["action"] == "escalate"


@patch("agents.orchestrator.save_ticket")
@patch("agents.orchestrator.get_user_history")
@patch("agents.orchestrator.evaluate_response", new_callable=AsyncMock)
@patch("agents.orchestrator.generate_response", new_callable=AsyncMock)
@patch("agents.orchestrator.get_relevant_context")
@patch("agents.orchestrator.classify_ticket", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_orchestrator_status_from_decision(
    mock_classify,
    mock_relevant,
    mock_gen,
    mock_eval,
    mock_get_history,
    mock_save,
) -> None:
    """Orchestrator sets status=resolved when decision is resolve, else escalated."""
    mock_classify.return_value = "Billing"
    mock_relevant.return_value = ["Double charges are refunded within 3-5 business days."]
    mock_gen.return_value = "We will process your refund in 3-5 business days."
    mock_eval.return_value = {
        "confidence": 0.9,
        "decision": "resolve",
        "action": "refund",
        "reason": "ok",
    }
    mock_get_history.side_effect = [
        [],
        [{"id": 1, "user_id": "u1", "message": "prior", "category": "Billing"}],
    ]
    mock_classify.side_effect = ["Billing", "Technical"]

    result = await process_ticket("u1", "I was charged twice")
    assert result["status"] == "resolved"
    assert result["decision"] == "resolve"
    assert result["confidence"] >= 0.8
    assert result["past_tickets_count"] == 0
    assert result["action"] == "refund"
    assert "action_result" in result
    assert "Refund" in result["action_result"] or "refund" in result["action_result"].lower()
    mock_save.assert_called()

    mock_eval.return_value = {
        "confidence": 0.2,
        "decision": "escalate",
        "action": "escalate",
        "reason": "vague",
    }
    result2 = await process_ticket("u1", "My app behaves weird sometimes")
    assert result2["status"] == "escalated"
    assert result2["decision"] == "escalate"
    assert result2["confidence"] < 0.5
    assert result2["past_tickets_count"] == 1
    assert result2["action"] == "escalate"
    assert "Escalation" in result2["action_result"] or "escalat" in result2["action_result"].lower()
