"""
Post-response decision engine: confidence and resolve vs escalate using Ollama.
"""

import json
import os
import re
from typing import Any, Dict, Optional

import httpx
import utils.env  # noqa: F401  — load backend/.env
from tools.actions import ACTION_ESCALATE, ACTION_NONE, ACTION_REFUND

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
REQUEST_TIMEOUT_SECONDS = 45.0

# Aligns with evaluator prompt: "confident" for automation when clearly supported
_CONFIDENT_BILLING_THRESHOLD = 0.8

_DEFAULT_RESULT: Dict[str, Any] = {
    "confidence": 0.0,
    "decision": "escalate",
    "action": ACTION_ESCALATE,
    "reason": "Evaluation could not be completed; defaulting to escalation.",
}


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = _strip_code_fence(raw)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


def _normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    decision_raw = str(data.get("decision", "escalate")).lower().strip()
    if decision_raw not in ("resolve", "escalate"):
        decision_raw = "escalate"

    reason = str(data.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided."

    return {
        "confidence": confidence,
        "decision": decision_raw,
        "reason": reason,
    }


def _select_action(category: str, decision: str, confidence: float) -> str:
    """
    Server-side action routing (authoritative, not from the LLM).

    Rules (first match):
    1. Billing + confident → refund
    2. decision = escalate → escalate
    3. else → none
    """
    if (category or "").strip() == "Billing" and confidence > _CONFIDENT_BILLING_THRESHOLD:
        return ACTION_REFUND
    if (decision or "").strip().lower() == "escalate":
        return ACTION_ESCALATE
    return ACTION_NONE


def _build_evaluation(
    data: Dict[str, Any], category: str
) -> Dict[str, Any]:
    base = _normalize_result(data)
    base["action"] = _select_action(
        category, base["decision"], base["confidence"]
    )
    return base


def _build_prompt(
    message: str,
    category: str,
    response: str,
    context: str,
) -> str:
    ctx = context.strip() if context.strip() else "(none provided)"
    return f"""You are a strict AI quality evaluator.

Be conservative:
- Only give confidence > 0.8 if the answer is clearly supported by context
- If vague, incomplete, or generic → keep confidence below 0.6
- If unsure → escalate

Message: {message.strip()}
Category: {category.strip()}
Context: {ctx}
Response: {response.strip()}

Return JSON (only the JSON object, no markdown, no other text):
{{
  "confidence": float (0 to 1),
  "decision": "resolve" or "escalate",
  "reason": "short explanation"
}}
"""


async def evaluate_response(
    message: str,
    category: str,
    response: str,
    context: str,
) -> Dict[str, Any]:
    """
    Call Ollama for confidence, decision, reason; add ``action`` via routing rules.
    """
    prompt = _build_prompt(message, category, response, context)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            r = await client.post(OLLAMA_ENDPOINT, json=payload)
            r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return dict(_DEFAULT_RESULT)

    raw = str(data.get("response", "")).strip()
    if not raw:
        return dict(_DEFAULT_RESULT)

    parsed = _parse_json_object(raw)
    if not parsed:
        return {
            "confidence": 0.0,
            "decision": "escalate",
            "action": ACTION_ESCALATE,
            "reason": "Evaluator did not return valid JSON; escalating for safety.",
        }

    return _build_evaluation(parsed, category)
