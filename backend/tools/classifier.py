"""
Ticket classification utility using a local Ollama model.
"""

from typing import Optional

import httpx
import utils.env  # noqa: F401  — load backend/.env before reading environment values

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
REQUEST_TIMEOUT_SECONDS = 30.0
ALLOWED_CATEGORIES = {"Billing", "Technical", "Account", "General"}

CLASSIFICATION_PROMPT = """
You are a customer support classification system.

Classify the ticket into EXACTLY one of these categories:

1. Billing -> payment issues, refunds, charges, invoices
2. Technical -> bugs, crashes, system errors, app not working
3. Account -> login issues, password reset, account access, profile problems
4. General -> anything else

Rules:
- "forgot password", "cannot login", "reset password" -> Account
- Return ONLY the category name (no explanation)

Ticket:
{message}
"""


def _rule_based_category(message: str) -> Optional[str]:
    text = message.strip().lower()
    account_phrases = ("forgot password", "cannot login", "cant login", "reset password")
    if any(phrase in text for phrase in account_phrases):
        return "Account"
    return None


def _normalize_category(raw_text: str) -> str:
    text = raw_text.strip()
    for category in ALLOWED_CATEGORIES:
        if text.lower() == category.lower():
            return category
    return "General"


async def classify_ticket(message: str) -> str:
    """
    Classify a ticket into Billing, Technical, Account, or General.

    Falls back to "General" for uncertain/invalid outputs and request failures.
    """
    rule_match = _rule_based_category(message)
    if rule_match is not None:
        return rule_match

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": CLASSIFICATION_PROMPT.format(message=message.strip()),
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(OLLAMA_ENDPOINT, json=payload)
            response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return "General"

    return _normalize_category(str(data.get("response", "")))
