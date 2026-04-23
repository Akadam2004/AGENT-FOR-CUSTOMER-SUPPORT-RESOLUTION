"""
Simulated action execution for support workflows (refund, escalation).
"""

from typing import Final

ACTION_REFUND: Final = "refund"
ACTION_ESCALATE: Final = "escalate"
ACTION_NONE: Final = "none"


def process_refund(user_id: str) -> str:
    """Simulate processing a refund for the user."""
    _ = user_id
    return (
        "Refund simulation successful. If eligible, funds will appear per policy "
        "(typically 3-5 business days for payment reversals)."
    )


def escalate_ticket(user_id: str, message: str) -> str:
    """Simulate handing the ticket to a human agent."""
    preview = (message or "").strip().replace("\n", " ")[:200]
    return (
        f"Escalation recorded for user {user_id}. "
        f"A specialist will review: {preview!r}"
    )


def execute_action(action: str, user_id: str, message: str) -> str:
    """
    Route ``action`` to the appropriate handler.

    Supported: ``\"refund\"``, ``\"escalate\"``, ``\"none\"`` (case-insensitive).
    Unknown actions are treated like ``none``.
    """
    key = (action or "").strip().lower()
    if key == ACTION_REFUND:
        return process_refund(user_id)
    if key == ACTION_ESCALATE:
        return escalate_ticket(user_id, message)
    if key == ACTION_NONE:
        return "No automated action taken."
    return "No automated action taken."
