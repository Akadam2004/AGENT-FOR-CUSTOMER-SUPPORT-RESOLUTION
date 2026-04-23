"""
Ticket and conversation orchestration (LLM, tools, handoffs).
"""

from typing import Any, Dict, List

from agents.decision import evaluate_response
from db.memory import get_user_history, save_ticket
from tools.actions import ACTION_NONE, execute_action
from tools.classifier import classify_ticket
from tools.llm import generate_response
from tools.retriever import get_relevant_context


def _category_summary_from_history(history: List[Dict[str, Any]]) -> Dict[str, int]:
    categories = [str(t.get("category") or "") for t in history]
    return {
        "Billing": categories.count("Billing"),
        "Technical": categories.count("Technical"),
        "Account": categories.count("Account"),
        "General": categories.count("General"),
    }


def _format_user_memory_for_llm(
    past_count: int,
    recent_tickets: List[Dict[str, Any]],
    category_summary: Dict[str, int],
) -> str:
    """
    Structured user history for the LLM (stronger than a single count line).

    Example shape:
        User history:
        - Total tickets: 5
        - Billing: 3, Technical: 1, Account: 1
        - Recent issues:
           1. Billing – charged twice
           ...
    """
    lines: List[str] = [
        "User history:",
        f"- Total tickets: {past_count}",
    ]

    breakdown = ", ".join(
        f"{name}: {count}"
        for name, count in category_summary.items()
        if count > 0
    )
    if breakdown:
        lines.append(f"- {breakdown}")
    elif past_count > 0:
        lines.append("- Categories: (uncategorized or empty labels in history)")

    lines.append("- Recent issues:")
    if recent_tickets:
        for i, t in enumerate(recent_tickets, start=1):
            cat = str(t.get("category") or "Unknown").strip()
            msg = str(t.get("message") or "").replace("\n", " ").strip()
            if len(msg) > 120:
                msg = msg[:117].rstrip() + "..."
            lines.append(f"   {i}. {cat} – {msg}")
    else:
        lines.append("   (none)")

    return "\n".join(lines)


async def process_ticket(user_id: str, message: str) -> dict[str, Any]:
    """
    Run the support pipeline for a user message and return a structured result.

    Pipeline: memory → classify → RAG+LLM → evaluate → act → persist → respond.
    """
    # 1) Load prior tickets for this user (SQLite) and build LLM "User history" block
    history = get_user_history(user_id)
    past_tickets_count = len(history)

    # get_user_history returns newest first; [:3] = three most recent prior tickets
    recent_tickets = history[:3]
    category_summary = _category_summary_from_history(history)
    memory_context = _format_user_memory_for_llm(
        past_tickets_count,
        recent_tickets,
        category_summary,
    )

    # 2) Classify, 3) retrieve knowledge chunks, 4) generate answer (Ollama)
    category = await classify_ticket(message)
    context_chunks = get_relevant_context(message)
    context = "\n\n".join(context_chunks)
    response = await generate_response(
        message,
        context,
        memory_context=memory_context,
    )
    # 5) Quality check + action routing (Ollama + rules)
    evaluation = await evaluate_response(message, category, response, context)
    decision = str(evaluation.get("decision", "escalate"))
    status = "resolved" if decision == "resolve" else "escalated"
    confidence = float(evaluation.get("confidence", 0.0))
    action = str(evaluation.get("action", ACTION_NONE))
    # 6) Simulated post-processing (refund / escalate / none)
    action_result = execute_action(action, user_id, message)

    # 7) Persist this ticket for future turns
    save_ticket(
        user_id=user_id,
        message=message,
        category=category,
        decision=decision,
        confidence=confidence,
    )

    return {
        "user_id": user_id,
        "message": message,
        "category": category,
        "response": response,
        "confidence": confidence,
        "decision": decision,
        "action": action,
        "action_result": action_result,
        "status": status,
        "reason": str(evaluation.get("reason", "")),
        "past_tickets_count": past_tickets_count,
    }
