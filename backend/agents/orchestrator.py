"""
Ticket and conversation orchestration (LLM, tools, handoffs).
"""

from tools.llm import generate_response


async def process_ticket(user_id: str, message: str) -> dict[str, str]:
    """
    Run the support pipeline for a user message and return a structured result.
    """
    response = await generate_response(message)
    return {
        "user_id": user_id,
        "message": message,
        "response": response,
        "status": "processed",
    }
