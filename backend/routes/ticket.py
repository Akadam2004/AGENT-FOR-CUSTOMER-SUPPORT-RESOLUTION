"""
Ticket creation and support workflow endpoints.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.orchestrator import process_ticket

router = APIRouter()


class TicketCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="End-user or account identifier")
    message: str = Field(..., min_length=1, description="User message to process")


@router.post("/ticket")
async def create_ticket(body: TicketCreateRequest) -> dict[str, str]:
    """Accept a support ticket and run it through the agent orchestrator."""
    return await process_ticket(user_id=body.user_id, message=body.message)
