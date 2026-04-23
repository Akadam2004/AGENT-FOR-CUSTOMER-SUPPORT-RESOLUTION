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


class TicketResponse(BaseModel):
    user_id: str
    message: str
    category: str
    response: str
    confidence: float
    decision: str
    action: str
    action_result: str
    status: str
    reason: str
    past_tickets_count: int


@router.post("/ticket", response_model=TicketResponse)
async def create_ticket(body: TicketCreateRequest) -> TicketResponse:
    """Accept a support ticket and run it through the agent orchestrator."""
    return TicketResponse(
        **await process_ticket(user_id=body.user_id, message=body.message)
    )
