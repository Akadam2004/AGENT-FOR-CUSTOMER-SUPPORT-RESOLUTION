"""
Application entry point for the AI-powered customer support agent API.
"""

import utils.env  # noqa: F401  — load backend/.env before routes/tools read secrets

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from routes.ticket import router as ticket_router
from tools.llm import LLMGenerationError

app = FastAPI(
    title="Agentic AI Customer Support Resolution",
    version="0.1.0",
    description="RAG + classification + decision engine + memory + action simulation.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Public HTTP API: versionable prefix (e.g. add v2 or /api/v1 as the app grows)
app.include_router(ticket_router, prefix="/api", tags=["tickets"])
# Future: app.include_router(another_module.router, prefix="/api", tags=[...])


@app.exception_handler(LLMGenerationError)
async def llm_generation_error_handler(_request: Request, exc: LLMGenerationError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
    )


@app.get("/")
def root() -> dict[str, str]:
    """Liveness check and service identification."""
    return {"message": "AI Support Agent Running"}
