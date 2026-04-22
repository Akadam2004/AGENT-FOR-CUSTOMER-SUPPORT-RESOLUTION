"""
LLM integration for the support agent using a local Ollama server.
"""

import os

import httpx
import utils.env  # noqa: F401  — load backend/.env before reading environment values

SYSTEM_PROMPT = (
    "You are a professional customer support assistant. Respond clearly and helpfully."
)
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
REQUEST_TIMEOUT_SECONDS = 60.0


class LLMGenerationError(Exception):
    """Raised when the model response cannot be produced (config, network, or API)."""


async def generate_response(message: str) -> str:
    """
    Generate a support reply for the ticket message using Ollama.

    Returns assistant text only. Raises ``LLMGenerationError`` on failures.
    """
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Customer ticket message:\n"
        f"{message.strip()}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(OLLAMA_ENDPOINT, json=payload)
            response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException as exc:
        raise LLMGenerationError("The local Ollama request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise LLMGenerationError(
            f"Ollama returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise LLMGenerationError(
            "Could not connect to Ollama. Ensure it is running on localhost:11434."
        ) from exc
    except ValueError as exc:
        raise LLMGenerationError("Ollama returned invalid JSON.") from exc

    text = str(data.get("response", "")).strip()
    if not text:
        raise LLMGenerationError("Ollama returned an empty response.")
    return text
