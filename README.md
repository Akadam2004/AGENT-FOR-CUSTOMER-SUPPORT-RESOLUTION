# Agentic AI Customer Support Resolution System

Backend API for an **LLM-driven customer support agent** that classifies issues, answers from company knowledge (RAG), scores confidence, stores ticket history, and runs simulated follow-up actions (e.g. refund, escalation).

**Repository:** [github.com/Akadam2004/AGENT-FOR-CUSTOMER-SUPPORT-RESOLUTION](https://github.com/Akadam2004/AGENT-FOR-CUSTOMER-SUPPORT-RESOLUTION)

---

## Overview

Support teams drown in repetitive tickets and need consistent, policy-grounded answers. This project demonstrates an **agentic pipeline**: each message is **classified**, **retrieved** against a small knowledge base, **answered** by a local or remote LLM (Ollama), **evaluated** for quality and next-step **actions**, and **remembered** in SQLite—so the model can see prior context for the same user.

---

## Features

| Area | What it does |
|------|----------------|
| **Ticket classification** | Maps each message to a category (Billing, Technical, Account, General) via a local Ollama prompt + light rule hints. |
| **RAG (knowledge retrieval)** | Chunks `data/knowledge.txt`, embeds with `sentence-transformers` (`all-MiniLM-L6-v2`), and searches with **FAISS**; top chunks are passed to the LLM. |
| **Decision engine** | Ollama evaluates the reply vs. context: **confidence** (0–1), **resolve vs escalate**, plus **action** rules (e.g. refund for confident Billing). |
| **Memory layer (SQLite)** | `db/memory.py` stores tickets; orchestrator injects a structured **user history** block into the LLM. |
| **Action execution** | Simulated **refund** / **escalate** / **none** with human-readable `action_result` strings. |

---

## Architecture (high level)

```text
Client  →  FastAPI (`main.py`)  →  /api/ticket
                ↓
         Orchestrator: memory → classify → RAG → LLM → evaluate → act → save
                ↓
    Classifier, Retriever+embeddings, LLM, Decision, Actions, SQLite
```

- **API layer** (`routes/`): HTTP models and the ticket route.  
- **Agent layer** (`agents/`): Orchestration, decision logic, and action routing.  
- **Tools** (`tools/`): Ollama clients (LLM, classifier, decision), FAISS retriever, action simulators.  
- **Data** (`db/`, `data/`): SQLite ticket store and `knowledge.txt` for RAG.  
- **Config** (`utils/env.py`, `.env`): Ollama URL/model and env loading.

A fuller **directory map** is in [Project structure](#project-structure) below.

---

## Tech stack

- **Python 3.9+** (3.10+ recommended)
- **FastAPI** + **Uvicorn**
- **httpx** (async HTTP to Ollama)
- **sentence-transformers** + **faiss-cpu** (embeddings and vector search)
- **python-dotenv** (optional `.env` in `backend/`)
- **SQLite** (stdlib)
- **Ollama** (local LLM: `llama3` by default, embedding model on disk under `backend/models/`)

---

## Project structure

```text
PROJECT_0/   (or repo root)
├── README.md
├── .gitignore
└── backend/
    ├── main.py                 # FastAPI app, CORS not required for local
    ├── requirements.txt
    ├── pytest.ini
    ├── .env                    # (create locally; not committed) Ollama settings
    ├── data/
    │   └── knowledge.txt       # RAG source text
    ├── models/
    │   └── all-MiniLM-L6-v2/   # local embedder (optional; copy or download)
    ├── db/
    │   ├── memory.py           # SQLite: tickets
    │   └── memory.db           # created at runtime (gitignored)
    ├── agents/
    │   ├── orchestrator.py     # end-to-end pipeline
    │   └── decision.py         # Ollama evaluation + action selection
    ├── routes/
    │   ├── __init__.py
    │   └── ticket.py           # POST /ticket (mounted under /api in main)
    ├── tools/
    │   ├── llm.py              # reply generation
    │   ├── classifier.py       # category
    │   ├── retriever.py        # FAISS + sentence-transformers
    │   ├── actions.py          # execute_action, refund, escalate
    │   └── ...
    ├── utils/
    │   └── env.py              # load_dotenv for backend/
    └── tests/
        └── test_support_scenarios.py
```

---

## Setup (step by step)

### 1. Clone and enter the backend

```bash
git clone https://github.com/Akadam2004/AGENT-FOR-CUSTOMER-SUPPORT-RESOLUTION.git
cd AGENT-FOR-CUSTOMER-SUPPORT-RESOLUTION/backend
```

### 2. Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

*First run:* downloading `sentence-transformers` / torch may take a few minutes. If the embedding model is not under `models/all-MiniLM-L6-v2`, allow a one-time download or copy from the Hugging Face cache (see retriever in code).

### 4. Environment (optional)

Create `backend/.env` if you need to override defaults:

```env
OLLAMA_ENDPOINT=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=llama3
```

### 5. Ollama (LLM + classifier + decision all use the same Ollama server)

Install [Ollama](https://ollama.com), then:

```bash
ollama serve                # if not already running
ollama pull llama3           # or your chosen chat model; match OLLAMA_MODEL
```

### 6. Start the API

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive API docs.

---

## Quick start (recap)

1. `cd backend && python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `ollama serve` and `ollama pull llama3`
4. `uvicorn main:app --reload`

Run tests (optional): `pytest -q` from `backend/`.

---

## API usage: `POST /api/ticket`

| Item | Value |
|------|--------|
| **URL** | `http://127.0.0.1:8000/api/ticket` |
| **Method** | `POST` |
| **Body** | JSON with `user_id` and `message` |

### Example request

```bash
curl -s -X POST "http://127.0.0.1:8000/api/ticket" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-42",
    "message": "I was charged twice on my last invoice"
  }'
```

### Example response (illustrative; text varies with Ollama)

```json
{
  "user_id": "user-42",
  "message": "I was charged twice on my last invoice",
  "category": "Billing",
  "response": "…assistant reply grounded in knowledge…",
  "confidence": 0.85,
  "decision": "resolve",
  "action": "refund",
  "action_result": "Refund simulation successful. If eligible, funds will appear per policy (typically 3-5 business days for payment reversals).",
  "status": "resolved",
  "reason": "…evaluator reason…",
  "past_tickets_count": 0
}
```

- **`status`**: `resolved` if the decision engine returns `resolve`, else `escalated`.  
- **`action` / `action_result`**: simulated side-effect of the chosen action.  
- **`past_tickets_count`**: number of **prior** stored tickets for this `user_id` before the current one.

---

## Future improvements

- Authentication / rate limits for production.
- Real payment and ticketing integrations instead of simulated actions.
- Async SQLite or a managed DB; migrations.
- Optional OpenTelemetry / structured logging.
- CI (GitHub Actions) with cached models and a pinned Ollama in tests.
- Stronger RAG: chunking strategy, re-ranking, and dynamic knowledge updates.
- Bilingual prompts and policy versioning.

---

## License

Add a `LICENSE` file if you open-source this repo for recruiters.

---

*Built for learning and portfolio demos. Swap Ollama for another backend by adapting `tools/llm.py` and related callers.*
