"""
Load process environment for local and server runs.

**Where to put your OpenAI API key**

1. Create a file named ``.env`` in the ``backend/`` directory (the same
   directory that contains ``main.py``).
2. Add a single line (no spaces around ``=`` unless the value is quoted):
   ``OPENAI_API_KEY=sk-...`` — paste your secret key from the OpenAI dashboard
   after the equals sign. Do not add quotes around the value unless the key
   itself contains special characters.
3. Restart the API process after changing ``.env``.

You can also set ``OPENAI_API_KEY`` in your shell or in your host’s secret
manager (preferred in production). Values already in the real environment are
not overwritten by the ``.env`` file (see ``load_dotenv`` behavior).

**Security:** Never commit real API keys. Keep ``.env`` out of version control
(see the project ``.gitignore``).
"""

from pathlib import Path

from dotenv import load_dotenv

# backend/ — parent of this package (utils/ -> backend/)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"

# Populate os.environ from backend/.env (idempotent; safe to import more than once)
load_dotenv(_ENV_PATH)
