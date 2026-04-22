"""
Route modules: each file defines an ``APIRouter``; `main` mounts them on the app.

To add a new area (e.g. `routes/users.py`):

1. Create ``router = APIRouter()`` in that module and register paths on it.
2. In `main`, ``from routes import users`` and ``app.include_router(users.router, prefix="...", tags=[...])``.
"""
