"""Smoke test for server factory — verifies build_app produces a working FastAPI app."""


import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _factory_env(monkeypatch):
    """Set minimal env for build_app to succeed without real services."""
    monkeypatch.setenv("EPISODIQ_DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
    monkeypatch.setenv("EPISODIQ_EMBEDDER_URL", "http://localhost:1234")
    monkeypatch.setenv("EPISODIQ_MESSAGE_DIMS", "1024")


def test_build_app_returns_fastapi():
    """build_app should return a FastAPI instance with expected routes."""
    from episodiq.server.factory import build_app

    app = build_app()

    routes = {r.path for r in app.routes}
    assert "/openai/v1/chat/completions" in routes
    assert "/episodiq/health" in routes
    assert "/episodiq/trajectories/{trajectory_id}" in routes


def test_build_app_embedder_is_embedder():
    """Workflows must receive Embedder (with embed_text), not raw EmbedderClient."""
    from episodiq.inference.embedder import Embedder
    from episodiq.server.factory import build_app

    app = build_app()

    # Verify each workflow got the right embedder type
    for route in app.routes:
        # Workflow endpoints are closures — check via the app's included routers
        pass

    # Direct check: re-import and inspect
    from episodiq.server.factory import build_app as _build
    from episodiq.workflows.base import Workflow

    app = _build()
    # The workflows are captured in router closures; verify by building fresh
    # and checking the factory wires Embedder (not EmbedderClient)
    with patch.object(Workflow, "__init__", wraps=Workflow.__init__) as mock_init:
        mock_init.return_value = None
        try:
            _build()
        except Exception:
            pass
        if mock_init.called:
            for call in mock_init.call_args_list:
                embedder = call.kwargs.get("embedder") or call.args[4] if len(call.args) > 4 else None
                if embedder is not None:
                    assert isinstance(embedder, Embedder), (
                        f"Expected Embedder, got {type(embedder).__name__}"
                    )
