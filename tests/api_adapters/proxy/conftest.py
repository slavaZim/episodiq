from unittest.mock import MagicMock

import pytest

from episodiq.workflows import Workflow
from episodiq.workflows.steps import ForwardStep
from episodiq.workflows.steps.base import Step, StepResult
from episodiq.server.app import create_app


class DummyStep(Step):
    """Step that immediately returns passable=False to trigger fallback."""

    async def exec(self) -> StepResult:
        return StepResult(passable=False)


def create_proxy_workflow(api_adapter):
    """Create a simple proxy workflow for testing adapters."""
    return Workflow(
        api_adapter=api_adapter,
        steps=[DummyStep],
        fallback_step=ForwardStep,
        session_factory=MagicMock(),
        embedder=MagicMock(),
    )


@pytest.fixture
def proxy_app_factory():
    """Factory for creating proxy apps with given adapter."""
    def _factory(api_adapter):
        workflow = create_proxy_workflow(api_adapter)
        return create_app([workflow])
    return _factory
