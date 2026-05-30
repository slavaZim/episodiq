from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

from episodiq.workflows.context import WorkflowContext

logger = structlog.stdlib.get_logger(__name__)


@dataclass
class StepResult:
    """Result of step execution."""

    passable: bool = False
    value: Any = None
    reason: str | None = None
    terminal: bool = False


class Step(ABC):
    """Base class for pipeline steps."""

    step_id: str  # Subclasses must set
    deferred: bool = False
    has_timeout: bool = True

    def __init__(self, ctx: WorkflowContext):
        self.ctx = ctx

    @abstractmethod
    async def exec(self) -> StepResult:
        """Execute step, mutate ctx, return result."""
        ...
