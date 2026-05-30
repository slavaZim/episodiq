from episodiq.workflows.base import Workflow
from episodiq.workflows.steps import (
    TrajectoryStep,
    SaveInputStep,
    ForwardStep,
    SaveOutputStep,
    ProcessInputStep,
    ProcessOutputStep,
    BuildPathStep,
)


class LoggingPipeline(Workflow):
    """Pipeline: save trajectory + input, forward, save output, then defer embed/cluster/path."""

    def __init__(self, **kwargs):
        super().__init__(
            steps=[
                TrajectoryStep,
                SaveInputStep,
                ForwardStep,
                SaveOutputStep,
                ProcessInputStep,
                ProcessOutputStep,
                BuildPathStep,
            ],
            fallback_step=ForwardStep,
            **kwargs,
        )
