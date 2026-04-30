"""Shared fixtures for annotator tests."""

from uuid import uuid4

from episodiq.api_adapters.base import Usage
from tests.helpers import text_to_embedding
from tests.in_memory_repos import Message


class MockGenerator:
    """Fake Generator that returns pre-set responses."""

    def __init__(self, responses: list[str] | None = None, default: str = "annotated"):
        self._responses = iter(responses) if responses else None
        self._default = default
        self.total_usage = Usage(input_tokens=0, output_tokens=0)
        self.calls: list[tuple] = []

    async def generate(self, messages, *, max_tokens=256):
        self.calls.append((messages, max_tokens))
        if self._responses:
            return next(self._responses)
        return self._default


async def make_cluster(repo, msg_repo, type, category, label,
                       n_messages=3, dim=50, content=None):
    """Create a cluster with messages and embeddings in in-memory repos."""
    cluster = await repo.create(type=type, category=category, label=label)
    msg_repo.add_cluster(cluster)
    tid = uuid4()
    role = "user" if type == "observation" else "assistant"
    for i in range(n_messages):
        msg_content = content or [{"type": "text", "text": f"example {label} {i}"}]
        msg = Message(
            id=uuid4(), trajectory_id=tid, role=role,
            content=msg_content, index=i,
            embedding=text_to_embedding(f"{label}-{i}", dim),
            cluster_id=cluster.id, category=category, cluster_type=type,
        )
        msg_repo.add_message(msg)
    return cluster
