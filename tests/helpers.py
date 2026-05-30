"""Helper utilities for tests."""

import hashlib
import re
import struct
from typing import Callable
from uuid import UUID

import numpy as np
import pytest

from episodiq.utils import l2_normalize
from episodiq.workflows.context import WorkflowContext


def text_to_embedding(text: str, dim: int, *, normalize: bool = True, salt: str = "episodiq-test-v1") -> list[float]:
    """
    Deterministic test embedding: text -> embedding vector.

    - Stable across runs/machines (depends only on text + salt)
    - Not semantic; only 1-1 mapping property matters
    - Uses SHA256 in counter mode to generate deterministic floats in [-1, 1]
    """
    payload = (salt + "\n" + text).encode("utf-8")

    out = []
    counter = 0

    while len(out) < dim:
        h = hashlib.sha256(payload + b"\n" + str(counter).encode("ascii")).digest()
        counter += 1

        for (u32,) in struct.iter_unpack(">I", h):
            x = (u32 / 0xFFFFFFFF) * 2.0 - 1.0
            out.append(float(x))
            if len(out) >= dim:
                break

    if normalize:
        out = l2_normalize(out)

    return out


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def semantic_text_to_embedding(
    text: str,
    dim: int,
    *,
    normalize: bool = True,
    salt: str = "episodiq-test-v3",
) -> list[float]:
    """
    Deterministic semantic embedding that produces similar vectors for similar texts.

    - Each token generates a full vector by hashing token+position
    - All token vectors are averaged
    - Texts with overlapping tokens will have similar embeddings
    - Suitable for testing semantic retrieval
    """
    tokens = _TOKEN_RE.findall(text.lower())

    if not tokens:
        return [0.0] * dim

    token_vecs = []
    for t in tokens:
        # Generate dim numbers deterministically from token
        vec = []
        for i in range(dim):
            h = hashlib.blake2b((salt + "|" + t + "|" + str(i)).encode("utf-8"), digest_size=8).digest()
            # Convert to number in [-1, 1]
            val = int.from_bytes(h[:4], "big") / 0xFFFFFFFF * 2.0 - 1.0
            vec.append(val)
        token_vecs.append(vec)

    # Average all token vectors
    import numpy as np
    avg = np.mean(token_vecs, axis=0).tolist()

    if normalize:
        avg = l2_normalize(avg)

    return avg


@pytest.fixture
def capture_contexts(monkeypatch):
    """Capture all WorkflowContext instances by trajectory_id.

    Returns dict[UUID, WorkflowContext] where key is trajectory_id.
    """
    contexts = {}
    original_init = WorkflowContext.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._capture_dict = contexts

    def capturing_setattr(self, name, value):
        object.__setattr__(self, name, value)
        if name == "trajectory_id" and value is not None:
            self._capture_dict[value] = self

    monkeypatch.setattr(WorkflowContext, "__init__", capturing_init)
    monkeypatch.setattr(WorkflowContext, "__setattr__", capturing_setattr)
    return contexts


async def run_scenario(
    app, trajectory_id: UUID, messages: list[dict],
    *, capture: list | None = None, body_extra: dict | None = None,
):
    """Run  scenario with sequential requests.

    Each assistant message triggers a client POST with all
    preceding messages. The upstream mock matches requests by hashing body
    messages against precomputed keys.

    Args:
        app: FastAPI application (must be used with LifespanManager outside)
        trajectory_id: UUID for X-Trajectory-ID header
        messages: Flat conversation list; assistant messages are upstream responses
        capture: If provided, each upstream request body is appended to this list
        body_extra: Extra fields to merge into POST body (e.g. ``{"tools": [...]}``).

    Returns:
        Last response
    """
    import respx
    from httpx import Response as HttpxResponse, ASGITransport, AsyncClient
    import json

    def _clean_msg(msg):
        """Strip test-only keys (prefixed with _) from message dict."""
        return {k: v for k, v in msg.items() if not k.startswith("_")}

    def _make_response(msg, response_id):
        return HttpxResponse(200, json={
            "id": f"chatcmpl-{response_id}",
            "model": "gpt-4",
            "choices": [{"message": _clean_msg(msg)}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def _hash_messages(msgs):
        return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()

    # Hash engine: hash(messages_before_assistant) → response
    response_map = {}
    history = []
    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get("_internal"):
            response_map[_hash_messages(history)] = _make_response(msg, len(response_map))
        history.append(msg)


    def mock_response(request):
        body = json.loads(request.content)
        if capture is not None:
            capture.append(body)

        key = _hash_messages(body["messages"])
        if key in response_map:
            return response_map[key]

        return HttpxResponse(200, json={
            "id": "chatcmpl-fallback",
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
        })

    respx.post("https://test.local/chat").mock(side_effect=mock_response)

    # Send client requests: POST before each non-internal assistant message
    client_history = []
    last_response = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for msg in messages:
            if msg.get("_internal"):
                continue
            if msg.get("role") == "assistant":
                post_body = {"messages": client_history}
                if body_extra:
                    post_body.update(body_extra)
                last_response = await client.post(
                    "/test/v1/chat",
                    json=post_body,
                    headers={"X-Trajectory-ID": str(trajectory_id)},
                )
            client_history.append(msg)

    return last_response


def sequential_embedding_fn():
    """Create a function that returns sequential embeddings [1,2,3], [2,3,4], etc."""
    counter = [0]

    def fn(text: str, dim: int) -> list[float]:
        result = [float(counter[0] + j + 1) for j in range(dim)]
        counter[0] += 1
        return result

    return fn


class MockEmbedder:
    """Mock embedder for testing with configurable embedding function.

    The embedding_fn must accept (text, dim) -> list[float].
    Defaults to text_to_embedding.
    """

    def __init__(self, embedding_fn: Callable[[str, int], list[float]] = text_to_embedding, dims: int = 50):
        self.embedding_fn = embedding_fn
        self._dims = dims
        self.calls = []  # Track all calls for verification

    async def embed(self, texts: list[str], dims: int = 0) -> list[list[float]]:
        """Generate embeddings for texts using configured function."""
        self.calls.append(texts)
        return [self.embedding_fn(text, dims or self._dims) for text in texts]

    async def embed_text(self, text: str) -> list[float]:
        """Generate single embedding (used by annotator)."""
        self.calls.append(text)
        return self.embedding_fn(text, self._dims)


def make_embedder_fixture(url: str):
    """Create embedder client fixture for integration tests."""
    import pytest_asyncio
    from episodiq.config import EmbedderConfig
    from episodiq.inference import EmbedderClient

    @pytest_asyncio.fixture
    async def embedder():
        config = EmbedderConfig(
            url=url,
            chunk_size=512,
            batch_size=32,
        )
        client = EmbedderClient(config)
        await client.startup()
        yield client
        await client.shutdown()

    return embedder


def make_mock_embeddings_fixture(url: str, embedding_fn: Callable[[str, int], list[float]] = text_to_embedding):
    """Create respx mock fixture for embedder service."""
    import pytest
    import respx
    from httpx import Response

    @pytest.fixture
    def mock_embeddings():
        def generate_response(request):
            import json
            body = json.loads(request.content)
            content = body.get("content", [""])[0]
            dims = body.get("truncate", 0) or body.get("dimensions", 0)
            embedding = embedding_fn(content, dims)
            return Response(200, json=[{"embedding": embedding}])

        return respx.post(f"{url}/embedding").mock(side_effect=generate_response)

    return mock_embeddings


def make_distant_vector(base: list[float], min_distance: float) -> list[float]:
    """Create random unit vector at cosine distance >= min_distance from base.

    Args:
        base: Reference vector
        min_distance: Minimum cosine distance (0 = same, 1 = orthogonal, 2 = opposite)

    Returns:
        Unit vector at exactly min_distance from normalized base
    """
    base_arr = np.array(l2_normalize(base))

    # Random vector orthogonal to base
    random = np.random.randn(len(base))
    random = random - np.dot(random, base_arr) * base_arr
    random = np.array(l2_normalize(random.tolist()))

    # cos(theta) = 1 - distance
    cos_theta = 1 - min_distance
    sin_theta = np.sqrt(1 - cos_theta**2)

    result = cos_theta * base_arr + sin_theta * random
    return result.tolist()
