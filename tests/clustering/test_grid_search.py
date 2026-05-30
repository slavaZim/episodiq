"""Tests for ClusterGridSearch and helpers."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np

from episodiq.clustering.constants import Params
from episodiq.clustering.grid_search import (
    ClusterGridSearch,
    GridJobSpec,
    GridSearchEntry,
    _select_winner,
    resolve_grid_jobs,
)
from tests.in_memory_repos import InMemoryMessageRepository, Message


def _seed_messages(repo: InMemoryMessageRepository, n: int, role: str, category: str, cluster_type: str = "action") -> None:
    rng = np.random.RandomState(42)
    tid = uuid4()
    for i in range(n):
        repo.add_message(Message(
            id=uuid4(), trajectory_id=tid, role=role, content=[], index=i,
            embedding=rng.randn(50).tolist(), category=category, cluster_type=cluster_type,
        ))


class TestSelectWinner:

    def test_picks_max_clusters_in_top_bucket(self):
        e1 = GridSearchEntry(Params(min_cluster_size=5), 2, 0.1, n_clusters=3, dbcv=0.5, entropy=0.9, score=0.40)
        e2 = GridSearchEntry(Params(min_cluster_size=10), 1, 0.05, n_clusters=5, dbcv=0.5, entropy=0.9, score=0.42)
        # Same bucket (0.40/0.05=8, 0.42/0.05=8)
        winner = _select_winner([e1, e2], max_clusters=None, bucket_size=0.05)
        assert winner.n_clusters == 5

    def test_respects_max_clusters(self):
        e1 = GridSearchEntry(Params(), 0, 0.0, n_clusters=3, dbcv=0.8, entropy=0.9, score=0.7)
        e2 = GridSearchEntry(Params(), 0, 0.0, n_clusters=10, dbcv=0.8, entropy=0.9, score=0.7)
        winner = _select_winner([e1, e2], max_clusters=5, bucket_size=0.05)
        assert winner.n_clusters == 3

    def test_returns_none_when_all_exceed_max(self):
        e1 = GridSearchEntry(Params(), 0, 0.0, n_clusters=10, dbcv=0.8, entropy=0.9, score=0.7)
        assert _select_winner([e1], max_clusters=5, bucket_size=0.05) is None

    def test_returns_none_for_empty(self):
        assert _select_winner([], max_clusters=None, bucket_size=0.05) is None


class TestResolveGridJobs:

    async def test_empty_specs_uses_defaults(self):
        """Empty specs → default grid jobs: action/observation × text + discovered tools."""
        repo = InMemoryMessageRepository()
        tid = uuid4()
        for cat in ["bash", "editor"]:
            repo.add_message(Message(
                id=uuid4(), trajectory_id=tid, role="assistant", content=[], index=0, category=cat, cluster_type="action",
            ))

        jobs = await resolve_grid_jobs(repo, [])

        types = {(j.type, j.category) for j, _, _ in jobs}
        assert ("action", "text") in types
        assert ("observation", "text") in types
        assert ("action", "bash") in types
        assert ("action", "editor") in types

    async def test_discovers_tool_categories(self):
        repo = InMemoryMessageRepository()
        tid = uuid4()
        for cat in ["text", "bash", "editor"]:
            repo.add_message(Message(
                id=uuid4(), trajectory_id=tid, role="assistant", content=[], index=0, category=cat, cluster_type="action",
            ))

        jobs = await resolve_grid_jobs(
            repo, [GridJobSpec(type="action", category="tool")],
        )

        assert {j.category for j, _, _ in jobs} == {"bash", "editor"}

    async def test_specific_category(self):
        repo = InMemoryMessageRepository()
        jobs = await resolve_grid_jobs(
            repo, [GridJobSpec(type="observation", category="text")],
        )
        assert len(jobs) == 1
        assert jobs[0][0].category == "text"


class TestClusterGridSearch:

    async def test_empty_messages_no_winners(self):
        repo = InMemoryMessageRepository()
        grid = ClusterGridSearch(
            repo,
            [GridJobSpec(type="action", category="text")],
        )
        winners, report = await grid.run()
        assert winners == []
        assert report.entries == {}

    async def test_too_few_messages_skips_params(self):
        repo = InMemoryMessageRepository()
        _seed_messages(repo, 3, "assistant", "text")

        grid = ClusterGridSearch(
            repo,
            [GridJobSpec(
                type="action", category="text",
                params_list=[Params(min_cluster_size=10)],
            )],
        )
        winners, report = await grid.run()

        assert winners == []
        assert report.entries["action:text"] == []

    async def test_produces_winner_and_report(self):
        repo = InMemoryMessageRepository()
        _seed_messages(repo, 30, "assistant", "text")

        cr = MagicMock(noise_count=2, n_clusters=3, dbcv=0.7, entropy=0.8,
                       labels=np.array([0]*10 + [1]*10 + [2]*8 + [-1]*2))

        with patch("episodiq.clustering.grid_search.Clusterer") as MC:
            MC.return_value.fit.return_value = cr
            grid = ClusterGridSearch(
                repo,
                [GridJobSpec(
                    type="action", category="text",
                    params_list=[Params(min_cluster_size=5)],
                )],
            )
            winners, report = await grid.run()

        assert len(winners) == 1
        assert winners[0].type == "action"
        assert winners[0].category == "text"
        assert winners[0].params == Params(min_cluster_size=5)
        assert len(report.entries["action:text"]) == 1
