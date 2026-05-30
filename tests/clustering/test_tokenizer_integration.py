"""Integration test: full tokenizer pipeline persists TokenCluster +
TokenMapping rows for act_obs pairs and TokenAssigner resolves them.
"""

from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import select

from episodiq.clustering.constants import Params
from episodiq.clustering.tokenizer.assigner import TokenAssigner
from episodiq.clustering.tokenizer.pipeline import TokenizerPipeline
from episodiq.storage.postgres.models import (
    Cluster,
    Message,
    TokenCluster,
    TokenMapping,
    Trajectory,
    TrajectoryPath,
)
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    TokenClusterRepository,
    TokenMappingRepository,
)
from episodiq.utils import l2_normalize

DIM = 1024
PARAMS = Params(min_cluster_size=3, min_samples=2, umap_dims=10, umap_n_neighbors=4)


def _seeded_unit(seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    return l2_normalize(rng.randn(DIM).tolist())


def _perturbed(base: list[float], rng: np.random.RandomState) -> list[float]:
    v = np.array(base) + rng.randn(DIM) * 0.005
    return l2_normalize(v.tolist())


@pytest.fixture(autouse=True)
def reset_token_assigner_cache():
    TokenAssigner.invalidate()
    yield
    TokenAssigner.invalidate()


@pytest.mark.asyncio(loop_scope="session")
class TestTokenizerIntegration:
    """Seed clustered messages → build paths → run TokenizerPipeline →
    verify TokenCluster + TokenMapping rows are written and the assigner
    resolves a known pair.
    """

    async def test_pipeline_creates_act_obs_tokens(self, session_factory):
        traj_id = uuid4()
        # 3 action clusters × 3 observation clusters → 9 distinct pairs.
        a_bases = [_seeded_unit(seed) for seed in (1, 2, 3)]
        o_bases = [_seeded_unit(seed) for seed in (11, 12, 13)]
        n_per_cluster = 4
        n_pair_repeats = 4

        async with session_factory() as session:
            session.add(Trajectory(id=traj_id))
            await session.flush()

            rng = np.random.RandomState(0)
            action_msgs: list[list[Message]] = []
            obs_msgs: list[list[Message]] = []
            idx = 0
            for base in a_bases:
                bucket = []
                for _ in range(n_per_cluster):
                    m = Message(
                        trajectory_id=traj_id, role="assistant", content=[],
                        embedding=_perturbed(base, rng), category="text",
                        cluster_type="action", index=idx,
                    )
                    bucket.append(m)
                    session.add(m)
                    idx += 1
                action_msgs.append(bucket)
            for base in o_bases:
                bucket = []
                for _ in range(n_per_cluster):
                    m = Message(
                        trajectory_id=traj_id, role="user", content=[],
                        embedding=_perturbed(base, rng), category="text",
                        cluster_type="observation", index=idx,
                    )
                    bucket.append(m)
                    session.add(m)
                    idx += 1
                obs_msgs.append(bucket)
            await session.flush()

            action_clusters: list[Cluster] = []
            for i, bucket in enumerate(action_msgs):
                c = Cluster(type="action", category="text", label=f"a:text:{i}")
                session.add(c)
                await session.flush()
                action_clusters.append(c)
                for m in bucket:
                    m.cluster_id = c.id
            obs_clusters: list[Cluster] = []
            for i, bucket in enumerate(obs_msgs):
                c = Cluster(type="observation", category="text", label=f"o:text:{i}")
                session.add(c)
                await session.flush()
                obs_clusters.append(c)
                for m in bucket:
                    m.cluster_id = c.id
            await session.flush()

            # Build closed paths: every (action, observation) pair repeated so
            # the tokenizer has enough density to cluster.
            path_idx = 0
            for a_bucket in action_msgs:
                for o_bucket in obs_msgs:
                    for r in range(n_pair_repeats):
                        a_msg = a_bucket[r % n_per_cluster]
                        o_msg = o_bucket[r % n_per_cluster]
                        session.add(TrajectoryPath(
                            trajectory_id=traj_id,
                            from_observation_id=obs_msgs[0][0].id,
                            action_message_id=a_msg.id,
                            to_observation_id=o_msg.id,
                            trace=[],
                            trajectory_status="success",
                            index=path_idx,
                        ))
                        path_idx += 1
            await session.commit()

        pipeline = TokenizerPipeline(session_factory, params=PARAMS)
        result = await pipeline.run()
        assert len(result.labels) > 0, "tokenizer should label at least one act_obs"

        async with session_factory() as session:
            tcs = (await session.execute(select(TokenCluster))).scalars().all()
            tms = (await session.execute(select(TokenMapping))).scalars().all()
            assert len(tcs) >= 1
            # 9 distinct (a, o) pairs → 9 mapping rows.
            assert len(tms) == 9

            tc_ids = {tc.id for tc in tcs}
            for tm in tms:
                assert tm.token_cluster_id in tc_ids

            assigner = TokenAssigner(
                TokenMappingRepository(session),
                TokenClusterRepository(session),
                ClusterRepository(session),
            )
            ordinal = await assigner.assign(
                action_clusters[0].id, obs_clusters[0].id,
            )
            assert ordinal is not None
            assert ordinal in {int(tc.cluster_id) for tc in tcs}
