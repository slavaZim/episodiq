"""TrajectoryPathUpdater: rebuilds trajectory paths from message cluster labels.

Matches online BuildPathStep behavior: one row per observation, with the trace
built incrementally.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.api_adapters.base import CanonicalMessage
from episodiq.retrieval.path_state import PathStateCalculator
from episodiq.storage.postgres.repository import MessageRepository, TrajectoryPathRepository

logger = logging.getLogger(__name__)

WORKERS = 10


class TrajectoryPathUpdater:
    """Drop all trajectory paths, rebuild from message cluster labels.

    Creates one row per observation (matching online BuildPathStep), populating
    the alternating obs/action label trace.
    """

    def __init__(
        self,
        msg_repo: MessageRepository,
        path_repo: TrajectoryPathRepository,
        calc: PathStateCalculator,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        workers: int = WORKERS,
    ):
        self._msg_repo = msg_repo
        self._path_repo = path_repo
        self._calc = calc
        self._session_factory = session_factory
        self._workers = workers

    async def update(self) -> int:
        """Rebuild all trajectory paths. Returns total rows created."""
        await self._path_repo.delete_all()

        traj_ids = await self._msg_repo.get_distinct_trajectory_ids()
        logger.info("build_paths_start trajectories=%d", len(traj_ids))

        total = 0
        for i, tid in enumerate(traj_ids, 1):
            total += await self._build_trajectory(tid)
            if i % 100 == 0:
                logger.info("build_paths_progress %d/%d trajectories paths=%d", i, len(traj_ids), total)

        logger.info("build_paths_done trajectories=%d paths=%d", len(traj_ids), total)

        await self._path_repo.sync_trajectory_status()

        return total

    async def _build_trajectory(self, trajectory_id) -> int:
        """Build paths for a single trajectory. Returns rows created.

        Walks ``[obs, assistant, (tool|obs)*]`` segments. An assistant
        with N>1 ``tool_call`` blocks emits N paths sharing the same
        ``action_message_id`` and ``parallel_group`` (= the assistant
        ``index``); each path pairs with one tool response by matching
        ``tool_call_id``. Sequential paths (N≤1) emit one path.
        """
        rows = await self._msg_repo.get_trajectory_with_clusters(trajectory_id)
        msgs = [m for m in rows if m.role != "system"]
        if not msgs:
            return 0

        prev_path = None
        count = 0
        i = 0
        while i + 1 < len(msgs):
            from_obs = msgs[i]
            action = msgs[i + 1]
            if action.role != "assistant":
                i += 1
                continue

            canon = CanonicalMessage.from_db(action)
            n_calls = canon.tool_calls_count

            if n_calls <= 1:
                to_obs = msgs[i + 2] if i + 2 < len(msgs) else None
                trace = self._calc.granular_step(prev_path, from_obs.cluster_label)
                prev_path = await self._path_repo.create(
                    trajectory_id=trajectory_id,
                    from_observation_id=from_obs.id,
                    action_message_id=action.id,
                    to_observation_id=to_obs.id if to_obs else None,
                    trace=trace,
                )
                count += 1
                i += 2
                continue

            responses = msgs[i + 2 : i + 2 + n_calls]
            if len(responses) < n_calls:
                break
            resp_by_call_id: dict[str, object] = {}
            for resp in responses:
                resp_canon = CanonicalMessage.from_db(resp)
        
                for cid in resp_canon.tool_call_ids:
                    resp_by_call_id[str(cid)] = resp

            # Walk calls in assistant.content order; tokenizer resorts
            # within ``parallel_group`` by ordinal ASC before writing
            # ``trace_tokens``, so order chosen here is not load-bearing.
            base_prev = prev_path
            for tc in canon.tool_calls or []:
                resp = resp_by_call_id.get(str(tc.id))
                if resp is None:
                    continue
                trace = self._calc.granular_step(
                    base_prev, from_obs.cluster_label,
                )
                prev_path = await self._path_repo.create(
                    trajectory_id=trajectory_id,
                    from_observation_id=from_obs.id,
                    action_message_id=action.id,
                    to_observation_id=resp.id,
                    trace=trace,
                    parallel_group=action.index,
                )
                count += 1
            i += 1 + n_calls

        if i < len(msgs) and msgs[i].role != "assistant":
            last_obs = msgs[i]
            trace = self._calc.granular_step(prev_path, last_obs.cluster_label)
            await self._path_repo.create(
                trajectory_id=trajectory_id,
                from_observation_id=last_obs.id,
                action_message_id=None,
                trace=trace,
            )
            count += 1

        return count
