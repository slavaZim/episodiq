"""Dead-end model training: neighbor features → LogReg."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import async_sessionmaker

from episodiq.analytics.dead_end import extract_neighbor_features
from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.config.config import AnalyticsConfig
from episodiq.storage.postgres.repository import TrajectoryPathRepository

logger = logging.getLogger(__name__)

MIN_TRACE_LEN = 5
CONCURRENCY = 10


@dataclass
class TrainingSample:
    trajectory_id: str
    status: str
    features: list[float]
    trace: list[str]
    index: int = 0


@dataclass
class WalkTrajectoryResult:
    trajectory_id: str
    status: str
    total_steps: int
    flagged_at: int | None
    turns_remaining: int | None


@dataclass
class WalkStepResult:
    trajectory_id: str
    index: int
    status: str
    prob: float


@dataclass
class WalkResult:
    trajectories: list[WalkTrajectoryResult]
    steps: list[WalkStepResult]
    n_detected: int
    n_missed: int
    n_false_positive: int
    avg_turns_remaining: float | None
    detection_rate: float

    def save_csv(self, path: Path) -> None:
        """Save per-step walk evaluation to CSV for offline analysis."""
        import csv

        fieldnames = ["id", "traj_id", "index", "status", "prob"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, step in enumerate(self.steps):
                writer.writerow({
                    "id": i,
                    "traj_id": step.trajectory_id,
                    "index": step.index,
                    "status": step.status,
                    "prob": f"{step.prob:.6f}",
                })


@dataclass
class ClassificationResult:
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


@dataclass
class TrainResult:
    n_train_samples: int
    n_test_samples: int
    n_train_trajectories: int
    n_test_trajectories: int
    feature_shape: tuple[int, int]
    classification: ClassificationResult | None = None
    walk: WalkResult | None = None


class DeadEndTrainer:
    """Owns the full lifecycle: extract → split → train → eval → save."""

    def __init__(
        self,
        *,
        path_repo: TrajectoryPathRepository,
        session_factory: async_sessionmaker | None = None,
        analytics_config: AnalyticsConfig | None = None,
        test_size: float = 0.2,
        threshold: float,
        seed: int = 42,
        min_trace: int = MIN_TRACE_LEN,
        concurrency: int = CONCURRENCY,
        save_samples: Path | None = None,
        load_samples: Path | None = None,
    ):
        self._path_repo = path_repo
        self._session_factory = session_factory
        self._analytics_config = analytics_config
        self._test_size = test_size
        self._threshold = threshold
        self._seed = seed
        self._min_trace = min_trace
        self._concurrency = concurrency
        self._save_samples = save_samples
        self._load_samples = load_samples

        # Populated after run()
        self._pipeline = None

    async def run(self, *, eval: bool = False) -> TrainResult:
        """Extract samples, split, train, optionally evaluate."""
        if self._load_samples:
            samples = _load_samples_file(self._load_samples)
            logger.info("loaded_samples path=%s n=%d", self._load_samples, len(samples))
        else:
            samples = await self._extract_samples()

        if self._save_samples:
            _save_samples_file(samples, self._save_samples)
            logger.info("saved_samples path=%s n=%d", self._save_samples, len(samples))

        train_samples, test_samples = self._split(samples)

        self._train(train_samples)

        classification = None
        walk = None
        if eval and test_samples:
            classification = self._eval_classification(test_samples)
            walk = self._eval_walk(test_samples)

        return TrainResult(
            n_train_samples=len(train_samples),
            n_test_samples=len(test_samples),
            n_train_trajectories=len({s.trajectory_id for s in train_samples}),
            n_test_trajectories=len({s.trajectory_id for s in test_samples}),
            feature_shape=self._last_feature_shape,
            classification=classification,
            walk=walk,
        )

    def save(self, path: Path) -> None:
        """Save trained model bundle to disk."""
        if self._pipeline is None:
            raise RuntimeError("No trained model. Call run() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline}, path)

    async def _extract_samples(self) -> list[TrainingSample]:
        """Load paths and extract feature vectors concurrently."""
        all_paths = await self._path_repo.get_completed(require_embed=True)

        all_paths = [
            p for p in all_paths
            if p.trajectory and p.trajectory.status in ("success", "failure")
        ]

        logger.info("extract_samples_start n_paths=%d", len(all_paths))

        sem = asyncio.Semaphore(self._concurrency)
        total = len(all_paths)
        done = 0

        async def _extract(path):
            nonlocal done
            async with sem:
                trace = path.trace or []
                if len(trace) < self._min_trace:
                    return None

                if self._session_factory:
                    async with self._session_factory() as session:
                        analyzer = TransitionAnalyzer(path_repo=TrajectoryPathRepository(session), config=self._analytics_config)
                        analytics = await analyzer.analyze(path)
                else:
                    analyzer = TransitionAnalyzer(path_repo=self._path_repo, config=self._analytics_config)
                    analytics = await analyzer.analyze(path)

                features = extract_neighbor_features(path, analytics)
                if features is None:
                    return None

                done += 1
                if done % 500 == 0 or done == total:
                    logger.info("extract_samples_progress %d/%d", done, total)

                return TrainingSample(
                    trajectory_id=str(path.trajectory_id),
                    status=path.trajectory.status,
                    features=features,
                    trace=trace,
                    index=path.index or 0,
                )

        raw = await asyncio.gather(*[_extract(p) for p in all_paths])
        samples = [r for r in raw if r is not None]
        logger.info("extract_samples_done samples=%d skipped=%d", len(samples), total - len(samples))
        return samples

    def _split(self, samples: list[TrainingSample]) -> tuple[list[TrainingSample], list[TrainingSample]]:
        """Split by trajectory_id."""
        traj_ids = sorted({s.trajectory_id for s in samples})
        random.seed(self._seed)
        random.shuffle(traj_ids)
        split = int(len(traj_ids) * (1.0 - self._test_size))
        train_tids = set(traj_ids[:split])

        train = [s for s in samples if s.trajectory_id in train_tids]
        test = [s for s in samples if s.trajectory_id not in train_tids]
        return train, test

    def _train(self, train_samples: list[TrainingSample]) -> None:
        """Fit LogReg on neighbor features."""
        labels = np.array([1 if s.status == "failure" else 0 for s in train_samples])

        X = np.array([s.features for s in train_samples])
        self._last_feature_shape = X.shape

        self._pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=self._seed),
        )
        self._pipeline.fit(X, labels)

    def _eval_walk(self, test_samples: list[TrainingSample]) -> WalkResult:
        """Walk each test trajectory path-by-path, predict at each step."""
        by_traj: dict[str, list[TrainingSample]] = {}
        for s in test_samples:
            by_traj.setdefault(s.trajectory_id, []).append(s)

        for paths in by_traj.values():
            paths.sort(key=lambda s: s.index)

        results: list[WalkTrajectoryResult] = []
        all_steps: list[WalkStepResult] = []

        for tid, paths in by_traj.items():
            status = paths[0].status
            total_steps = len(paths)
            flagged_at = None

            for i, sample in enumerate(paths):
                proba = self._predict_proba(sample)
                all_steps.append(WalkStepResult(
                    trajectory_id=tid,
                    index=i,
                    status=status,
                    prob=proba,
                ))
                if flagged_at is None and proba >= self._threshold:
                    flagged_at = i

            turns_remaining = (total_steps - flagged_at) if flagged_at is not None else None
            results.append(WalkTrajectoryResult(
                trajectory_id=tid,
                status=status,
                total_steps=total_steps,
                flagged_at=flagged_at,
                turns_remaining=turns_remaining,
            ))

        failure_results = [r for r in results if r.status == "failure"]
        success_results = [r for r in results if r.status == "success"]

        n_detected = sum(1 for r in failure_results if r.flagged_at is not None)
        n_missed = sum(1 for r in failure_results if r.flagged_at is None)
        n_false_positive = sum(1 for r in success_results if r.flagged_at is not None)

        detected_turns = [r.turns_remaining for r in failure_results if r.turns_remaining is not None]
        avg_turns = float(np.mean(detected_turns)) if detected_turns else None
        detection_rate = n_detected / len(failure_results) if failure_results else 0.0

        return WalkResult(
            trajectories=results,
            steps=all_steps,
            n_detected=n_detected,
            n_missed=n_missed,
            n_false_positive=n_false_positive,
            avg_turns_remaining=avg_turns,
            detection_rate=detection_rate,
        )

    def _eval_classification(self, test_samples: list[TrainingSample]) -> ClassificationResult:
        """Per-sample classification on test set."""
        labels = np.array([1 if s.status == "failure" else 0 for s in test_samples])
        X_test = np.array([s.features for s in test_samples])

        y_proba = self._pipeline.predict_proba(X_test)[:, 1]
        y_pred = self._pipeline.predict(X_test)

        return ClassificationResult(y_test=labels, y_pred=y_pred, y_proba=y_proba)

    def _predict_proba(self, sample: TrainingSample) -> float:
        """Predict dead-end probability for a single sample."""
        X = np.array(sample.features).reshape(1, -1)
        return float(self._pipeline.predict_proba(X)[:, 1][0])





# --- Pure functions ---


def _save_samples_file(samples: list[TrainingSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump([{
        "trajectory_id": s.trajectory_id,
        "status": s.status,
        "features": s.features,
        "trace": s.trace,
        "index": s.index,
    } for s in samples], path)


def _load_samples_file(path: Path) -> list[TrainingSample]:
    raw = joblib.load(path)
    return [TrainingSample(**d) for d in raw]
