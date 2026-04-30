"""Dead-end model inference: neighbor features → prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from episodiq.analytics.dead_end import extract_neighbor_features
from episodiq.analytics.transition_types import TrajectoryAnalytics
from episodiq.storage.postgres.models import TrajectoryPath


@dataclass
class DeadEndPrediction:
    probability: float
    confidence: float
    is_dead_end: bool


class DeadEndPredictor:
    """Load model bundle, extract features from path + analytics, predict."""

    def __init__(self, model_path: Path, threshold: float):
        self._path = model_path
        self._threshold = threshold
        self._pipeline = None

    @property
    def is_available(self) -> bool:
        return self._pipeline is not None

    def load(self) -> bool:
        """Load model bundle from disk. Returns False if file doesn't exist."""
        if not self._path.exists():
            return False
        bundle = joblib.load(self._path)
        self._pipeline = bundle["pipeline"]
        return True

    def predict(
        self,
        current_path: TrajectoryPath,
        analytics: TrajectoryAnalytics,
    ) -> DeadEndPrediction | None:
        """Extract features and predict. Returns None if insufficient data."""
        neighbor = extract_neighbor_features(current_path, analytics)
        if neighbor is None:
            return None

        X = np.array(neighbor).reshape(1, -1)
        proba = float(self._pipeline.predict_proba(X)[:, 1][0])

        return DeadEndPrediction(
            probability=proba,
            confidence=max(proba, 1.0 - proba),
            is_dead_end=proba >= self._threshold,
        )
