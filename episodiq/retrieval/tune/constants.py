"""Defaults for the Optuna-driven retrieval sweep. Single source of
truth; CLI flags reference these directly so overrides stay readable.
"""

DEFAULT_N_TRIALS = 100
# Early stopping: stop the study once this many trials in a row finish
# without improving best_seen. 0 disables.
DEFAULT_EARLY_STOP_PATIENCE = 30
# Outer parallelism: concurrent trials within one metric study via
# ``study.optimize(n_jobs=...)``. Threading; Numba JIT inside score_pair
# releases the GIL so this scales.
DEFAULT_N_JOBS = 4
# Inner parallelism: concurrent snapshots inside a single trial via a
# ThreadPoolExecutor. Each snapshot evaluation is independent.
DEFAULT_N_WORKERS = 4
DEFAULT_WINDOW_GRID: tuple[int, ...] = (10, 14)
DEFAULT_EVAL_MIN_STEP = 50
DEFAULT_AGGREGATION_GRID: tuple[str, ...] = ("mean", "min_distance")
DEFAULT_OPTUNA_SEED = 0
DEFAULT_MULTIVARIATE = False

# Continuous Optuna search ranges.
DEFAULT_PREFETCH_N_UNIQ_RANGE: tuple[int, int] = (50, 300)
DEFAULT_JACCARD_N_UNIQ_RANGE: tuple[int, int] = (40, 200)
DEFAULT_TOP_K_RANGE: tuple[int, int] = (5, 50)
DEFAULT_LAM_RANGE: tuple[float, float] = (0.0, 5.0)
DEFAULT_GAP_OPEN_RANGE: tuple[float, float] = (0.0, 3.0)
DEFAULT_GAP_EXTEND_RANGE: tuple[float, float] = (0.0, 2.0)
# Gaussian-only sigma. Larger σ → flatter penalty (closer to const).
DEFAULT_SIGMA_RANGE: tuple[float, float] = (0.5, 3.0)
DEFAULT_PENALTY_SHAPE_CHOICES: tuple[str, ...] = ("lin", "const", "quad", "gauss")
