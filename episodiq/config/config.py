import os
from dataclasses import dataclass

from .embedder_config import EmbedderConfig
from .retrieval_config import RetrievalConfig


@dataclass(frozen=True)
class AnalyticsConfig:
    """TransitionAnalyzer tuning parameters."""

    loop_threshold: int
    low_entropy: float
    high_entropy: float


@dataclass(frozen=True)
class Config:
    """Central configuration for Episodiq."""

    # Database
    database_url: str | None
    db_host: str | None
    db_port: str | None
    db_name: str | None
    db_user: str | None
    db_password: str | None

    # Timeouts
    process_input_timeout: float
    postprocess_timeout: float

    # Embedder service
    embedder: EmbedderConfig

    # Embedding dimensions (written to DB)
    message_dims: int

    # Analytics
    analytics: AnalyticsConfig

    # Retrieval (MinHash Jaccard prefilter)
    retrieval: RetrievalConfig

    # Clustering
    merge_threshold: float
    api_concurrency: int

    # When True, exclude each trajectory's initial observation (the task
    # prompt) from observation clustering.
    skip_initial_observation: bool

    # Logging
    log_level: str
    log_format: str
    log_file: str | None

    @property
    def has_database(self) -> bool:
        return bool(self.database_url) or all([self.db_host, self.db_name, self.db_user])

    @classmethod
    def from_env(cls) -> "Config":
        # Defaults from clustering.annotator.constants (inlined to avoid circular import)
        _MERGE_THRESHOLD = 0.9
        _API_CONCURRENCY = 8

        return cls(
            # Database
            database_url=os.getenv("EPISODIQ_DATABASE_URL"),
            db_host=os.getenv("EPISODIQ_DB_HOST"),
            db_port=os.getenv("EPISODIQ_DB_PORT", "5432"),
            db_name=os.getenv("EPISODIQ_DB_NAME"),
            db_user=os.getenv("EPISODIQ_DB_USER"),
            db_password=os.getenv("EPISODIQ_DB_PASSWORD", ""),
            # Timeouts
            process_input_timeout=float(os.getenv("EPISODIQ_PROCESS_INPUT_TIMEOUT", "15.0")),
            postprocess_timeout=float(os.getenv("EPISODIQ_POSTPROCESS_TIMEOUT", "120.0")),
            # Embedder service
            embedder=EmbedderConfig.from_env(),
            # Embedding dimensions
            message_dims=int(os.getenv("EPISODIQ_MESSAGE_DIMS", "1024")),
            # Analytics
            analytics=AnalyticsConfig(
                loop_threshold=int(os.getenv("EPISODIQ_LOOP_THRESHOLD", "2")),
                low_entropy=float(os.getenv("EPISODIQ_LOW_ENTROPY", "0.5")),
                high_entropy=float(os.getenv("EPISODIQ_HIGH_ENTROPY", "2.5")),
            ),
            # Retrieval
            retrieval=RetrievalConfig.from_env(),
            # Clustering
            merge_threshold=float(os.getenv(
                "EPISODIQ_MERGE_THRESHOLD",
                str(_MERGE_THRESHOLD),
            )),
            api_concurrency=int(os.getenv(
                "EPISODIQ_API_CONCURRENCY",
                str(_API_CONCURRENCY),
            )),
            skip_initial_observation=os.getenv(
                "EPISODIQ_SKIP_INITIAL_OBSERVATION", "true",
            ).lower() == "true",
            # Logging
            log_level=os.getenv("EPISODIQ_LOG_LEVEL", "info"),
            log_format=os.getenv("EPISODIQ_LOG_FORMAT", "json"),
            log_file=os.getenv("EPISODIQ_LOG_FILE"),
        )

    def get_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


def get_config() -> Config:
    return Config.from_env()
