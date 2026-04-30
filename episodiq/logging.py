"""Structured logging configuration with structlog."""

import logging

import structlog


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = None,
) -> None:
    """Configure structlog with JSON or pretty renderer.

    Uses stdlib integration via ProcessorFormatter so existing
    logging.getLogger() calls automatically get structured formatting.
    Context variables (e.g. trajectory_id) are merged into every log line.

    When log_file is set, logs are written to both stdout and the file.
    File always uses JSON format for machine readability.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", key="ts"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "pretty":
        console_renderer = structlog.dev.ConsoleRenderer()
    else:
        console_renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    root = logging.getLogger()
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    if log_file:
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(json_formatter)
        root.addHandler(file_handler)

    root.setLevel(getattr(logging, log_level.upper()))

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
