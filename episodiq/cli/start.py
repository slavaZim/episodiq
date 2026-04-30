import os
from pathlib import Path

import typer

from episodiq.cli.env import _load_dotenv
from episodiq.config import get_config


def _check_database(database_url: str) -> None:
    """Verify database is reachable and migrations are applied."""
    from sqlalchemy import create_engine, text
    from alembic.script import ScriptDirectory

    from episodiq.cli.db import get_alembic_config

    sync_url = database_url.replace("+asyncpg", "+psycopg")
    engine = create_engine(sync_url)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(
            f"Database unreachable: {exc}\n"
            "Run 'episodiq db init' or check your database configuration."
        ) from exc

    # Check alembic version matches head
    try:
        alembic_cfg = get_alembic_config()
        script = ScriptDirectory.from_config(alembic_cfg)
        head = script.get_current_head()

        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            row = result.fetchone()

        if not row:
            raise SystemExit(
                "Database has no migrations applied.\n"
                "Run 'episodiq db migrate' to apply migrations."
            )

        if row[0] != head:
            raise SystemExit(
                f"Database migration mismatch: current={row[0]}, head={head}\n"
                "Run 'episodiq db migrate' to apply pending migrations."
            )
    except SystemExit:
        raise
    except Exception:
        # alembic.ini not found or other non-critical issue — skip check
        pass
    finally:
        engine.dispose()


def _check_embedder(url: str, api_key: str | None) -> None:
    """Verify embedder service is reachable."""
    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
        # Any response means the service is up — even 404 is fine
        typer.echo(f"  Embedder: reachable ({response.status_code})")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        typer.secho(
            f"  Embedder: unreachable at {url}",
            fg=typer.colors.YELLOW,
        )
        typer.echo("  (will retry on first request)")


def up(
    env: Path = typer.Option(
        Path(".env"),
        "--env",
        help="Path to .env file",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        help="Port to bind to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
    tail: bool = typer.Option(
        False,
        "--tail",
        help="Stream analytics log entries to stdout",
    ),
    analytics: bool = typer.Option(
        False,
        "--analytics",
        help="Enable online analytics",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        help="Log level (debug, info, warning, error)",
    ),
    log_format: str = typer.Option(
        None,
        "--log-format",
        help="Log format: json or pretty",
    ),
) -> None:
    """Start Episodiq proxy server.

    Loads configuration from .env file.
    System environment variables take precedence over .env file.
    """
    import uvicorn
    from episodiq.logging import configure_logging

    _load_dotenv(env)

    if analytics:
        os.environ["EPISODIQ_ANALYTICS"] = "1"
    if tail:
        os.environ["EPISODIQ_TAIL"] = "1"

    cfg = get_config()

    configure_logging(
        log_level=log_level or cfg.log_level,
        log_format=log_format or cfg.log_format,
        log_file=cfg.log_file,
    )

    # Validate required infrastructure
    errors = []
    if not cfg.has_database:
        errors.append(
            "Database not configured. Set EPISODIQ_DATABASE_URL or "
            "EPISODIQ_DB_HOST + EPISODIQ_DB_NAME + EPISODIQ_DB_USER"
        )
    if not cfg.embedder.url:
        errors.append("EPISODIQ_EMBEDDER_URL is not set")
    if errors:
        for e in errors:
            typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo("Starting Episodiq proxy server...")
    typer.echo(f"  Port: {port}")

    # Healthchecks
    db_url = cfg.get_database_url()
    typer.echo(f"  Database: {db_url.split('@')[-1]}")
    _check_database(db_url)
    typer.echo("  Database: OK")

    _check_embedder(cfg.embedder.url, cfg.embedder.api_key)

    if analytics:
        typer.echo("  Analytics: enabled")
    if tail:
        typer.echo("  Tail: enabled")

    uvicorn.run(
        "episodiq.server.factory:build_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        reload=reload,
    )
