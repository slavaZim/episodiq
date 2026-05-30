from pathlib import Path

import typer
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from episodiq.cli.env import load_config

app = typer.Typer()


def get_alembic_config() -> Config:
    """Build Alembic config programmatically.

    Points script_location to the migrations directory inside the installed
    package so `episodiq db migrate` works from any working directory.
    """
    import episodiq.storage.postgres.migrations as _migrations_pkg

    migrations_dir = str(Path(_migrations_pkg.__file__).resolve().parent)

    cfg = Config()
    cfg.set_main_option("script_location", migrations_dir)
    return cfg


@app.command()
def create(
    env: Path = typer.Option(
        Path(".env"),
        "--env",
        help="Path to .env file",
    ),
) -> None:
    """Create the database."""
    cfg = load_config(env)
    database_url = cfg.get_database_url()

    db_name = database_url.rsplit("/", 1)[-1]
    base_url = database_url.rsplit("/", 1)[0]

    maintenance_url = f"{base_url}/postgres".replace("+asyncpg", "+psycopg")
    engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": db_name},
        )
        if result.fetchone():
            typer.echo(f"Database '{db_name}' already exists")
        else:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            typer.echo(f"Database '{db_name}' created")


@app.command()
def migrate(
    env: Path = typer.Option(
        Path(".env"),
        "--env",
        help="Path to .env file",
    ),
) -> None:
    """Run database migrations."""
    load_config(env)  # ensure env is loaded for alembic
    alembic_cfg = get_alembic_config()
    command.upgrade(alembic_cfg, "head")
    typer.echo("Migrations applied")


@app.command()
def init(
    env: Path = typer.Option(
        Path(".env"),
        "--env",
        help="Path to .env file",
    ),
) -> None:
    """Create database and run migrations."""
    cfg = load_config(env)
    # Pass config-derived URL directly to avoid double load_config
    database_url = cfg.get_database_url()

    db_name = database_url.rsplit("/", 1)[-1]
    base_url = database_url.rsplit("/", 1)[0]

    maintenance_url = f"{base_url}/postgres".replace("+asyncpg", "+psycopg")
    engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": db_name},
        )
        if result.fetchone():
            typer.echo(f"Database '{db_name}' already exists")
        else:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            typer.echo(f"Database '{db_name}' created")

    engine.dispose()

    alembic_cfg = get_alembic_config()
    command.upgrade(alembic_cfg, "head")
    typer.echo("Migrations applied")
