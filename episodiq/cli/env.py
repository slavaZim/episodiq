"""Shared .env loading utilities for CLI commands.

python-dotenv is an optional dependency — in production, environment variables
come from the container/orchestrator (Docker, k8s, systemd). dotenv is only
needed for local development.
"""

from pathlib import Path

from episodiq.config import get_config
from episodiq.config.config import Config


def _load_dotenv(env_file: Path) -> None:
    """Load .env file into os.environ.

    python-dotenv is a dev dependency — not available in production installs.
    Warns when missing so the user knows --env is ignored.
    """
    import typer

    try:
        from dotenv import load_dotenv
    except ImportError:
        typer.secho(
            f"Warning: python-dotenv not installed (install with episodiq[dev]), "
            f"--env {env_file} ignored.",
            fg=typer.colors.YELLOW,
        )
        return

    load_dotenv(env_file, override=False)


def load_config(env_file: Path) -> Config:
    """Load .env file and return Config.

    Guarantees correct order: dotenv first, then get_config().
    Silently skips dotenv if python-dotenv is not installed (production).
    """
    _load_dotenv(env_file)
    return get_config()
