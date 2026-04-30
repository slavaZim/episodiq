"""CLI command for dead-end model training."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.analytics.dead_end import DEFAULT_THRESHOLD
from episodiq.analytics.dead_end.train import CONCURRENCY
from episodiq.cli.env import _load_dotenv
from episodiq.config import get_config

app = typer.Typer()
console = Console()


@app.command()
def train(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output model path (default: EPISODIQ_DEAD_END_MODEL or dead_end.joblib)"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test split ratio"),
    threshold: float = typer.Option(DEFAULT_THRESHOLD, "--threshold", help="Dead-end probability threshold"),
    min_trace: int = typer.Option(5, "--min-trace", help="Min trace length"),
    workers: int = typer.Option(CONCURRENCY, "--workers", "-w", help="Concurrency for feature extraction"),
    eval_mode: bool = typer.Option(False, "--eval", help="Run evaluation on test split"),
    save_eval: Path | None = typer.Option(None, "--save-eval", help="Save per-step walk eval CSV (requires --eval)"),
    save_samples: Path | None = typer.Option(None, "--save-samples", help="Save extracted samples to file"),
    load_samples: Path | None = typer.Option(None, "--load-samples", help="Load samples from file (skip extraction)"),
) -> None:
    """Train dead-end prediction model from trajectory data."""
    if save_eval and not eval_mode:
        console.print("[red]--save-eval requires --eval[/red]")
        raise typer.Exit(1)

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from episodiq.analytics.dead_end.train import DeadEndTrainer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    if output is None:
        output = Path(config.analytics.dead_end_model)
    engine = create_async_engine(config.get_database_url(), poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _run():
        from episodiq.storage.postgres.repository import TrajectoryPathRepository

        async with session_factory() as session:
            path_repo = TrajectoryPathRepository(session)
            trainer = DeadEndTrainer(
                path_repo=path_repo,
                session_factory=session_factory,
                analytics_config=config.analytics,
                test_size=test_size,
                threshold=threshold,
                min_trace=min_trace,
                concurrency=workers,
                save_samples=save_samples,
                load_samples=load_samples,
            )
            result = await trainer.run(eval=eval_mode)
            trainer.save(output)

        console.print(f"\nTrain: {result.n_train_samples} ({result.n_train_trajectories} traj)")
        console.print(f"Test:  {result.n_test_samples} ({result.n_test_trajectories} traj)")
        console.print(f"Features: {result.feature_shape[0]} x {result.feature_shape[1]}")

        if result.walk:
            w = result.walk
            n_fail = w.n_detected + w.n_missed
            n_succ = sum(1 for r in w.trajectories if r.status == "success")
            precision = w.n_detected / (w.n_detected + w.n_false_positive) * 100 if (w.n_detected + w.n_false_positive) > 0 else 0
            recall = w.detection_rate * 100

            table = Table(title="Walk Evaluation")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            table.add_row("Precision", f"{precision:.1f}%")
            table.add_row("Recall", f"{recall:.1f}%")
            table.add_row("Detected / total failures", f"{w.n_detected}/{n_fail}")
            table.add_row("False positives", f"{w.n_false_positive}/{n_succ}" if n_succ else str(w.n_false_positive))
            table.add_row("Avg turns remaining", f"{w.avg_turns_remaining:.1f}" if w.avg_turns_remaining else "N/A")
            console.print(table)

            if save_eval:
                w.save_csv(save_eval)
                console.print(f"[bold green]Eval CSV saved to {save_eval}[/bold green]")

        console.print(f"\n[bold green]Model saved to {output}[/bold green]")

    asyncio.run(_run())
