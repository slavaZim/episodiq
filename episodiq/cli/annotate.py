"""CLI commands for cluster annotation."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.api_adapters.anthropic import AnthropicConfig, AnthropicMessagesAdapter
from episodiq.api_adapters.base import ApiAdapter, ApiAdapterConfig
from episodiq.api_adapters.openai import OpenAICompletionsAdapter, OpenAIConfig
from episodiq.cli.env import _load_dotenv
from episodiq.clustering.annotator.generator import (
    AnthropicMessagesGenerator,
    Generator,
    OpenAICompletionsGenerator,
)
from episodiq.clustering.annotator.annotator import AnnotatingJobSpec
from episodiq.clustering.annotator.constants import API_CONCURRENCY
from episodiq.clustering.annotator.pipeline import AnnotationPipeline
from episodiq.clustering.annotator.summarizer import MapReduceSummarizer
from episodiq.config import get_config
from episodiq.inference.embedder import Embedder, EmbedderClient

app = typer.Typer()
console = Console()


_ADAPTER_REGISTRY: dict[str, tuple[type[ApiAdapterConfig], type[ApiAdapter], type[Generator]]] = {
    "openai": (OpenAIConfig, OpenAICompletionsAdapter, OpenAICompletionsGenerator),
    "anthropic": (AnthropicConfig, AnthropicMessagesAdapter, AnthropicMessagesGenerator),
}


def _make_session_factory(database_url: str) -> async_sessionmaker:
    engine = create_async_engine(database_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


def _create_adapter(adapter_id: str) -> ApiAdapter:
    entry = _ADAPTER_REGISTRY.get(adapter_id)
    if not entry:
        available = ", ".join(_ADAPTER_REGISTRY.keys())
        raise typer.BadParameter(f"Unknown adapter '{adapter_id}'. Available: {available}")
    config_cls, adapter_cls, _ = entry
    return adapter_cls(config_cls())


def _create_generator(adapter_id: str, adapter: ApiAdapter, model: str) -> Generator:
    _, _, generator_cls = _ADAPTER_REGISTRY[adapter_id]
    return generator_cls(adapter, model)


def _results_table(results, merged_count: int, usage) -> Table:
    table = Table(title="Annotation Results")
    table.add_column("Type", style="cyan")
    table.add_column("Category", style="cyan")
    table.add_column("Cluster", style="green")
    table.add_column("Annotation")
    table.add_column("Merged", justify="right")

    for ann in results:
        merged = str(len(ann.merged_ids) - 1) if len(ann.merged_ids) > 1 else ""
        table.add_row(
            ann.type,
            ann.category,
            ann.label,
            (ann.text or "")[:80],
            merged,
        )

    return table


_TYPE_ALIASES: dict[str, str] = {"a": "action", "o": "observation"}


async def _run_annotation(
    adapter_id: str,
    annotate_model: str,
    summarizer_model: str | None,
    dry_run: bool,
    workers: int,
    specs: list[AnnotatingJobSpec] | None = None,
) -> None:
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    adapter = _create_adapter(adapter_id)
    await adapter.startup()

    try:
        generator = _create_generator(adapter_id, adapter, annotate_model)

        summarizer = None
        if summarizer_model:
            summarizer_gen = _create_generator(adapter_id, adapter, summarizer_model)
            summarizer = MapReduceSummarizer(summarizer_gen)

        embedder_client = EmbedderClient(config.embedder)
        await embedder_client.startup()

        try:
            embedder = Embedder(embedder_client, dims=config.message_dims)

            pipeline = AnnotationPipeline(
                session_factory,
                generator,
                embedder,
                summarizer=summarizer,
                workers=workers,
            )

            result = await pipeline.run(specs=specs, dry_run=dry_run)

            console.print(_results_table(result.results, result.merged_count, result.usage))
            u = result.usage
            s = result.summarizer_usage
            console.print(
                f"\n[bold]{len(result.results)}[/bold] annotations, "
                f"[bold]{result.merged_count}[/bold] merged"
            )
            console.print(
                f"  annotator  ({annotate_model}): "
                f"[bold]{u.input_tokens}[/bold] input tokens, "
                f"[bold]{u.output_tokens}[/bold] output tokens"
            )
            if summarizer_model:
                console.print(
                    f"  summarizer ({summarizer_model}): "
                    f"[bold]{s.input_tokens}[/bold] input tokens, "
                    f"[bold]{s.output_tokens}[/bold] output tokens"
                )
            if result.merged_count > 0:
                console.print(
                    f"\n[yellow]Warning:[/yellow] {result.merged_count} clusters were merged. "
                    "Run [bold]episodiq cluster rebuild-paths[/bold] to update trajectory paths."
                )
        finally:
            await embedder_client.shutdown()
    finally:
        await adapter.shutdown()


def _build_specs(types: list[str] | None, categories: list[str] | None) -> list[AnnotatingJobSpec] | None:
    """Build specs from CLI flags. Returns None for defaults (all)."""
    if not types and not categories:
        return None
    types = [_TYPE_ALIASES.get(t, t) for t in (types or ["action", "observation"])]
    categories = categories or ["text", "tool"]
    return [AnnotatingJobSpec(type=t, category=c) for t in types for c in categories]


@app.callback(invoke_without_command=True)
def annotate(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    adapter: str = typer.Option(..., "--adapter", "-a", help="Adapter ID (openai, anthropic)"),
    annotate_model: str = typer.Option(..., "--annotate-model", "-am", help="Model for annotation"),
    summarizer_model: Optional[str] = typer.Option(None, "--summarizer-model", "-sm", help="Model for summarization (optional, defaults to annotate model)"),
    types: Optional[list[str]] = typer.Option(None, "--type", "-t", help="action/observation/a/o"),
    categories: Optional[list[str]] = typer.Option(None, "--category", "-c", help="text, tool, or specific tool name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip DB writes"),
    workers: int = typer.Option(API_CONCURRENCY, "--workers", "-w", help="Max concurrent LLM calls"),
) -> None:
    """Run cluster annotation pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    specs = _build_specs(types, categories)
    console.print("[dim]This may take a while...[/dim]")
    asyncio.run(_run_annotation(adapter, annotate_model, summarizer_model, dry_run, workers, specs=specs))
