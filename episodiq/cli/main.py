import typer

from episodiq.cli import annotate, cluster, db, index, report, start, tune

app = typer.Typer(help="Episodiq: pattern mining tool for agentic trajectories")

# Top-level commands
app.command(name="up", help="Start Episodiq proxy server")(start.up)
app.command(name="report", help="Render trajectory report")(report.report)

# Subcommands
app.add_typer(cluster.app, name="cluster", help="Clustering operations")
app.add_typer(annotate.app, name="annotate", help="Cluster annotation")
app.add_typer(index.app, name="index", help="Path indexing (trace_tokens + minhash_sig)")
app.add_typer(tune.app, name="tune", help="Parameter tuning")
app.add_typer(db.app, name="db", help="Database management (create/migrate)")

if __name__ == "__main__":
    app()
