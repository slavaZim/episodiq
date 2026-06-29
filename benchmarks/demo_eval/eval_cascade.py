"""Evaluate a tune-winner cascade config on the held-out eval slice.

Reads ``tune_config.json`` (winner params + shuffle_seed + tune_limit
from ``08_tune.sh``). Loads all completed trajectories from the DB,
applies the same ``Random(seed).shuffle`` as the sweep, and runs the
cascade with the tuned hyperparameters on queries from index
``tune_limit:`` (the eval slice). Each query excludes its own
trajectory; corpus = all other completed trajectories.

The split lives in **hyperparameter selection**, not the data: the
eval AUC is unbiased w.r.t. the eval queries because Optuna only saw
tune-slice AUCs during tuning. Eval queries see the full corpus
(including tune-slice trajs), mirroring production retrieval where
new queries hit a static historical index.

Output: JSON report with the winner config, eval AUC under all three
metrics (cummax / cummean / cummeanmax), and headline numbers for the
winner's tuned metric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random as _random
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.cli.env import _load_dotenv
from episodiq.config import get_config
from episodiq.config.retrieval_config import (
    RetrievalConfig, WindowMinHashConfig,
)
from episodiq.config.scoring_config import AggShiftConfig
from episodiq.retrieval.cache import RetrievalCache
from episodiq.retrieval.tune.lsh_rebuild import (
    drop_alt_lsh_table, rebuild_lsh_into,
)
from episodiq.analytics.metrics import (
    SIMILARITY_METRICS as METRICS, bootstrap_aucs, compute_metric_curves,
)
from episodiq.retrieval.tune.sweep import (
    TuneSnapshot, _resolve_shuffle_key, _run_searches,
    _stratified_interleave,
)
from episodiq.storage.postgres.repository import TrajectoryRepository

logger = logging.getLogger(__name__)


async def _load_eval_snapshots(
    session_factory, shuffle_seed: int, tune_limit: int,
    shuffle_field: str | None = None,
    stratify_field: str | None = None,
) -> tuple[list[TuneSnapshot], dict]:
    """Re-derive the eval slice using the same shuffle as the sweep.
    Returns ``(snapshots, status_by_tid)``. Snapshots come from the
    trajectories at indices ``[tune_limit:]`` of the shuffled list.
    """
    async with session_factory() as session:
        repo = TrajectoryRepository(session)
        trajs = await repo.get_with_completed_paths(["success", "failure"])
    if not trajs:
        return [], {}
    trajs = list(trajs)
    if shuffle_field:
        trajs.sort(key=lambda t: _resolve_shuffle_key(t, shuffle_field))
    rng = _random.Random(shuffle_seed)
    rng.shuffle(trajs)
    if stratify_field:
        trajs = _stratified_interleave(trajs, stratify_field)
    eval_trajs = trajs[tune_limit:]
    status = {t.id: t.status for t in trajs}

    snapshots: list[TuneSnapshot] = []
    for t in eval_trajs:
        for p in sorted(t.paths, key=lambda x: x.index or 0):
            if p.trace_tokens and p.index is not None:
                snapshots.append(TuneSnapshot(
                    trajectory_id=t.id,
                    step=int(p.index),
                    tokens=list(p.trace_tokens),
                    path_id=p.id,
                ))
    return snapshots, status


async def run(args):
    cfg = json.loads(Path(args.config).read_text())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    db_url = get_config().get_database_url()
    engine = create_async_engine(db_url, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    W = int(cfg["window"])
    mh_cfg = WindowMinHashConfig(window=W)
    cas_cfg = RetrievalConfig(
        aggregation=cfg["aggregation"],
        prefetch_n_uniq=int(cfg["prefetch_n_uniq"]),
        jaccard_n_uniq=int(cfg["jaccard_n_uniq"]),
        top_k=int(cfg["top_k"]),
    )
    ms_cfg = AggShiftConfig(
        window=W,
        lam=float(cfg["lam"]),
        penalty_shape=cfg["penalty_shape"],
        gap_open=float(cfg["gap_open"]),
        gap_extend=float(cfg["gap_extend"]),
        sigma=float(cfg["sigma"]),
    )

    snapshots, status = await _load_eval_snapshots(
        session_factory,
        shuffle_seed=int(cfg["shuffle_seed"]),
        tune_limit=int(cfg["tune_limit"]),
        shuffle_field=cfg.get("shuffle_field"),
        stratify_field=cfg.get("stratify_field"),
    )
    if not snapshots:
        raise SystemExit("no eval snapshots — check tune_limit vs DB size")
    logger.info(
        "eval slice: %d snapshots from %d trajs",
        len(snapshots), len({s.trajectory_id for s in snapshots}),
    )

    # Build a fresh alt LSH for the tune W over the entire completed
    # corpus. Each eval query self-excludes via trajectory_id, so the
    # corpus naturally hides only the query's own trajectory.
    table_name = f"trajectory_window_lsh_eval_w{W}"
    async with session_factory() as session:
        n_trajs, n_rows = await rebuild_lsh_into(
            session, mh_cfg, table_name,
        )
    logger.info(
        "alt LSH built: %d trajs %d rows -> %s",
        n_trajs, n_rows, table_name,
    )

    try:
        cache = RetrievalCache()
        fail_sims = await _run_searches(
            snapshots, status, session_factory, table_name,
            mh_cfg=mh_cfg, cas_cfg=cas_cfg, ms_cfg=ms_cfg,
            n_workers=args.n_workers, cache=cache,
        )
        curves = compute_metric_curves(
            fail_sims, status, eval_min_step=args.eval_min_step,
        )
        # Per-step AUC curve recorded in the report as a diagnostic
        # (compute_metric_curves is documented for this). Floored below
        # the headline's ``eval_min_step`` so it also covers early steps.
        step_curves = compute_metric_curves(
            fail_sims, status, eval_min_step=args.curve_min_step,
        )
    finally:
        async with session_factory() as session:
            await drop_alt_lsh_table(session, table_name)
        await engine.dispose()

    if not curves:
        raise SystemExit("no AUC curves produced — eval slice too thin?")

    weighted = {m: curves[m].weighted_auc for m in METRICS if m in curves}
    winner_metric = cfg.get("metric")
    headline = weighted.get(winner_metric) if winner_metric else None

    # Bootstrap CI per metric so the reader can see whether the
    # eval-side AUC is meaningfully above 0.5 / above the tune number
    # vs falling inside a wide band that's compatible with noise.
    ci_by_metric = bootstrap_aucs(
        fail_sims, status, eval_min_step=args.eval_min_step,
        n_boot=args.n_boot, seed=args.boot_seed,
    )
    ci_json = {
        m: {"lo": ci.lo, "hi": ci.hi, "mean": ci.mean}
        for m, ci in ci_by_metric.items()
    }
    headline_ci = ci_by_metric.get(winner_metric) if winner_metric else None

    report = {
        "winner_metric": winner_metric,
        "tune_auc": cfg.get("tune_auc"),
        "eval_auc": headline,
        "eval_auc_ci": (
            {"lo": headline_ci.lo, "hi": headline_ci.hi}
            if headline_ci else None
        ),
        "eval_auc_per_metric": weighted,
        "eval_auc_ci_per_metric": ci_json,
        "curve_min_step": args.curve_min_step,
        "eval_step_curves": {
            m: [
                {"step": sa.step, "auc": sa.auc, "n_active": sa.n_active}
                for sa in step_curves[m].per_step
            ]
            for m in METRICS if m in step_curves
        },
        "n_eval_trajs": len({s.trajectory_id for s in snapshots}),
        "n_eval_snapshots": len(snapshots),
        "n_boot": args.n_boot,
        "config": cfg,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    ci_str = (
        f" [{headline_ci.lo:.4f}, {headline_ci.hi:.4f}]"
        if headline_ci else ""
    )
    per_metric_lines = []
    for m in METRICS:
        if m not in weighted:
            continue
        ci = ci_by_metric.get(m)
        if ci:
            per_metric_lines.append(
                f"{m}={weighted[m]:.4f} [{ci.lo:.4f}, {ci.hi:.4f}]"
            )
        else:
            per_metric_lines.append(f"{m}={weighted[m]:.4f}")
    print(
        f"\n=== Eval result ===\n"
        f"  W={W}  agg={cfg['aggregation']}  metric={winner_metric}\n"
        f"  tune_auc = {cfg.get('tune_auc'):.4f}\n"
        f"  eval_auc = {headline:.4f}{ci_str}  (under metric={winner_metric})\n"
        f"  per-metric eval AUC (95% CI):\n    "
        + "\n    ".join(per_metric_lines)
        + f"\nSaved to {args.output}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env")
    p.add_argument("--config", required=True,
                   help="path to tune_config.json from 08_tune.sh")
    p.add_argument("--output", required=True,
                   help="path to write the eval JSON report")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--eval-min-step", type=int, default=50)
    p.add_argument("--curve-min-step", type=int, default=1,
                   help="Floor for the per-step AUC curve recorded in the "
                        "report (independent of --eval-min-step, which sets "
                        "the headline weighted-AUC floor).")
    p.add_argument("--n-boot", type=int, default=200,
                   help="Bootstrap draws for per-metric AUC CI (95 pct).")
    p.add_argument("--boot-seed", type=int, default=42,
                   help="RNG seed for bootstrap reproducibility.")
    args = p.parse_args()
    _load_dotenv(Path(args.env))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
