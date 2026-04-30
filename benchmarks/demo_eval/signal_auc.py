"""Compute AUC for trajectory-level signal rates (action/transition).

Queries trajectory_paths, aggregates signal counts per trajectory,
computes ROC-AUC against trajectory outcome (success/failure).

Usage:
    uv run python signal_auc.py --env .env
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

# Success signals predict success, so AUC < 0.5 for failure label.
# We invert their AUC to make all signals comparable (higher = better at predicting failure).
INVERT_AUC = {"success_signal_action", "success_signal_transition"}


async def compute_signal_auc(session_factory) -> dict:
    async with session_factory() as session:
        rows = await session.execute(text("""
            SELECT
                t.id,
                t.status,
                MAX(tp.index) + 1 as n_steps,
                COALESCE(MAX(tp.fail_risk_action_count), 0) as fra,
                COALESCE(MAX(tp.fail_risk_transition_count), 0) as frt,
                COALESCE(MAX(tp.success_signal_action_count), 0) as ssa,
                COALESCE(MAX(tp.success_signal_transition_count), 0) as sst
            FROM trajectories t
            JOIN trajectory_paths tp ON tp.trajectory_id = t.id
            WHERE t.status IN ('success', 'failure')
            GROUP BY t.id, t.status
            HAVING MAX(tp.index) + 1 >= 3
        """))
        data = rows.fetchall()

    if not data:
        return {"error": "no trajectories with paths"}

    labels = []
    rates = {
        "fail_risk_action": [],
        "fail_risk_transition": [],
        "success_signal_action": [],
        "success_signal_transition": [],
    }

    for row in data:
        tid, status, n_steps, fra, frt, ssa, sst = row
        is_failure = 1 if status == "failure" else 0
        labels.append(is_failure)
        rates["fail_risk_action"].append(fra / n_steps)
        rates["fail_risk_transition"].append(frt / n_steps)
        rates["success_signal_action"].append(ssa / n_steps)
        rates["success_signal_transition"].append(sst / n_steps)

    results = {
        "n_trajectories": len(labels),
        "n_failure": sum(labels),
        "n_success": len(labels) - sum(labels),
    }

    if len(set(labels)) < 2:
        results["error"] = "single class"
        return results

    for signal_name, signal_rates in rates.items():
        if any(r > 0 for r in signal_rates):
            try:
                auc = float(roc_auc_score(labels, signal_rates))
                # Invert success signals so all AUCs are "higher = better failure predictor"
                if signal_name in INVERT_AUC:
                    auc = 1.0 - auc
                results[signal_name] = {"auc": round(auc, 4), "mean_rate": round(float(np.mean(signal_rates)), 4)}
            except Exception as e:
                results[signal_name] = {"auc": None, "error": str(e)}
        else:
            results[signal_name] = {"auc": None, "mean_rate": 0.0}

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute signal AUC")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--output", default="output/signal_auc.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(args.env)

    import os
    db_url = os.environ["EPISODIQ_DATABASE_URL"]

    engine = create_async_engine(db_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    import asyncio

    async def _run():
        result = await compute_signal_auc(session_factory)
        await engine.dispose()
        return result

    result = asyncio.run(_run())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Signal AUC (n={result.get('n_trajectories', 0)}) ===")
    for key in ["fail_risk_action", "fail_risk_transition", "success_signal_action", "success_signal_transition"]:
        info = result.get(key, {})
        auc = info.get("auc")
        rate = info.get("mean_rate", 0)
        print(f"  {key:<30} AUC={auc}  rate={rate}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
