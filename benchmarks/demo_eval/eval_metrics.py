"""Compute the headline metric (AUC@step at min_step=50) from
`episodiq report --format json` output (JSONL).

Inputs:
  --traj-ids   JSON mapping {trajectory_id: {"status": "failure"|"success"}}
               (the eval slice picked by 09_eval.sh).
  --reports    JSONL: one observation/action entry per line, produced by
               `episodiq report <tid> --format json`. Each line includes
               `trajectory_id`, `index`, `type`, and (for observations of
               paths that produced a prediction) `fail_frac`.

Output: one number — length-stratified per-step running-max ROC AUC,
mean over absolute steps s ≥ 50.

Usage:
    uv run python eval_metrics.py \\
      --traj-ids output/eval_traj_ids.json \\
      --reports output/reports.jsonl \\
      --output output/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

MIN_STEP = 50


def _parse_reports(
    reports_path: Path, valid_ids: set[str],
) -> dict[str, list[tuple[int, float]]]:
    """Read JSONL, return {trajectory_id: [(step_index, fail_frac), ...]}."""
    per_traj: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with reports_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") != "observation":
                continue
            tid = entry.get("trajectory_id")
            if tid not in valid_ids:
                continue
            ff = entry.get("fail_frac")
            if ff is None:
                continue
            idx = entry.get("index")
            if idx is None:
                continue
            per_traj[tid].append((int(idx), float(ff)))
    return dict(per_traj)


def _auc_per_step_mean(
    per_traj: dict[str, list[tuple[int, float]]],
    traj_label: dict[str, int],
    min_step: int,
) -> tuple[float, int]:
    """Length-strat running-max AUC: per absolute step s ≥ min_step, ROC AUC
    over active trajs (score = max fail_frac up to s); mean across steps.
    """
    if not per_traj:
        return float("nan"), 0
    max_step_per_traj = {
        tid: max(s for s, _ in snaps) for tid, snaps in per_traj.items()
    }
    max_step = max(max_step_per_traj.values())
    aucs: list[float] = []
    for s in range(min_step, max_step + 1):
        y, scores = [], []
        for tid, snaps in per_traj.items():
            if max_step_per_traj[tid] < s:
                continue
            ffs = [ff for st, ff in snaps if st <= s]
            if not ffs:
                continue
            y.append(traj_label[tid])
            scores.append(max(ffs))
        if len(set(y)) >= 2:
            aucs.append(roc_auc_score(y, scores))
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def compute_summary(
    per_traj: dict[str, list[tuple[int, float]]],
    traj_label: dict[str, int],
) -> dict:
    auc, n_steps = _auc_per_step_mean(per_traj, traj_label, MIN_STEP)
    n_fail = sum(1 for tid in per_traj if traj_label[tid] == 1)
    n_success = sum(1 for tid in per_traj if traj_label[tid] == 0)
    return {
        "n_trajectories": len(per_traj),
        "n_fail": n_fail,
        "n_success": n_success,
        "base_fail_rate": (
            round(n_fail / (n_fail + n_success), 4)
            if (n_fail + n_success) > 0 else None
        ),
        "min_step": MIN_STEP,
        "auc_step": round(auc, 4) if not np.isnan(auc) else None,
        "n_steps_averaged": n_steps,
    }


def _print_summary(summary: dict) -> None:
    print(
        f"\neval set: {summary['n_trajectories']} trajectories "
        f"({summary['n_fail']} fail / {summary['n_success']} success), "
        f"base fail rate {summary['base_fail_rate']}"
    )
    auc = summary["auc_step"]
    n_steps = summary["n_steps_averaged"]
    print(f"\nAUC@step (min_step={summary['min_step']}): "
          f"{'n/a' if auc is None else f'{auc:.4f}'}  (n_steps={n_steps})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-ids", type=Path, required=True,
                   help='JSON: {traj_id: {"status": "failure"|"success"}}')
    p.add_argument("--reports", type=Path, required=True,
                   help="JSONL produced by `episodiq report --format json`")
    p.add_argument("--output", type=Path, default=Path("output/eval_summary.json"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = json.loads(args.traj_ids.read_text())
    traj_label = {
        tid: 1 if meta.get("status") == "failure" else 0
        for tid, meta in raw.items()
    }
    per_traj = _parse_reports(args.reports, set(traj_label))
    logger.info("loaded %d trajs from reports, %d in traj_ids",
                len(per_traj), len(traj_label))

    summary = compute_summary(per_traj, traj_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    _print_summary(summary)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
