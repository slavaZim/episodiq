"""Aggregate eval metrics from trajectory JSONL reports.

Reads a single JSONL file with all trajectory reports (each line has trajectory_id).
Collects raw dead_end_prob, computes AUC and precision/recall sweep.

Usage:
    uv run python eval_metrics.py \
      --traj-ids output/eval_traj_ids.json \
      --reports output/reports.jsonl \
      --output output/eval_summary.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _load_reports(path: Path) -> dict[str, list[dict]]:
    """Load a single JSONL file and group entries by trajectory_id."""
    from collections import defaultdict

    by_traj: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                tid = entry.get("trajectory_id", "")
                by_traj[tid].append(entry)
    return dict(by_traj)


def _trajectory_signals(entries: list[dict]) -> dict:
    observations = [e for e in entries if e.get("type") == "observation"]
    actions = [e for e in entries if e.get("type") == "action"]

    # Raw dead-end probs per step
    dead_end_probs = [e["dead_end_prob"] for e in observations if "dead_end_prob" in e]
    max_prob = max(dead_end_probs) if dead_end_probs else None

    # Signal counts from report entries
    n_fail_risk_action = sum(1 for e in actions if e.get("fail_risk_action"))
    n_success_signal_action = sum(1 for e in actions if e.get("success_signal_action"))
    n_fail_risk_transition = sum(1 for e in observations if e.get("fail_risk_transition"))
    n_success_signal_transition = sum(1 for e in observations if e.get("success_signal_transition"))

    # Normalize all rates by path count (= n_steps = max(path.index) + 1)
    n_steps = len(observations) or 1

    # Per-step probs for "turns remaining" computation at any threshold
    step_probs = []
    for i, e in enumerate(observations):
        if "dead_end_prob" in e:
            step_probs.append((i, e["dead_end_prob"]))

    return {
        "max_dead_end_prob": max_prob,
        "step_probs": step_probs,
        "fail_risk_action_rate": n_fail_risk_action / n_steps,
        "fail_risk_transition_rate": n_fail_risk_transition / n_steps,
        "success_signal_action_rate": n_success_signal_action / n_steps,
        "success_signal_transition_rate": n_success_signal_transition / n_steps,
        "n_steps": n_steps,
    }


def _precision_recall_at(trajectories: list[dict], threshold: float) -> dict:
    """Compute precision/recall/F1 at a given threshold."""
    tp = fp = fn = 0
    turns_remaining_list = []

    for t in trajectories:
        prob = t["max_dead_end_prob"]
        if prob is None:
            # No prediction — count as not flagged
            if t["is_failure"]:
                fn += 1
            continue

        flagged = prob >= threshold
        if flagged and t["is_failure"]:
            tp += 1
            # Find first step exceeding threshold for turns remaining
            for step_idx, step_prob in t["step_probs"]:
                if step_prob >= threshold:
                    turns_remaining_list.append(t["n_steps"] - step_idx)
                    break
        elif flagged and not t["is_failure"]:
            fp += 1
        elif not flagged and t["is_failure"]:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_turns = float(np.mean(turns_remaining_list)) if turns_remaining_list else None

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "avg_turns_remaining": round(avg_turns, 1) if avg_turns is not None else None,
    }


def compute_metrics(traj_ids: dict, reports_path: Path) -> dict:
    trajectories = []

    all_reports = _load_reports(reports_path)

    for traj_uuid, meta in traj_ids.items():
        entries = all_reports.get(traj_uuid, [])
        if not entries:
            logger.warning("Missing report for %s, skipping", traj_uuid[:8])
            continue

        signals = _trajectory_signals(entries)
        signals["uuid"] = traj_uuid
        signals["status"] = meta["status"]
        signals["is_failure"] = meta["status"] == "failure"
        signals["instance_id"] = meta.get("instance_id", "")
        trajectories.append(signals)

    if not trajectories:
        return {"error": "no valid reports"}

    n_total = len(trajectories)
    n_success = sum(1 for t in trajectories if not t["is_failure"])
    n_failure = n_total - n_success

    # --- Dead-end AUC (threshold-independent) ---
    dead_end_auc = None
    scored = [t for t in trajectories if t["max_dead_end_prob"] is not None]
    if scored:
        labels = [1 if t["is_failure"] else 0 for t in scored]
        probs = [t["max_dead_end_prob"] for t in scored]
        if len(set(labels)) == 2:
            try:
                dead_end_auc = float(roc_auc_score(labels, probs))
            except Exception as e:
                logger.warning("Could not compute dead-end AUC: %s", e)

    # --- Threshold sweep ---
    sweep = [_precision_recall_at(trajectories, t) for t in THRESHOLDS]
    best = max(sweep, key=lambda r: r["f1"])

    # --- Signal AUCs ---
    labels_all = [1 if t["is_failure"] else 0 for t in trajectories]
    signals = {
        "fail_risk_action": [t["fail_risk_action_rate"] for t in trajectories],
        "fail_risk_transition": [t["fail_risk_transition_rate"] for t in trajectories],
        "success_signal_action": [t["success_signal_action_rate"] for t in trajectories],
        "success_signal_transition": [t["success_signal_transition_rate"] for t in trajectories],
    }
    # Success signals: invert AUC (higher rate → success, not failure)
    invert_auc = {"success_signal_action", "success_signal_transition"}

    signal_results = {}
    for name, rates in signals.items():
        auc = None
        if len(set(labels_all)) == 2 and any(r > 0 for r in rates):
            try:
                auc = float(roc_auc_score(labels_all, rates))
                if name in invert_auc:
                    auc = 1.0 - auc
            except Exception as e:
                logger.warning("Could not compute %s AUC: %s", name, e)
        signal_results[name] = {
            "auc": round(auc, 4) if auc is not None else None,
            "rate": round(float(np.mean(rates)), 4),
        }

    return {
        "n_trajectories": n_total,
        "n_success": n_success,
        "n_failure": n_failure,
        "n_scored": len(scored),
        "dead_end": {
            "auc": round(dead_end_auc, 4) if dead_end_auc is not None else None,
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "best_avg_turns_remaining": best["avg_turns_remaining"],
            "sweep": sweep,
        },
        "signals": signal_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute eval metrics from JSONL reports")
    parser.add_argument("--traj-ids", required=True, help="Path to eval_traj_ids.json")
    parser.add_argument("--reports", required=True, help="Single JSONL file with all trajectory reports")
    parser.add_argument("--output", required=True, help="Output eval_summary.json path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.traj_ids) as f:
        traj_ids = json.load(f)

    summary = compute_metrics(traj_ids, Path(args.reports))

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    # Print key results
    de = summary.get("dead_end", {})
    print("\n=== Dead-End Prediction ===")
    print(f"AUC: {de.get('auc')}")
    print(f"Best threshold: {de.get('best_threshold')} (F1={de.get('best_f1')}, P={de.get('best_precision')}, R={de.get('best_recall')})")
    if de.get("best_avg_turns_remaining"):
        print(f"Avg turns remaining: {de['best_avg_turns_remaining']}")

    print("\n=== Threshold Sweep ===")
    print(f"{'Threshold':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'TP':>4} {'FP':>4} {'FN':>4}")
    for row in de.get("sweep", []):
        marker = " <-" if row["threshold"] == de.get("best_threshold") else ""
        print(f"{row['threshold']:>9.1f} {row['precision']:>9.4f} {row['recall']:>9.4f} {row['f1']:>9.4f} {row['tp']:>4} {row['fp']:>4} {row['fn']:>4}{marker}")

    print("\n=== Signals ===")
    for name, info in summary.get("signals", {}).items():
        print(f"  {name:<30} AUC={info.get('auc')}  rate={info.get('rate')}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
