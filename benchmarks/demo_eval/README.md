# Demo Evaluation Pipeline

> **Note:** The `--env` argument accepted by all CLI commands in this demo is a development convenience that loads a local `.env` file. In production the proxy reads configuration from environment variables directly and `--env` is not used.

End-to-end walkthrough of the Episodiq pipeline on **SWE-smith** trajectories (getmoto/moto subset).

## Overview

```
01 Seed 500 train trajectories (mock server + proxy)
02 Grid search clustering parameters
03 Cluster with chosen params (manually per type/category)
04 Build trajectory paths
05 Tune thresholds (prefetch, anomaly, stuck, path-freq)
06 Rebuild paths with signals (using tuned thresholds)
07 Train dead-end prediction model
08 Seed 250 eval trajectories
09 Generate reports + compute metrics
```

## Prerequisites

- Docker: Postgres with pgvector running on port 5433
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed (`uv sync` in repo root to set up the dev environment)
- OpenRouter API key in `.env` (for embeddings)

## Dataset

**SWE-bench/SWE-smith-trajectories** (HuggingFace), split `tool`, filtered by `getmoto__moto` in `instance_id`.

- Train: first 500 trajectories
- Eval: next 250 trajectories
- Messages are in OpenAI chat format with tool_calls

## Steps

### Step 1: Seed Training Data

```bash
./01_seed_train.sh
```

Starts a mock LLM server (returns pre-recorded responses from the dataset) and the Episodiq proxy. Replays 500 trajectories turn-by-turn through the proxy, which embeds and saves each message. Marks trajectory status (`success`/`failure`) immediately after each trajectory.

Output: `output/train_traj_ids.json`

### Step 2: Grid Search

```bash
./02_grid_search.sh
```

Runs HDBSCAN/UMAP parameter grid search across all type/category combinations. Produces a CSV with parameter combinations sorted by composite score. 

Output: `output/grid_search.csv`

**How to pick params:** Open the CSV. For each `(type, category)` pair, look at the top entries. The composite score balances DBCV (cluster quality), entropy (information content), and noise ratio. Prefer configs with:
- Low noise% (<20%)
- Reasonable cluster count (not too few, not fragmented). Too many clusters make the transition space sparse — analytics needs significantly more trajectories to produce reliable signals. If human-readable labels are the priority over analytics, pick the config with the lowest noise regardless of cluster count. Keep in mind that more clusters also increase annotator token cost.
- High DBCV (>0.3 is decent)

Current cluster counts (after step 03):

| type        | category           | clusters |
|-------------|--------------------|---------:|
| observation | text               |      128 |
| observation | bash               |       80 |
| observation | str_replace_editor |      117 |
| observation | submit             |        5 |
| action      | text               |      125 |
| action      | bash               |       76 |
| action      | str_replace_editor |      137 |
| action      | submit             |        5 |

Save chosen params to `output/cluster_config.json`:
```json
[
  {"type": "observation", "category": "text", "min_cluster_size": 10, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15},
  {"type": "action", "category": "text", "min_cluster_size": 10, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15},
  {"type": "observation", "category": "bash", "min_cluster_size": 15, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15},
  {"type": "action", "category": "bash", "min_cluster_size": 15, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15}
]
```

### Step 3: Cluster

```bash
./03_cluster.sh
```

Runs clustering for each entry in `cluster_config.json`. Creates cluster assignments in DB.

### Step 4: Build Paths

```bash
./04_build_paths.sh
```

Reconstructs trajectory paths from cluster labels. Computes transition profiles and traces. No signal filling yet — thresholds not tuned.

### Step 5: Tune Thresholds

```bash
./05_tune.sh
```

Runs three tuning commands:
- **prefetch-topk**: sweeps HNSW prefetch_n × top_k combinations and reports hit@5. Pick the smallest prefetch/top_k where further increases yield negligible hit@5 gain.
- **signal-thresholds**: sweeps `fail_similarity` thresholds for two signals — fail-risk action (maximise AUC) and success-signal action (minimise AUC). Outputs suggested `EPISODIQ_FAIL_RISK_ACTION_THRESHOLD` and `EPISODIQ_SUCCESS_SIGNAL_ACTION_THRESHOLD`.
- **path-freq**: entropy percentiles for action variance (low/high entropy thresholds)

Each outputs suggested values. Update `.env` with the suggested thresholds before proceeding.

Output: `output/tune_*.txt`

### Step 6: Rebuild Paths with Signals

```bash
./06_rebuild_paths.sh
```

Rebuilds paths with `--fill-signals` using the tuned thresholds. Populates `anomaly_count` and `stuck_count` on each path.

### Step 7: Train Dead-End Model

```bash
./07_train_dead_end.sh
```

Trains logistic regression on full train set (no test split). Features: 15 neighbor-based + 4 DPM-SVD = 19 total.

Output: `output/dead_end_model.joblib`

### Step 8: Seed Eval Data

```bash
./08_seed_eval.sh
```

Seeds 250 eval trajectories through proxy. Status marked immediately after each (data moat — each subsequent trajectory benefits from growing pool).

Output: `output/eval_traj_ids.json`

### Step 9: Evaluate

```bash
./09_eval.sh
```

For each eval trajectory, generates a JSONL report via `episodiq report --format json`. Then computes aggregate metrics:

- **Dead-end AUC**: threshold-independent, raw probability vs actual outcome
- **Threshold sweep**: precision/recall/F1 at thresholds 0.1-0.9
- **Anomaly AUC**: anomaly rate as failure predictor
- **Stuck rates**: success vs failure comparison

Output: `output/reports/*.jsonl`, `output/eval_summary.json`

## Architecture

```
Dataset (HuggingFace)
    |
    v
Mock Server (port 9999) <-- Proxy (port 8081) <-- seed_via_proxy.py
    |                           |
    |                           v
    |                     PostgreSQL (port 5433)
    |                           |
    v                           v
Pre-recorded responses    Embeddings + Clusters + Paths
                                |
                                v
                    CLI: cluster, tune, dead-end train
                                |
                                v
                    CLI: episodiq report --format json
                                |
                                v
                    eval_metrics.py → eval_summary.json
```
