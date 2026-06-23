# Demo Evaluation Pipeline

End-to-end walkthrough of the Episodiq pipeline on **SWE-rebench-openhands-trajectories** — 275 [`tobymao/sqlglot`](https://github.com/tobymao/sqlglot) trajectories. Seeds the DB, clusters messages and act_obs pairs, builds the per-window MinHash LSH index, tunes & evaluates the cascade retrieval signal, and renders demo reports.

> **Note:** the `--env` flag in every script loads a local `.env` file — convenience for the bench only. In production Episodiq reads config from environment variables directly.

## Headline numbers (latest run)

`cummean` is the headline metric — running mean of per-snapshot `fail_similarity`, the most noise-robust of the three Episodiq aggregations. Per-trajectory bootstrap 95% CI (200 draws).

| Metric | Value |
|---|---|
| Tune AUC@s50 (cummean, n=100) | **0.6440** |
| Eval AUC@s50 (cummean, n=175) | **0.6983** [0.609, 0.782] |
| Picked config | `W=10  agg=min_distance  prefetch_n_uniq=220  jaccard_n_uniq=140  top_k=10  penalty=lin  lam=1.2  σ=0.5  gap_open=2.8  gap_extend=1.2`; LSH layout B=64, R=1 |

Both slices come from the same `Random(42).shuffle` of instance-id-sorted trajectories, then a proportional interleave by status — both slices land at ~65% failure rate (population mean). The 100-trajectory tune slice drives hyperparam selection; the 175-trajectory eval slice is held out.

Top-level [README](../../README.md#pattern-retrieval-vs-basic-rag) has the head-to-head against the Basic RAG baseline.

## Overview

```
01 Seed 275 sqlglot trajectories (mock server + proxy)
02 Grid-search HDBSCAN/UMAP params per (type, category)
03 Cluster messages with chosen params
04 Annotate clusters with LLM labels, then rebuild paths
05 Grid-search AO tokenizer params
06 Tokenize (act_obs pairs → token vocabulary)
07 Build the per-window MinHash LSH index (trace_tokens + bands)
08 Tune retrieval cascade on a 100-traj stratified tune slice (Optuna TPE)
09 Eval — score the remaining 175 trajectories with the tune winner
10 Tune path-frequency entropy thresholds (optional)
11 Demo — dump every trajectory's report (JSONL with -a analytics)
```

Steps 02 → 06 are fit on the full DB. Steps 08 + 09 share the same `Random(42).shuffle + proportional-interleave-by-status` ordering: 100 trajectories drive hyperparam selection, the remaining 175 are held out for eval. Both slices retain the population's ~65% failure rate.

## Prerequisites

- Docker: Postgres+pgvector on port 5433 (DB name comes from `EPISODIQ_DATABASE_URL` in `.env`)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed (`uv sync` in repo root)
- Embedding endpoint in `.env`: `EPISODIQ_EMBEDDER_URL` + `EPISODIQ_EMBEDDER_API_KEY` + `EPISODIQ_EMBEDDER_MODEL`
- Annotation endpoint in `.env` (OpenAI- or Anthropic-compatible): **both** base URL and API key — `EPISODIQ_OPENAI_BASE_URL` + `EPISODIQ_OPENAI_API_KEY`, or `EPISODIQ_ANTHROPIC_BASE_URL` + `EPISODIQ_ANTHROPIC_API_KEY`
- Bulk-seeding-friendly embedder settings in `.env`: `EPISODIQ_POSTPROCESS_TIMEOUT=600`, `EPISODIQ_EMBEDDER_MAX_CONNECTIONS=100`, `EPISODIQ_EMBEDDER_MAX_RETRIES=5`, `EPISODIQ_EMBEDDER_BACKOFF_MODE=constant` (see [Troubleshooting](#troubleshooting))

## Steps

### Step 1: Seed dataset

```bash
./01_seed_dataset.sh
```

Boots the mock LLM server (replays pre-recorded dataset responses) and the Episodiq proxy. Streams every sqlglot trajectory through the proxy, which embeds & saves messages and marks `success` / `failure`. Writes `output/sqlglot_traj_ids.json` (uuid → instance_id + traj_id mapping for later mapping back to the source dataset).

### Step 2: Grid search

```bash
./02_grid_search.sh
```

Runs HDBSCAN+UMAP grid search for every `(type, category)` pair. Output: `output/grid_search.csv` with columns: `min_cluster_size, min_samples, umap_dims, umap_n_neighbors, selection_method, selection_epsilon, n_clusters, noise_count, noise_ratio, dbcv, entropy, score`.

**How to pick clustering params** — see [Picking clustering parameters](#picking-clustering-parameters) below. Save chosen params to `output/cluster_config.json`.

### Step 3: Cluster

```bash
./03_cluster.sh
```

Runs HDBSCAN clustering per entry in `cluster_config.json`. Writes cluster assignments back to the DB.

### Step 4: Annotate & rebuild paths

```bash
./04_annotate_rebuild.sh
```

Generates human-readable labels per cluster (contrastive annotation via `claude-sonnet-4-5` + `claude-haiku-4-5`), then rebuilds paths so any cluster merges from annotation flow into trajectory paths. Also prints token-efficiency comparison vs naive per-message annotation. Output: `output/annotate_tokens.txt`.

> Annotation is technically optional (labels are cosmetic), but if you skip it run `episodiq cluster build-paths` yourself — the index in step 07 reads from the path table.

### Step 5 & 6: Tokenize act_obs pairs

```bash
./05_tokenize_grid.sh   # explore vocab size
./06_tokenize.sh        # apply chosen tokenizer config
```

Grid-search then run an HDBSCAN tokenizer over (action, observation) pair embeddings to produce a compact act_obs token vocabulary (the alphabet for the per-window MinHash LSH bands consumed by the retrieval cascade). Same selection heuristic as step 02. Saved config: `output/tokenize_config.json`. Aim for ~50–60 tokens.

### Step 7: Build MinHash index

```bash
./07_index_build.sh
```

Builds `trace_tokens` + per-window MinHash LSH bands per `trajectory_path` row from the AO token mapping. Exports `EPISODIQ_WMH_SIG_SIZE=64`, `EPISODIQ_WMH_NUM_BANDS=64` (= B=64, R=1; bench uses the wider band layout — see top-level README), and `EPISODIQ_RETRIEVAL_WINDOW=10`. Override via env to sweep.

### Step 8: Tune retrieval (top_k, similarity_threshold)

```bash
./08_tune.sh
```

Two phases:

1. **Stratify** — `stratify.py` partitions completed trajectories by (status × length quartile) and stride-interleaves them so any prefix mirrors the full distribution. Cached in `output/stratified_order.json` — delete to regenerate.
2. **Sweep** — `episodiq tune retrieval-sweep` runs leave-one-out retrieval over the first 55 trajectories of the stratified order, sweeping `top_k ∈ {5,10,15,25}` × `similarity_threshold ∈ {0.01..0.50}`. Picks the highest `auc_step60_current` row with coverage in `0.60 ± 0.20`; falls back to overall best AUC if the band is empty.

Output: `output/sweep.csv`, `output/tune_config.json`.

### Step 9: Eval

```bash
./09_eval.sh
```

Runs `eval_metrics.py` on trajectories **after** the tune slice (offset=55 in the same stratified order). Uses the tuned `top_k` + `similarity_threshold`. Output: `output/eval_report.json` with `eval.coverage_step60` + `eval.auc_step60_current`.

### Step 10: Tune path-frequency thresholds

```bash
./10_path_freq.sh
```

`episodiq tune path-freq` computes action-variance entropy percentiles and suggests `EPISODIQ_LOW_ENTROPY` / `EPISODIQ_HIGH_ENTROPY`. Exports the same retrieval config from `tune_config.json` so the entropy distribution it observes matches what step 11's report sees in production.

### Step 11: Demo

```bash
./11_demo.sh
```

Renders every seeded trajectory through `episodiq report uuid --format json -a` and dumps JSONL to `output/demo_reports.jsonl`. Each record carries `instance_id` + `dataset_traj_id` from the seed mapping so reports map back to the SWE-rebench source. Exports `top_k` / `similarity_threshold` from `tune_config.json`.

## Picking clustering parameters

`grid_search.csv` and `tokenize_grid.csv` give you four signals per row: `n_clusters`, `noise_ratio`, `dbcv`, `entropy`. Composite `score` blends them but isn't the final word — read the columns directly. The heuristics that drove the picks in this bench:

1. **Noise is fine. Heatlhy is 10–20%.** A bit of noise means HDBSCAN is being honest about borderline points instead of forcing every embedding into a cluster — and noise points are still tokenized via centroid fallback in the AO step. Be suspicious of two extremes:
   - **`noise_ratio == 0%` with very few clusters** → UMAP collapsed the manifold into one blob; you're not getting granularity, just a trivial partition (high entropy, high DBCV, but useless).
   - **`noise_ratio > 30%`** → params are too strict; you're discarding signal that the downstream retrieval index could've used.
2. **DBCV is the main cluster-quality anchor**; entropy is the guardrail. `dbcv > 0.5` is good, `> 0.7` is excellent. Entropy is usually healthy out of the box (aim for `> 0.8`) but check it explicitly — high DBCV with low entropy is the failure mode where one mega-cluster swallows the population and a few small siblings hang off it. A row with DBCV 0.83 / 8 clusters / 6% noise / entropy 0.95 beats DBCV 0.49 / 263 clusters / 19.7% noise even if the latter has a higher composite `score`.
3. **Fewer, well-separated clusters > many noisy ones.** The downstream MinHash retrieval index needs the transition space to stay dense — too many clusters and any given n-gram only matches a handful of paths, killing coverage.
4. **Filter, then minimize `n_clusters`.** Workflow: filter `noise_ratio ≤ 0.25`, sort by `dbcv` desc, then look at `n_clusters` to break ties. The current `cluster_config.json` was picked by inspecting top-DBCV rows per pair and choosing the cleanest one with non-trivial cluster count.
5. **Per-category — don't reuse one config.** `execute_bash` observations have a different shape than `think` actions. The grid runs per pair specifically so you can tune each independently.

Same rules apply to the AO tokenizer in step 05 — pick a config that gives a small but meaningful token vocabulary (~50–60 tokens) with healthy noise.

## Troubleshooting

**Symptom:** during step 1 the `Embeddings: X / Y` counter slows to a crawl and the log fills with `embed failed (attempt N/10), retrying` warnings.

**Cause:** the embedding endpoint throttles under sustained load (especially on shared services). With the proxy's default settings the connection pool saturates with in-flight slow/retrying requests, new embeds starve, and the postprocess pipeline hits its timeout, dropping affected `messages.embedding` to NULL.

**Fix (the four env vars in Prerequisites):**

- `EPISODIQ_EMBEDDER_MAX_CONNECTIONS=100` — bigger httpx pool.
- `EPISODIQ_EMBEDDER_MAX_RETRIES=5` — fail tasks fast instead of holding workflow slots for tens of minutes.
- `EPISODIQ_EMBEDDER_BACKOFF_MODE=constant` + `EPISODIQ_EMBEDDER_RETRY_BACKOFF=2.0` — bounded retry budget instead of `2^attempt` growth.
- `EPISODIQ_POSTPROCESS_TIMEOUT=600` — pipeline ceiling above the worst-case retry budget.

Production defaults (`episodiq.config.embedder_config.EmbedderConfig`) are conservative: `max_connections=20`, `max_retries=10`, `backoff_mode=exponential`.

## Architecture

```
Dataset (HuggingFace)
    |
    v
Mock Server (port 9999) <-- Proxy (port 8081) <-- seed_via_proxy.py
    |                           |
    |                           v
    |                     PostgreSQL+pgvector (port 5433)
    |                           |
    v                           v
Pre-recorded responses    Embeddings + Clusters + AO tokens + Paths + MinHash index
                                |
                                v
                    CLI: cluster run / annotate / tokenize / index build / tune
                                |
                                v
                    eval_metrics.py  (TrajectoryReportBuilder)
                                |
                                v
                          eval_report.json + demo_reports.jsonl
```
