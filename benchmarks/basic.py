"""Basic naive RAG baseline for agent-trajectory failure prediction.

Mirrors the demo_eval pipeline's tune/eval split (`08_tune.sh` /
`09_eval.sh`):

  1. Load + dedupe trajectories from HuggingFace.
  2. ``Random(--shuffle-seed).shuffle`` the trajectory list — same seed
     in ``episodiq tune retrieval-sweep --shuffle-seed S`` gives a
     parallel slice on the production cascade.
  3. **Tune** phase: full grid over ``(dims, include_initial_task,
     last_n, top_k)`` on the first ``--tune-limit`` trajectories
     (leave-one-out — queries from tune, corpus = tune \\ self-traj).
     Pick the config maximising AUC@step50 under ``--eval-metric``
     (default ``cummeanmax`` to match prod sweep).
  4. **Eval** phase: the single tune winner, queries = remainder of
     the shuffled list, corpus = tune slice only.

Output JSON carries the winner config, the tune-side AUC under all 3
``SIMILARITY_METRICS`` (cummax / cummean / cummeanmax), and the
eval-side AUCs.

Usage
-----
    OPENROUTER_API_KEY=sk-or-... \\
        uv run python benchmarks/basic.py \\
            --dataset <hf-dataset-id> \\
            --repo <repo-name> \\
            --shuffle-seed 42 --tune-limit 55 \\
            --output benchmarks/basic_results.json

Dataset layout
--------------
Dataset rows must expose:
  - ``trajectory`` : list of {role, content} message dicts (role in
    {"system", "user", "assistant", "tool"}).
  - ``resolved``   : bool. True = success, False = failure.
  - ``repo``       : str. Used by --repo to filter.
  - ``instance_id``: str. Used to deduplicate (1 trajectory per instance).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from episodiq.analytics.metrics import (
    SIMILARITY_METRICS, bootstrap_aucs, compute_metric_curves, weighted_aucs,
)

logger = logging.getLogger(__name__)

LAST_NS: list[int | None] = [*range(1, 21), None]   # 1..20 + unbounded
TOP_KS: list[int] = list(range(1, 51))   # 1..50
EVAL_STEP = 50
EMBED_DIMS = [1024, 2048]
INCLUDE_INITIAL_TASK_VARIANTS = [True, False]
EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_BATCH = 16
MAX_CHARS = 40000   # cap per chunk; ~10k tokens, within qwen3-8b 32k context
def content_text(msg: dict) -> str:
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if not isinstance(p, dict):
                continue
            if "text" in p:
                parts.append(p["text"])
            elif "content" in p:
                parts.append(str(p["content"]))
        return "".join(parts)
    return ""


def text_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def load_repo(dataset: str, repo: str, limit: int | None) -> list[dict]:
    from datasets import load_dataset
    logger.info("Loading %s, filtering to repo=%s ...", dataset, repo)
    ds = load_dataset(dataset, split="train")
    by_inst: dict[str, dict] = {}
    for row in ds:
        if (row.get("repo") or "") != repo:
            continue
        iid = row.get("instance_id")
        if iid and iid not in by_inst:
            by_inst[iid] = row
    insts = list(by_inst.values())
    logger.info("1-traj-per-instance: %d trajectories", len(insts))
    if limit and limit < len(insts):
        # Stratify by outcome so the cap preserves base fail rate.
        fail = [r for r in insts if not r["resolved"]]
        succ = [r for r in insts if r["resolved"]]
        fr = len(fail) / len(insts)
        n_fail = round(limit * fr)
        n_succ = limit - n_fail
        insts = fail[:n_fail] + succ[:n_succ]
        logger.info("limited to %d (%d fail / %d succ)",
                    len(insts), n_fail, n_succ)
    return insts


def extract_messages(trajs: list[dict]) -> tuple[list[list[tuple[str, str]]], list[bool]]:
    """Each pair = (role, full_text); truncation happens at embed time."""
    out_msgs: list[list[tuple[str, str]]] = []
    out_resolved: list[bool] = []
    for t in trajs:
        msgs = t.get("trajectory") or []
        pairs = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            text = content_text(m).strip()
            if not text:
                continue
            pairs.append((role, text))
        out_msgs.append(pairs)
        out_resolved.append(bool(t["resolved"]))
    return out_msgs, out_resolved


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


async def embed_remote(texts: list[str], api_key: str, dims: int) -> np.ndarray:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": EMBED_MODEL, "input": texts, "dimensions": dims}
    async with httpx.AsyncClient(timeout=180.0) as client:
        for attempt in range(5):
            try:
                r = await client.post(EMBED_URL, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()["data"]
                vecs = np.asarray([d["embedding"] for d in data], dtype=np.float32)
                if vecs.shape[1] != dims:
                    raise ValueError(f"got dim {vecs.shape[1]} expected {dims}")
                return vecs
            except (httpx.HTTPError, ValueError, KeyError) as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                logger.warning("embed retry %d in %ds: %s", attempt + 1, wait, e)
                await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def embed_all(
    uniq_texts: list[str], api_key: str, cache_dir: Path, dims: int,
) -> dict[str, np.ndarray]:
    """Embed each text. Long texts are split into chunks, embedded
    independently, then mean-pooled. Result maps text_id(text) → vector.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx_file = cache_dir / f"idx_{dims}.json"
    arr_file = cache_dir / f"embs_{dims}.npy"
    cache: dict[str, int] = {}
    arr = np.zeros((0, dims), dtype=np.float32)
    if idx_file.exists() and arr_file.exists():
        cache = json.load(open(idx_file))
        arr_old = np.load(arr_file)
        if arr_old.shape[1] == dims:
            arr = arr_old
            logger.info("dims=%d: loaded %d cached chunk embeddings", dims, len(cache))
        else:
            logger.warning("dims=%d: cache dim mismatch (%s), rebuilding",
                           dims, arr_old.shape)
            cache = {}

    long_count = sum(1 for t in uniq_texts if len(t) > MAX_CHARS)
    total_chunks = sum(max(1, (len(t) + MAX_CHARS - 1) // MAX_CHARS) for t in uniq_texts)
    logger.info("dims=%d: %d texts (%d > MAX_CHARS=%d → chunked, %d total chunks)",
                dims, len(uniq_texts), long_count, MAX_CHARS, total_chunks)

    text_to_chunks: dict[str, list[str]] = {t: split_into_chunks(t, MAX_CHARS) for t in uniq_texts}
    uniq_chunks: list[str] = []
    seen_chunk_hashes: set[str] = set()
    for chunks in text_to_chunks.values():
        for c in chunks:
            h = text_id(c)
            if h not in seen_chunk_hashes:
                seen_chunk_hashes.add(h)
                uniq_chunks.append(c)

    todo = [t for t in uniq_chunks if text_id(t) not in cache]
    logger.info("dims=%d: need %d new chunk embeddings (of %d uniq chunks)",
                dims, len(todo), len(uniq_chunks))
    if todo:
        new_rows = []
        for i in range(0, len(todo), EMBED_BATCH):
            chunk = todo[i:i + EMBED_BATCH]
            vecs = await embed_remote(chunk, api_key, dims)
            new_rows.append(vecs)
            if (i // EMBED_BATCH) % 10 == 0:
                logger.info("  embedded %d/%d", i + len(chunk), len(todo))
        new_arr = np.vstack(new_rows)
        start = len(cache)
        for j, t in enumerate(todo):
            cache[text_id(t)] = start + j
        arr = np.vstack([arr, new_arr])
        np.save(arr_file, arr)
        json.dump(cache, open(idx_file, "w"))
        logger.info("dims=%d: saved %d total chunk embeddings", dims, len(cache))

    out: dict[str, np.ndarray] = {}
    for t in uniq_texts:
        chunks = text_to_chunks[t]
        chunk_vecs = np.vstack([arr[cache[text_id(c)]] for c in chunks])
        out[text_id(t)] = chunk_vecs.mean(axis=0)
    return out


def build_snapshots(per_traj_msgs, include_initial_task: bool):
    """Per trajectory, list of (snap_pos, [msg_texts up to and including snap]).

    A snapshot is emitted after each user|tool message (system role excluded
    from retrieval queries). If ``include_initial_task`` is False, the very
    first user message (typically the task description) is skipped.
    """
    per_traj_paths = []
    for msgs in per_traj_msgs:
        prefix_texts = []
        body_idx = []
        for role, text in msgs:
            if role == "system":
                continue
            prefix_texts.append(text)
            if role in ("user", "tool"):
                body_idx.append(len(prefix_texts) - 1)
        start = 0 if include_initial_task else 1
        paths = [(snap_pos, prefix_texts[: snap_pos + 1])
                 for snap_pos in body_idx[start:]]
        per_traj_paths.append(paths)
    return per_traj_paths


def build_query_matrix(per_traj_paths, hash_to_vec, last_n):
    Q, ti_arr, pp_arr = [], [], []
    for ti, paths in enumerate(per_traj_paths):
        for k, (_snap_pos, prefix) in enumerate(paths):
            # Skip snapshots whose history is shorter than the requested
            # window — for last_n=N the first usable snapshot has N
            # messages in its prefix. None = unbounded; always allowed.
            if last_n is not None and len(prefix) < last_n:
                continue
            chunk = prefix if last_n is None else prefix[-last_n:]
            ev = [hash_to_vec[text_id(t)] for t in chunk if text_id(t) in hash_to_vec]
            if not ev:
                continue
            q = np.vstack(ev).mean(axis=0)
            Q.append(q)
            ti_arr.append(ti)
            pp_arr.append(k)
    Qn = np.vstack(Q).astype(np.float32)
    norms = np.linalg.norm(Qn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Qn = Qn / norms
    return Qn, np.asarray(ti_arr), np.asarray(pp_arr)


def retrieve_with_sims(
    Qq, Qc, ti_q, ti_c, resolved, max_k,
):
    """Cosine kNN: rows of ``Qq`` query against the corpus rows of
    ``Qc``. ``ti_q``/``ti_c`` are trajectory indices for each row;
    candidates sharing the query's trajectory are excluded (covers
    leave-one-out when ``Qq is Qc`` and is a no-op for the disjoint
    tune/eval split). Returns per-query ``[bool labels]`` and
    ``[float sims]`` for up to ``max_k`` distinct corpus trajectories.
    """
    n_q = Qq.shape[0]
    batch = 512
    per_row_labels: list[list[bool]] = []
    per_row_sims: list[list[float]] = []
    for s in range(0, n_q, batch):
        e = min(s + batch, n_q)
        sims = Qq[s:e] @ Qc.T
        for off in range(e - s):
            q = s + off
            own = ti_q[q]
            sim = sims[off].copy()
            sim[ti_c == own] = -1.0
            order = np.argsort(-sim)
            seen, picks, sim_picks = set(), [], []
            for j in order:
                t = int(ti_c[j])
                if t in seen:
                    continue
                seen.add(t)
                picks.append(t)
                sim_picks.append(float(sim[j]))
                if len(picks) >= max_k:
                    break
            per_row_labels.append([not resolved[t] for t in picks])
            per_row_sims.append(sim_picks)
        logger.info("  retrieve %d/%d", e, n_q)
    return per_row_labels, per_row_sims


def _split_query_matrix(Qn, ti_arr, pp_arr, traj_mask):
    """Slice ``(Qn, ti_arr, pp_arr)`` to the rows whose trajectory
    index is in ``traj_mask`` (bool array, length = #trajectories).
    """
    keep = traj_mask[ti_arr]
    return Qn[keep], ti_arr[keep], pp_arr[keep]


def _auc_at_step(
    per_traj: dict,
    traj_label: dict,
    min_step: int,
) -> dict[str, float | None]:
    """Weighted-AUC per metric — thin shim over the shared utility.
    ``traj_label`` is 1=failure / 0=success; the utility expects
    string status, so map here."""
    status = {
        tid: "failure" if lbl else "success"
        for tid, lbl in traj_label.items()
    }
    weighted = weighted_aucs(per_traj, status, eval_min_step=min_step)
    return {m: weighted.get(m) for m in SIMILARITY_METRICS}


def _build_per_traj(
    per_row_labels: list[list[bool]],
    per_row_sims: list[list[float]],
    ti_arr: np.ndarray,
    pp_arr: np.ndarray,
    top_k: int,
) -> dict:
    """Collapse per-snapshot retrieval rows into the ``{tid: [(step,
    fail_sim), ...]}`` shape consumed by the metric utilities."""
    per_traj: dict = defaultdict(list)
    for row_i, neighs in enumerate(per_row_labels):
        cand = neighs[:top_k]
        if not cand:
            continue
        ff = sum(1 for lab in cand if lab) / len(cand)
        per_traj[int(ti_arr[row_i])].append((int(pp_arr[row_i]), ff))
    return per_traj


def evaluate(
    per_row_labels: list[list[bool]],
    per_row_sims: list[list[float]],
    ti_arr: np.ndarray,
    pp_arr: np.ndarray,
    resolved: list[bool],
    top_k: int,
) -> dict:
    per_traj = _build_per_traj(
        per_row_labels, per_row_sims, ti_arr, pp_arr, top_k,
    )
    traj_label = {ti: 1 if not resolved[ti] else 0 for ti in per_traj}
    aucs = _auc_at_step(per_traj, traj_label, EVAL_STEP)
    return {
        "n_trajectories": len(per_traj),
        **{f"auc_step{EVAL_STEP}_{m}": aucs[m] for m in SIMILARITY_METRICS},
    }


def evaluate_with_ci(
    per_row_labels: list[list[bool]],
    per_row_sims: list[list[float]],
    ti_arr: np.ndarray,
    pp_arr: np.ndarray,
    resolved: list[bool],
    top_k: int,
    *,
    n_boot: int = 200,
    boot_seed: int = 42,
) -> dict:
    """Point-estimate AUC per metric + per-trajectory bootstrap 95% CI.
    Eval-only — the sweep does NOT pay this cost per trial.
    """
    per_traj = _build_per_traj(
        per_row_labels, per_row_sims, ti_arr, pp_arr, top_k,
    )
    status = {
        ti: "failure" if not resolved[ti] else "success" for ti in per_traj
    }
    cis = bootstrap_aucs(
        per_traj, status, eval_min_step=EVAL_STEP,
        n_boot=n_boot, seed=boot_seed,
    )
    base = evaluate(
        per_row_labels, per_row_sims, ti_arr, pp_arr, resolved, top_k,
    )
    base["auc_ci_per_metric"] = {
        m: {"lo": ci.lo, "hi": ci.hi, "mean": ci.mean}
        for m, ci in cis.items()
    }
    return base


async def run(args):
    api_key = os.environ.get("OPENROUTER_API_KEY") or args.api_key
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY env var or pass --api-key")

    trajs = load_repo(args.dataset, args.repo, args.limit)
    per_traj_msgs, resolved = extract_messages(trajs)
    n_total = len(per_traj_msgs)
    logger.info("messages collected for %d trajectories", n_total)

    # Sort by instance_id, seed-shuffle, then (optionally) stratify by
    # ``resolved`` so the tune/eval prefix inherits the population's
    # fail-rate regardless of seed — mirrors
    # ``episodiq tune retrieval-sweep --stratify-field status`` on the
    # DB side.
    indices = list(range(n_total))
    indices.sort(key=lambda i: str(trajs[i].get("instance_id") or ""))
    random.Random(args.shuffle_seed).shuffle(indices)
    if args.stratify:
        # Group → proportional interleave. Each item's ticket is its
        # within-group position normalised to (0, 1].
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for i in indices:
            groups[bool(resolved[i])].append(i)
        tickets = []
        for cls, group in groups.items():
            n = len(group)
            for k, i in enumerate(group):
                tickets.append(((k + 1) / n, cls, i))
        tickets.sort(key=lambda x: (x[0], int(x[1])))
        indices = [i for _, _, i in tickets]
    per_traj_msgs = [per_traj_msgs[i] for i in indices]
    resolved = [resolved[i] for i in indices]

    tune_limit = min(args.tune_limit, n_total)
    if tune_limit <= 0 or tune_limit >= n_total:
        raise SystemExit(
            f"--tune-limit must be in (0, {n_total}); got {args.tune_limit}",
        )
    tune_mask = np.zeros(n_total, dtype=bool)
    tune_mask[:tune_limit] = True
    eval_mask = ~tune_mask
    logger.info(
        "split: tune=%d eval=%d  seed=%d",
        tune_mask.sum(), eval_mask.sum(), args.shuffle_seed,
    )

    uniq, seen = [], set()
    for msgs in per_traj_msgs:
        for _role, t in msgs:
            h = text_id(t)
            if h not in seen:
                seen.add(h)
                uniq.append(t)
    logger.info("uniq texts to embed: %d", len(uniq))

    cache_dir = Path(args.cache_dir) / args.repo.replace("/", "__")
    tune_results: dict = {}

    # --- Tune phase: full grid; queries=tune slice, corpus=ALL --------
    # Mirrors the cascade sweep: tune-slice snapshots drive AUC for the
    # hyperparam picker, but the corpus is the full population (LSH
    # alt-table in production indexes everything; here Qc=Qn).
    for dims in EMBED_DIMS:
        hash_to_vec = await embed_all(uniq, api_key, cache_dir, dims)
        for include_task in INCLUDE_INITIAL_TASK_VARIANTS:
            per_traj_paths = build_snapshots(per_traj_msgs, include_task)
            total_paths = sum(len(p) for p in per_traj_paths)
            logger.info("=== tune: dims=%d include_initial_task=%s: %d snapshots ===",
                        dims, include_task, total_paths)
            for n in LAST_NS:
                Qn, ti_arr, pp_arr = build_query_matrix(
                    per_traj_paths, hash_to_vec, n)
                # Restrict queries to the tune slice; corpus stays full.
                Qt, ti_t, pp_t = _split_query_matrix(
                    Qn, ti_arr, pp_arr, tune_mask,
                )
                logger.info("tune dims=%d include_task=%s last_n=%s: Qq=%s corpus=%s",
                            dims, include_task, "all" if n is None else n,
                            Qt.shape, Qn.shape)
                per_row_labels, per_row_sims = retrieve_with_sims(
                    Qt, Qn, ti_t, ti_arr, resolved, max(TOP_KS),
                )
                for top_k in TOP_KS:
                    res = evaluate(
                        per_row_labels, per_row_sims, ti_t, pp_t,
                        resolved, top_k,
                    )
                    key = (
                        include_task, dims,
                        "all" if n is None else n, top_k,
                    )
                    tune_results[key] = res

    # --- Pick top config per metric, eval each on the held-out slice.
    # The naive RAG baseline gets its strongest possible footing: for
    # every metric ``m`` we let it pick its OWN best hyperparameters
    # (those that maximise tune AUC under ``m``), then evaluate that
    # config separately. Reporting all three winners with CIs shows
    # the baseline's best shot across metrics — no single-metric
    # restriction biases the comparison against it.
    winners_per_metric: dict[str, dict] = {}
    for m in SIMILARITY_METRICS:
        col = f"auc_step{EVAL_STEP}_{m}"
        def _tune_auc(k, _col=col):
            v = tune_results[k].get(_col)
            return v if v is not None else -1.0
        best_key = max(tune_results, key=_tune_auc)
        best_include, best_dims, best_last_n, best_top_k = best_key
        tune_auc = _tune_auc(best_key)
        logger.info(
            "metric=%s tune winner: include_task=%s dims=%d last_n=%s "
            "top_k=%d tune_auc=%.4f",
            m, best_include, best_dims, best_last_n, best_top_k, tune_auc,
        )

        # Eval the per-metric winner against the held-out slice.
        hash_to_vec = await embed_all(uniq, api_key, cache_dir, best_dims)
        per_traj_paths = build_snapshots(per_traj_msgs, best_include)
        last_n_val = None if best_last_n == "all" else int(best_last_n)
        Qn, ti_arr, pp_arr = build_query_matrix(
            per_traj_paths, hash_to_vec, last_n_val,
        )
        Qq, ti_q, pp_q = _split_query_matrix(Qn, ti_arr, pp_arr, eval_mask)
        per_row_labels, per_row_sims = retrieve_with_sims(
            Qq, Qn, ti_q, ti_arr, resolved, best_top_k,
        )
        eval_res = evaluate_with_ci(
            per_row_labels, per_row_sims, ti_q, pp_q, resolved, best_top_k,
            n_boot=args.n_boot, boot_seed=args.boot_seed,
        )
        # Per-step AUC curve for THIS metric's winner, for the
        # AUC-vs-step plot. Floored at curve_min_step (not EVAL_STEP) so
        # the curve spans the early trajectory; the headline eval_auc
        # below stays anchored at EVAL_STEP.
        eval_per_traj = _build_per_traj(
            per_row_labels, per_row_sims, ti_q, pp_q, best_top_k,
        )
        eval_status = {
            ti: "failure" if not resolved[ti] else "success"
            for ti in eval_per_traj
        }
        m_curve = compute_metric_curves(
            eval_per_traj, eval_status, eval_min_step=args.curve_min_step,
        ).get(m)
        step_curve = (
            [{"step": sa.step, "auc": sa.auc, "n_active": sa.n_active}
             for sa in m_curve.per_step]
            if m_curve else []
        )
        col_m = f"auc_step{EVAL_STEP}_{m}"
        ci = eval_res.get("auc_ci_per_metric", {}).get(m)
        winners_per_metric[m] = {
            "include_initial_task": best_include,
            "embedding_dim": best_dims,
            "last_n": best_last_n,
            "top_k": best_top_k,
            "tune_auc": tune_auc,
            "eval_auc": eval_res.get(col_m),
            "eval_auc_ci": (
                {"lo": ci["lo"], "hi": ci["hi"]} if ci else None
            ),
            "eval_all_metrics": {
                m2: eval_res.get(f"auc_step{EVAL_STEP}_{m2}")
                for m2 in SIMILARITY_METRICS
            },
            "eval_ci_all_metrics": eval_res.get("auc_ci_per_metric"),
            "eval_step_curve": step_curve,
        }
    return tune_results, winners_per_metric, n_total, sum(1 for r in resolved if not r)


def _fmt_auc(v):
    return f"{v:.4f}" if v is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(
        description="Basic naive RAG baseline with tune/eval split",
    )
    parser.add_argument("--dataset", default="nebius/SWE-rebench-openhands-trajectories")
    parser.add_argument("--repo", default="tobymao/sqlglot")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap trajectory count (default: all, stratified)")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Random(seed).shuffle of trajectories before "
                             "tune/eval split. Use the same seed in "
                             "`episodiq tune retrieval-sweep --shuffle-seed S` "
                             "for a parallel split on the cascade.")
    parser.add_argument("--tune-limit", type=int, default=55,
                        help="number of trajectories in the tune slice "
                             "(first N after shuffle); the rest become eval queries.")
    parser.add_argument("--stratify", action="store_true",
                        help="Proportionally interleave shuffled trajectories "
                             "by ``resolved`` so tune/eval prefix preserves the "
                             "population fail-rate.")
    parser.add_argument("--cache-dir", default="benchmarks/.basic_cache")
    parser.add_argument("--output", default="benchmarks/basic_results.json")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter API key (falls back to OPENROUTER_API_KEY env)")
    parser.add_argument("--curve-min-step", type=int, default=1,
                        help="Floor for the per-step AUC curve dumped per "
                             "winner for plotting (independent of EVAL_STEP, "
                             "which anchors the headline weighted AUC).")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Bootstrap draws for per-metric eval AUC CI (95 pct).")
    parser.add_argument("--boot-seed", type=int, default=42,
                        help="RNG seed for bootstrap reproducibility.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tune_results, winners_per_metric, n_traj, n_fail = asyncio.run(run(args))

    print(f"\n=== {args.dataset} | repo={args.repo} ===")
    print(f"n_trajectories: {n_traj}  n_fail: {n_fail}  fail_rate: {n_fail/n_traj:.3f}")
    print(f"shuffle_seed: {args.shuffle_seed}  tune_limit: {args.tune_limit}\n")

    print("=== Per-metric winners (each metric picks its OWN best config) ===")
    for m in SIMILARITY_METRICS:
        w = winners_per_metric[m]
        ci = w.get("eval_auc_ci")
        ci_str = f"  [{ci['lo']:.4f}, {ci['hi']:.4f}]" if ci else ""
        print(f"\n--- metric: {m} ---")
        print(f"  include_initial_task = {w['include_initial_task']}")
        print(f"  embedding_dim        = {w['embedding_dim']}")
        print(f"  last_n               = {w['last_n']}")
        print(f"  top_k                = {w['top_k']}")
        print(f"  tune AUC@s{EVAL_STEP}        = {_fmt_auc(w['tune_auc'])}")
        print(f"  eval AUC@s{EVAL_STEP}        = {_fmt_auc(w['eval_auc'])}{ci_str}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tune_payload = []
    for (include_task, dims, n, top_k), r in tune_results.items():
        tune_payload.append({
            "include_initial_task": include_task,
            "embedding_dim": dims,
            "last_n": n,
            "top_k": top_k,
            **r,
        })
    json.dump({
        "dataset": args.dataset,
        "repo": args.repo,
        "n_trajectories": n_traj,
        "n_fail": n_fail,
        "fail_rate": round(n_fail / n_traj, 3),
        "shuffle_seed": args.shuffle_seed,
        "tune_limit": args.tune_limit,
        "stratified": bool(args.stratify),
        "winners_per_metric": winners_per_metric,
        "tune_grid": tune_payload,
    }, open(args.output, "w"), indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
