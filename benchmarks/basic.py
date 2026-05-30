"""Basic naive RAG baseline for agent-trajectory failure prediction.

Pipeline
--------
1. Load trajectories from a HuggingFace dataset (one per instance).
2. Embed every distinct message via the configured embedder (default:
   qwen3-embedding-8b on OpenRouter). Embeddings are cached on disk so
   re-runs are cheap.
3. For each trajectory build snapshots — points in time where retrieval
   runs. A snapshot is emitted after each user|tool message.
4. Query for a snapshot = mean of the last_N message embeddings in its
   prefix. Cosine kNN across snapshots of OTHER trajectories; the top_k
   distinct-trajectory neighbours produce fail_frac.
5. Each candidate must clear `min_cos` (cosine similarity cutoff). Snapshots
   with no surviving candidates are dropped. Per-snapshot fail_frac drives
   the AUC@step current aggregation (most recent filtered snapshot <= s)
   averaged over s in [60, max_step].

Sweep
-----
- ``embedding_dim``       : {1024, 2048}
- ``last_N``              : {1, 3, 5, 10, all}
- ``top_k``               : {5, 10, 25}
- ``min_cos``             : cosine cutoff for candidate filtering
- ``include_initial_task``: {True, False}

Output
------
JSON record per swept config: top_k, last_n, min_cos, dims,
include_initial_task, coverage@s60, AUC@s60.

Usage
-----
    OPENROUTER_API_KEY=sk-or-... \\
        uv run python benchmarks/basic.py \\
            --dataset <hf-dataset-id> \\
            --repo <repo-name> \\
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
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

LAST_NS: list[int | None] = [1, 3, 5, 10, None]   # None = "all"
TOP_KS: list[int] = [5, 10, 25]
MIN_COS_GRID: list[float] = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
EVAL_STEP = 60
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


def retrieve_with_sims(Qn, traj_arr, resolved, max_k):
    """Return (labels per snapshot, cosine sim per candidate per snapshot)."""
    n = Qn.shape[0]
    batch = 512
    per_row_labels: list[list[bool]] = []
    per_row_sims: list[list[float]] = []
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sims = Qn[s:e] @ Qn.T
        for off in range(e - s):
            q = s + off
            own = traj_arr[q]
            sim = sims[off].copy()
            sim[traj_arr == own] = -1.0
            order = np.argsort(-sim)
            seen, picks, sim_picks = set(), [], []
            for j in order:
                t = int(traj_arr[j])
                if t in seen:
                    continue
                seen.add(t)
                picks.append(t)
                sim_picks.append(float(sim[j]))
                if len(picks) >= max_k:
                    break
            per_row_labels.append([not resolved[t] for t in picks])
            per_row_sims.append(sim_picks)
        logger.info("  retrieve %d/%d", e, n)
    return per_row_labels, per_row_sims


def _current_auc_at_step(
    per_traj: dict,
    traj_label: dict,
    min_step: int,
) -> float | None:
    """Mean ROC AUC over s >= min_step using current aggregation."""
    if not per_traj:
        return None
    max_step_per_traj = {
        tid: max(s for s, _ in snaps) for tid, snaps in per_traj.items()
    }
    max_step = max(max_step_per_traj.values())
    aucs = []
    for s in range(min_step, max_step + 1):
        y, scores = [], []
        for tid, snaps in per_traj.items():
            if max_step_per_traj[tid] < s:
                continue
            ordered = sorted([(st, ff) for st, ff in snaps if st <= s])
            if not ordered:
                continue
            y.append(traj_label[tid])
            scores.append(ordered[-1][1])
        if len(set(y)) >= 2:
            aucs.append(roc_auc_score(y, scores))
    return float(np.mean(aucs)) if aucs else None


def evaluate(
    per_row_labels: list[list[bool]],
    per_row_sims: list[list[float]],
    ti_arr: np.ndarray,
    pp_arr: np.ndarray,
    resolved: list[bool],
    top_k: int,
    min_cos: float,
) -> dict:
    per_traj: dict = defaultdict(list)
    for row_i, neighs in enumerate(per_row_labels):
        sims = per_row_sims[row_i]
        # Drop candidates below the threshold, then take the top-k of the
        # filtered list — matches the MinHash prefilter's traj-level cutoff.
        kept_neighs = [
            (lab, sim) for lab, sim in zip(neighs, sims) if sim >= min_cos
        ]
        if not kept_neighs:
            continue
        cand = kept_neighs[:top_k]
        ff = sum(1 for lab, _ in cand if lab) / len(cand)
        per_traj[int(ti_arr[row_i])].append((int(pp_arr[row_i]), ff))
    traj_label = {ti: 1 if not resolved[ti] else 0 for ti in per_traj}
    auc = _current_auc_at_step(per_traj, traj_label, EVAL_STEP)
    # Per-snapshot density at step s: fraction of snapshots with step >= s
    # that themselves passed the similarity filter.
    filtered_set = {
        (ti, st) for ti, snaps in per_traj.items() for st, _ in snaps
    }
    total = sum(
        1 for i in range(len(per_row_labels))
        if int(pp_arr[i]) >= EVAL_STEP
    )
    if total == 0:
        cov = None
    else:
        covered = sum(
            1 for i in range(len(per_row_labels))
            if int(pp_arr[i]) >= EVAL_STEP
            and (int(ti_arr[i]), int(pp_arr[i])) in filtered_set
        )
        cov = covered / total
    return {
        "n_trajectories": len(per_traj),
        "coverage_step60": cov,
        "auc_step60_current": auc,
    }


async def run(args):
    api_key = os.environ.get("OPENROUTER_API_KEY") or args.api_key
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY env var or pass --api-key")

    trajs = load_repo(args.dataset, args.repo, args.limit)
    per_traj_msgs, resolved = extract_messages(trajs)
    logger.info("messages collected for %d trajectories", len(per_traj_msgs))

    uniq, seen = [], set()
    for msgs in per_traj_msgs:
        for _role, t in msgs:
            h = text_id(t)
            if h not in seen:
                seen.add(h)
                uniq.append(t)
    logger.info("uniq texts to embed: %d", len(uniq))

    cache_dir = Path(args.cache_dir) / args.repo.replace("/", "__")
    all_results: dict = {}

    for dims in EMBED_DIMS:
        hash_to_vec = await embed_all(uniq, api_key, cache_dir, dims)
        for include_task in INCLUDE_INITIAL_TASK_VARIANTS:
            per_traj_paths = build_snapshots(per_traj_msgs, include_task)
            total_paths = sum(len(p) for p in per_traj_paths)
            logger.info("=== dims=%d include_initial_task=%s: %d snapshots ===",
                        dims, include_task, total_paths)
            for n in LAST_NS:
                Qn, ti_arr, pp_arr = build_query_matrix(
                    per_traj_paths, hash_to_vec, n)
                logger.info("dims=%d include_task=%s last_n=%s: Q %s",
                            dims, include_task, "all" if n is None else n, Qn.shape)
                per_row_labels, per_row_sims = retrieve_with_sims(
                    Qn, ti_arr, resolved, max(TOP_KS),
                )
                for top_k in TOP_KS:
                    for min_cos in MIN_COS_GRID:
                        res = evaluate(
                            per_row_labels, per_row_sims, ti_arr, pp_arr,
                            resolved, top_k, min_cos,
                        )
                        key = (
                            include_task, dims,
                            "all" if n is None else n, top_k, min_cos,
                        )
                        all_results[key] = res

    return all_results, len(per_traj_msgs), sum(1 for r in resolved if not r)


def _fmt_auc(v):
    return f"{v:.4f}" if v is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description="Basic naive RAG baseline (no bootstrap)")
    parser.add_argument("--dataset", default="nebius/SWE-rebench-openhands-trajectories")
    parser.add_argument("--repo", default="tobymao/sqlglot")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap trajectory count (default: all, stratified)")
    parser.add_argument("--cache-dir", default="benchmarks/.basic_cache")
    parser.add_argument("--output", default="benchmarks/basic_results.json")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter API key (falls back to OPENROUTER_API_KEY env)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results, n_traj, n_fail = asyncio.run(run(args))

    print(f"\n=== {args.dataset} | repo={args.repo} ===")
    print(f"n_trajectories: {n_traj}  n_fail: {n_fail}  fail_rate: {n_fail/n_traj:.3f}\n")

    def _fmt_cov(c):
        return f"{c * 100:.1f}%" if c is not None else "n/a"

    for include_task in INCLUDE_INITIAL_TASK_VARIANTS:
        for dims in EMBED_DIMS:
            print(f"\n--- include_initial_task={include_task}  dims={dims} ---")
            print(f"{'last_N':>6} {'top_k':>5} {'min_cos':>7}  "
                  f"{'cov@s60':>7} {'AUC@s60':>9}")
            for n in LAST_NS:
                n_lab = "all" if n is None else str(n)
                for top_k in TOP_KS:
                    for min_cos in MIN_COS_GRID:
                        r = results.get((include_task, dims, "all" if n is None else n, top_k, min_cos))
                        if r is None:
                            continue
                        print(
                            f"{n_lab:>6} {top_k:>5} {min_cos:>7.2f}  "
                            f"{_fmt_cov(r['coverage_step60']):>7} "
                            f"{_fmt_auc(r['auc_step60_current']):>9}"
                        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for (include_task, dims, n, top_k, min_cos), r in results.items():
        payload.append({
            "include_initial_task": include_task,
            "embedding_dim": dims,
            "last_n": n,
            "top_k": top_k,
            "min_cos": min_cos,
            **r,
        })
    json.dump({
        "dataset": args.dataset,
        "repo": args.repo,
        "n_trajectories": n_traj,
        "n_fail": n_fail,
        "fail_rate": round(n_fail / n_traj, 3),
        "results": payload,
    }, open(args.output, "w"), indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
