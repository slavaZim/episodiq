"""
Naive token efficiency comparison for cluster annotation.

Samples N random messages from DB, annotates each individually (map-reduce summarize
if long, then annotate), and extrapolates to the full message count. Compares against
actual Episodiq token usage parsed from the annotator output.

Uses the same MapReduceSummarizer as the main pipeline so token counts are comparable.
The summarizer is included in the naive baseline intentionally: feeding raw long messages
directly to the annotator produces unreliable results — even claude-sonnet-4-5 tends to
follow instructions embedded in the message content, hallucinate, or lose focus on the
annotation task when given very large inputs. Summarizing first is a prerequisite for
stable annotation quality, not an Episodiq-specific optimization.

Usage: set env vars and run via `uv run python token_efficiency.py`
  EPISODIQ_OPENAI_BASE_URL, EPISODIQ_OPENAI_API_KEY   — LLM endpoint (OpenAI-compatible)
  PSQL_URL                          — PostgreSQL connection string
  ANNOTATE_MODEL                    — annotator model
  SUMMARIZER_MODEL                  — summarizer model
  NAIVE_SAMPLE                      — number of messages to sample (default 30)
  ANNOT_OUTPUT_FILE                 — path to file with annotator CLI output
  OUTPUT_FILE                       — path to save the report (optional)
"""

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass

from episodiq.api_adapters.openai import OpenAICompletionsAdapter, OpenAIConfig
from episodiq.clustering.annotator.generator import (
    OpenAICompletionsGenerator,
    system_message,
    user_message,
)
from episodiq.clustering.annotator.summarizer import MapReduceSummarizer

PSQL_URL    = os.environ["PSQL_URL"]
ANNOTATE    = os.environ["ANNOTATE_MODEL"]
SUMMARIZER  = os.environ["SUMMARIZER_MODEL"]
N_SAMPLE    = int(os.environ.get("NAIVE_SAMPLE", "100"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "")


@dataclass
class TokenCounts:
    annotator_in:   int = 0
    annotator_out:  int = 0
    summarizer_in:  int = 0
    summarizer_out: int = 0

    @property
    def total(self) -> int:
        return self.annotator_in + self.annotator_out + self.summarizer_in + self.summarizer_out


# --- Parse Episodiq tokens from annotator CLI output ---
def parse_episodiq_tokens(output: str) -> tuple[TokenCounts, int]:
    counts = TokenCounts()

    m = re.search(r'annotator\s+\([^)]+\):\s+([\d,]+)\s+input tokens,\s+([\d,]+)\s+output tokens', output)
    if m:
        counts.annotator_in  = int(m.group(1).replace(',', ''))
        counts.annotator_out = int(m.group(2).replace(',', ''))

    m = re.search(r'summarizer\s+\([^)]+\):\s+([\d,]+)\s+input tokens,\s+([\d,]+)\s+output tokens', output)
    if m:
        counts.summarizer_in  = int(m.group(1).replace(',', ''))
        counts.summarizer_out = int(m.group(2).replace(',', ''))

    # Fallback: old single-line format
    if counts.annotator_in == 0:
        m_in  = re.search(r'([\d,]+)\s+input tokens',  output)
        m_out = re.search(r'([\d,]+)\s+output tokens', output)
        if m_in:
            counts.annotator_in  = int(m_in.group(1).replace(',', ''))
        if m_out:
            counts.annotator_out = int(m_out.group(1).replace(',', ''))

    m_ann = re.search(r'^(\d+)\s+annotations', output, re.MULTILINE)
    n_ann = int(m_ann.group(1)) if m_ann else 0
    return counts, n_ann


# --- DB helpers ---
def psql(sql: str) -> str:
    r = subprocess.run(["psql", PSQL_URL, "-t", "-A", "-c", sql],
                       capture_output=True, text=True)
    return r.stdout.strip()


def query_message_counts() -> tuple[int, int]:
    parts = psql("SELECT COUNT(*), COUNT(DISTINCT trajectory_id) FROM messages WHERE role != 'system'").split('|')
    return int(parts[0]) if parts else 0, int(parts[1]) if len(parts) > 1 else 0


def sample_messages(n: int) -> list[tuple[str, str]]:
    """Stratified sample by message length: divide into n_buckets length buckets,
    draw evenly from each so the sample represents the full length distribution.
    Excludes system messages (not annotated by Episodiq either)."""
    n_buckets = min(n, 10)
    per_bucket = max(1, n // n_buckets)
    sql = f"""
        WITH bucketed AS (
            SELECT role, content::text AS content,
                   NTILE({n_buckets}) OVER (ORDER BY length(content::text)) AS bucket
            FROM messages
            WHERE role != 'system'
        ),
        ranked AS (
            SELECT role, content, bucket,
                   ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY random()) AS rn
            FROM bucketed
        )
        SELECT role, content FROM ranked WHERE rn <= {per_bucket}
        ORDER BY random()
    """
    rows = []
    for line in psql(sql).split('\n'):
        if '|' in line:
            role, content_json = line.split('|', 1)
            rows.append((role.strip(), content_json.strip()))
    return rows


# --- Message formatting ---
def format_message(content_json: str) -> str:
    try:
        content = json.loads(content_json)
        parts = []
        if isinstance(content, list):
            for block in content:
                t = block.get('type')
                if t == 'text':
                    parts.append(block.get('text', ''))
                elif t == 'tool_call':
                    parts.append(f"tool: {block.get('tool_name', '')}\n{json.dumps(block.get('input', {}))}")
                elif t == 'tool_response':
                    parts.append(f"tool: {block.get('tool_name', '')}\n{json.dumps(block.get('tool_response', ''))}")
        elif isinstance(content, str):
            parts.append(content)
        return '\n'.join(parts)
    except Exception:
        return content_json[:2000]


# --- Prompts ---
ANNOTATE_SYS = (
    "You annotate individual messages from an LLM agent trajectory.\n"
    "Given a single message, produce a single short phrase (5-10 words) describing what it is about.\n"
    "Do NOT mention personal data or specific values. Describe the pattern, not specific details.\n"
    "Reply with ONLY the annotation phrase, nothing else."
)


# --- Annotate one message (map-reduce summarize if long, then annotate) ---
async def annotate_one(
    role: str,
    content_json: str,
    annotate_gen: OpenAICompletionsGenerator,
    summarizer: MapReduceSummarizer,
) -> None:
    text = format_message(content_json)
    if len(text) > 1000:
        text = await summarizer.summarize(text)
    await annotate_gen.generate(
        [system_message(ANNOTATE_SYS), user_message(text)],
        max_tokens=50,
    )


async def main() -> None:
    with open(os.environ["ANNOT_OUTPUT_FILE"]) as f:
        epi_output = f.read()

    epi, n_ann = parse_episodiq_tokens(epi_output)
    n_messages, n_trajs = query_message_counts()
    sample_rows = sample_messages(N_SAMPLE)

    print(f"  Sampled {len(sample_rows)} messages from DB", flush=True)

    adapter = OpenAICompletionsAdapter(OpenAIConfig())
    await adapter.startup()
    try:
        annotate_gen  = OpenAICompletionsGenerator(adapter, ANNOTATE)
        summarizer_gen = OpenAICompletionsGenerator(adapter, SUMMARIZER)
        summarizer    = MapReduceSummarizer(summarizer_gen)

        sem = asyncio.Semaphore(8)

        async def run_one(i: int, row: tuple) -> None:
            async with sem:
                try:
                    await annotate_one(*row, annotate_gen, summarizer)
                except Exception as e:
                    print(f"  Warning: sample {i+1} failed: {e}", flush=True)

        await asyncio.gather(*[run_one(i, r) for i, r in enumerate(sample_rows)])

        naive = TokenCounts(
            annotator_in=annotate_gen.total_usage.input_tokens,
            annotator_out=annotate_gen.total_usage.output_tokens,
            summarizer_in=summarizer_gen.total_usage.input_tokens,
            summarizer_out=summarizer_gen.total_usage.output_tokens,
        )
    finally:
        await adapter.shutdown()

    n_sampled = len(sample_rows)
    if n_sampled == 0:
        print("  (no messages sampled)")
        return

    # Extrapolate naive to full dataset
    scale = n_messages / n_sampled
    ext = TokenCounts(
        annotator_in   = int(naive.annotator_in   * scale),
        annotator_out  = int(naive.annotator_out  * scale),
        summarizer_in  = int(naive.summarizer_in  * scale),
        summarizer_out = int(naive.summarizer_out * scale),
    )

    reduction = n_messages / n_ann if n_ann else 0

    # At 1,000 trajectories: Episodiq annotation was done on current dataset,
    # naive scales linearly × (1000 / n_trajs).
    naive_at_1k = int(ext.total * (1000 / n_trajs))

    lines = [
        f"  Messages in DB        : {n_messages:>8,}  ({n_trajs} trajectories)",
        f"  Sampled for estimate  : {n_sampled:>8,}  (×{scale:.0f} extrapolation)",
        f"  Clusters annotated    : {n_ann:>8,}  ({reduction:.1f}x reduction vs messages)",
        "",
        f"  {'':28} {'input tk':>12}  {'output tk':>10}  {'total tk':>12}",
        f"  {'-'*68}",
        f"  {'Episodiq annotator':28} {epi.annotator_in:>12,}  {epi.annotator_out:>10,}  {epi.annotator_in+epi.annotator_out:>12,}",
        f"  {'Episodiq summarizer':28} {epi.summarizer_in:>12,}  {epi.summarizer_out:>10,}  {epi.summarizer_in+epi.summarizer_out:>12,}",
        f"  {'Episodiq total':28} {epi.annotator_in+epi.summarizer_in:>12,}  {epi.annotator_out+epi.summarizer_out:>10,}  {epi.total:>12,}",
        f"  {'-'*68}",
        f"  {'Naive annotator (extrap.)':28} {ext.annotator_in:>12,}  {ext.annotator_out:>10,}  {ext.annotator_in+ext.annotator_out:>12,}",
        f"  {'Naive summarizer (extrap.)':28} {ext.summarizer_in:>12,}  {ext.summarizer_out:>10,}  {ext.summarizer_in+ext.summarizer_out:>12,}",
        f"  {'Naive total':28} {ext.annotator_in+ext.summarizer_in:>12,}  {ext.annotator_out+ext.summarizer_out:>10,}  {ext.total:>12,}",
        "",
    ]

    if ext.total > 0 and epi.total > 0:
        saved_tok = ext.total - epi.total
        pct_tok   = 100 * saved_tok / ext.total
        lines += [
            f"  Tokens saved          : {saved_tok:>12,}  ({pct_tok:.0f}% reduction)",
            "",
            "  --- At 1,000 trajectories ---",
            f"  {'Episodiq':42} {epi.total:>12,} tokens",
            f"  {'Naive':42} {naive_at_1k:>12,} tokens",
            f"  {'Savings':42} {100*(naive_at_1k - epi.total)/naive_at_1k:>11.0f}%",
        ]

    output = '\n'.join(lines)
    print()
    print(output)

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as f:
            f.write(output + '\n')
        print(f"\n  Saved to {OUTPUT_FILE}")


asyncio.run(main())
