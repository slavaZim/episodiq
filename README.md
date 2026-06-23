# Episodiq

[![PyPI version](https://img.shields.io/pypi/v/episodiq-ai.svg)](https://pypi.org/project/episodiq-ai/) [![Python versions](https://img.shields.io/pypi/pyversions/episodiq-ai.svg)](https://pypi.org/project/episodiq-ai/) [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) [![CI](https://github.com/slavaZim/episodiq/actions/workflows/ci.yml/badge.svg)](https://github.com/slavaZim/episodiq/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/slavaZim/episodiq/branch/main/graph/badge.svg)](https://codecov.io/gh/slavaZim/episodiq) [![Email](https://img.shields.io/badge/email-skaiscan%40gmail.com-blue.svg)](mailto:skaiscan@gmail.com)

Episodiq is an LLM proxy that captures agent trajectories and represents each one as a sequence of typed action-observation tokens — a sequence- / pattern-based alternative to text-embedding retrieval. The same representation drives:

- **Human-readable logs at ~98% lower annotation cost.** Annotate one representative per cluster, reuse for every trajectory. Zero LLM calls in the hot path. ([details →](#token-efficiency))
- **Structural (event-pattern) retrieval over completed trajectories.** Find paths whose action-observation sequence matches, not whose text matches — useful for monitoring, experience / reflection pipelines, and evaluation. ([benchmark →](#pattern-retrieval-vs-basic-rag))
- **Per-snapshot fail-similarity score.** Running fail-prediction signal from KNN voting on retrieved neighbours.
- **Loop detection.** Surfaces stuck repeating-action patterns — useful even when the agent isn't broken (e.g. token-waste monitoring).
- **Action-predictability signal.** Flags steps where the next action is highly constrained vs steps where the agent could plausibly branch many ways.
- **Drop-in proxy.** OpenAI or Anthropic API, any LLM client / agent framework. Base URL change and additional header.

---

**With Episodiq:**

```bash
episodiq report <trajectory-id> --format pretty -a --steps 47-75 --metric cummean
```

<pre>
Trajectory d6cd3603 <b>[failure]</b>  89 steps, 2.1s  <b>fail_similarity=0.67</b>  variance high=15 low=2  loops=7  unclassified=23
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  47 +0.0s     o:  tool str_replace_editor configuring MySQL dialect syntax and mappings
               -&gt; a:  agent examines specific code ranges in MySQL dialect file
  48 +0.0s     o:  tool str_replace_editor configuring MySQL dialect syntax and mappings  | <b>loop x2</b>
               -&gt; a:  agent examines specific code ranges in MySQL dialect file
  49 +0.0s     o:  tool str_replace_editor configuring MySQL dialect syntax and mappings  | <b>loop x3</b>
               -&gt; a:  agent restores specialized function call after incorrect change
  50 +0.0s     o:  tool str_replace_editor configuring MySQL dialect syntax and mappings  | <b>fail_similarity=0.72</b>
               -&gt; a:  unclassified (action) (str_replace_editor)
   ⋮
  57 +0.1s     o:  tool execute_bash testing struct literal syntax variations  | <b>fail_similarity=0.71</b>
               -&gt; a:  agent debugging parsed SQL structure with print statements
  61 +0.2s     o:  tool execute_bash testing struct literal syntax variations  | <b>fail_similarity=0.71</b>
               -&gt; a:  agent restores specialized function call after incorrect change  | variance=high
  63 +0.2s     o:  tool execute_bash testing struct literal syntax variations  | <b>fail_similarity=0.70</b>
               -&gt; a:  agent restores specialized function call after incorrect change  | variance=high
   ⋮
  66 +0.3s     o:  tool execute_bash testing struct literal syntax variations  | <b>fail_similarity=0.70</b>
               -&gt; a:  agent runs comprehensive test files in workspace directories  | variance=high
  70 +0.4s     o:  unclassified (observation) (execute_bash)  | <b>fail_similarity=0.70</b>  | <b>loop x2</b>
               -&gt; a:  tool execute_bash verifying basic sqlglot import functionality
  73 +0.4s     o:  tool execute_bash displaying code search results for alias handling  | <b>fail_similarity=0.70</b>
               -&gt; a:  agent fixing SQL dialect type mapping compatibility issues
  74 +0.5s     o:  tool think responses successfully logged without errors  | <b>fail_similarity=0.70</b>
               -&gt; a:  agent restores specialized function call after incorrect change
</pre>

> **Zero LLM calls at runtime.**  All signals are computed from paths statistics.
>
> [98% fewer tokens than naive per-message annotation on 275 trajectories, scaling to ~165× (99%) at 1000 →](#token-efficiency)

**Raw:**

<details>
<summary>Raw trajectory — tobymao/sqlglot-3987 · 181 messages · failure · full: <code>benchmarks/demo_eval/raw_d6cd3603.json</code></summary>

```json
[
  {
    "role": "system",
    "content": [
      {
        "text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\n\n<ROLE>\nYour primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\n* If the user asks a question, like \"why is X happening\", don't try to fix the problem. Just give an answer to the question.\n</ROLE>\n\n<EFFICIENCY>\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\n</EFFICIENCY>\n\n<FILE_SYSTEM_GUIDELINES>\n* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\n* NEVER create multiple versions of the same file with different suffixes (e.g., file_test.py, file_fix.py, file_simple.py). Instead:\n  - Always modify the original file directly when making changes\n  - If you need to create a temporary file for testing, delete it once you've confirmed your solution works\n  - If you decide a file you created is no longer useful, delete it instead of creating a new version\n* Do NOT include documentation files explaining your changes in version control unless the user explicitly requests it\n* When reproducing bugs or implementing fixes, use a single file rather than creating multiple files with different versions\n</FILE_SYSTEM_GUIDELINES>\n\n<CODE_QUALITY>\n* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.\n* When implementing solutions, focus on making the minimal changes needed to solve the problem.\n* Before implementing any changes, first thoroughly understand the codebase through exploration.\n* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.\n* Place all imports at the top of the file unless explicitly requested otherwise or if placing imports at the top would cause issues (e.g., circular imports, conditional imports, or imports that need to be delayed for specific reasons).\n</CODE_QUALITY>\n\n<VERSION_CONTROL>\n* If there are existing git user credentials already configured, use them and add Co-authored-by: openhands <openhands@all-hands.dev> to any commits messages you make. if a git config doesn't exist use \"openhands\" as the user.name and \"openhands@all-hands.dev\" as the user.email by default, unless explicitly instructed otherwise.\n* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.\n* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.\n* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.\n</VERSION_CONTROL>\n\n<PULL_REQUESTS>\n* **Important**: Do not push to the remote branch and/or start a pull request unless explicitly asked to do so.\n* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.\n* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.\n* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.\n</PULL_REQUESTS>\n\n<PROBLEM_SOLVING_WORKFLOW>\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions\n2. ANALYSIS: Consider multiple approaches and select the most promising one\n3. TESTING:\n   * For bug fixes: Create tests to verify issues before implementing fixes\n   * For new features: Consider test-driven development when appropriate\n   * Do NOT write tests for documentation changes, README updates, configuration files, or other non-functionality changes\n   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure\n   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies\n4. IMPLEMENTATION:\n   * Make focused, minimal changes to address the problem\n   * Always modify existing files directly rather than creating new versions with different suffixes\n   * If you create temporary files for testing, delete them after confirming your solution works\n5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.\n</PROBLEM_SOLVING_WORKFLOW>\n\n<SECURITY>\n* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.\n* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.\n</SECURITY>\n\n<EXTERNAL_SERVICES>\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\n</EXTERNAL_SERVICES>\n\n<ENVIRONMENT_SETUP>\n* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.\n* If you encounter missing dependencies:\n  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)\n  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)\n  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed\n* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.\n</ENVIRONMENT_SETUP>\n\n<TROUBLESHOOTING>\n* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:\n  1. Step back and reflect on 5-7 different possible sources of the problem\n  2. Assess the likelihood of each possible cause\n  3. Methodically address the most likely causes, starting with the highest probability\n  4. Document your reasoning process\n* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.\n</TROUBLESHOOTING>\n\n<DOCUMENTATION>\n* When explaining changes or solutions to the user:\n  - Include explanations in your conversation responses rather than creating separate documentation files\n  - If you need to create documentation files for reference, do NOT include them in version control unless explicitly requested\n  - Never create multiple versions of documentation files with different suffixes\n* If the user asks for documentation:\n  - Confirm whether they want it as a separate file or just in the conversation\n  - Ask if they want documentation files to be included in version control\n</DOCUMENTATION>\n\n<PROCESS_MANAGEMENT>\n* When terminating processes:\n  - Do NOT use general keywords with commands like `pkill -f server` or `pkill -f python` as this might accidentally kill other important servers or processes\n  - Always use specific keywords that uniquely identify the target process\n  - Prefer using `ps aux` to find the exact process ID (PID) first, then kill that specific PID\n  - When possible, use more targeted approaches like finding the PID from a pidfile or using application-specific shutdown commands\n</PROCESS_MANAGEMENT>\n\n<TASK_MANAGEMENT>\n* You have access to the `task_tracker` tool to help you organize and monitor development work. Use this tool REGULARLY to maintain task visibility and provide users with clear progress updates. This tool is ESSENTIAL for systematic planning and decomposing complex development work into manageable components. Failing to use this tool for planning may result in overlooked requirements - which is unacceptable.\n* It is crucial that you update task status to \"done\" immediately upon completion of each work item. Do not accumulate multiple finished tasks before updating their status.\n* For complex, multi-phase development work, use `task_tracker` to establish a comprehensive plan with well-defined steps:\n  1. Begin by decomposing the overall objective into primary phases using `task_tracker`\n  2. Include detailed work items as necessary to break complex activities into actionable units\n  3. Update tasks to \"in_progress\" status when commencing work on them\n  4. Update tasks to \"done\" status immediately after completing each item\n  5. For each primary phase, incorporate additional work items as you identify new requirements\n  6. If you determine the plan requires substantial modifications, suggest revisions and obtain user confirmation before proceeding\n* Example workflow for debugging and resolution:\n  ```\n  User: \"Execute the test suite and resolve any validation failures\"\n  Assistant: I'm going to use the task_tracker tool to organize the following work items:\n  - Execute the test suite\n  - Resolve any validation failures\n  I'm now going to run the test suite using the terminal.\n  [After running tests and discovering 8 validation failures]\n  I found 8 validation failures that need attention. I'm going to use the task_tracker tool to add 8 specific items to the task list.\n  [Updating first task to in_progress]\n  Let me begin addressing the first validation issue...\n  [After resolving first failure]\n  The first validation issue has been resolved, let me mark that task as done and proceed to the second item...\n  ```\n* Example workflow for component development:\n  ```\n  User: \"Build a dashboard component that displays analytics data with interactive charts and filtering options\"\n  Assistant: I'll help you create an analytics dashboard with interactive charts and filtering. Let me first use the task_tracker tool to organize this development work.\n  Adding the following tasks to the tracker:\n  1. Analyze existing analytics data structure and requirements\n  2. Design dashboard layout and component architecture\n  3. Implement data visualization charts with interactivity\n  4. Create filtering and search functionality\n  5. Integrate components and perform testing\n  Let me start by examining the current analytics data structure to understand what we're working with...\n  [Assistant proceeds with implementation step by step, updating tasks to in_progress and done as work progresses]\n  ```\n</TASK_MANAGEMENT>",
        "type": "text"
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "text": "<uploaded_files>\n/workspace/tobymao__sqlglot__25.17\n</uploaded_files>\n\nI've uploaded a python code repository in the directory tobymao__sqlglot__25.17. Consider the following issue description:\n\n<issue_description>\nMissing Support for JSON Path Syntax in MySQL Dialect\n### Description:\r\nSQLGlot MySQL dialect does not support JSON path syntax functions like JSON_VALUE, resulting in a parse failure when attempting to convert a MySQL query to an Abstract Syntax Tree (AST).\r\n\r\n### Details:\r\n\r\nMySQL JSON Path Syntax Functions\r\n\r\nMySQL supports JSON path syntax functions, such as JSON_VALUE, which allow extracting values from JSON documents.\r\nJSON_VALUE Function Syntax\r\nThe specific functions and syntax that are not supported include:\r\n\r\n**`JSON_VALUE()` Function:**\r\n\r\n- **Syntax:**\r\n  ```sql\r\n  JSON_VALUE(json_doc, path [RETURNING type] [on_empty] [on_error])\r\n\r\n```\r\nUnsupported Syntax Elements\r\n\r\nThe specific syntax elements not supported by SQLGlot's MySQL dialect include:\r\n\r\n-     RETURNING type\r\n-     on_empty: {NULL | ERROR | DEFAULT value} ON EMPTY\r\n-     on_error: {NULL | ERROR | DEFAULT value} ON ERROR\r\n```\r\n**CODE**\r\n````from sqlglot.dialects.mysql import MySQL\r\nfrom sqlglot import parse_one\r\n\r\nsql = \"\"\"\r\nSELECT SELECT JSON_VALUE('{\"item\": \"shoes\", \"price\": \"49.95\"}', '$.price'\r\n     RETURNING DECIMAL(4,2)) AS price;\r\n\"\"\"\r\n\r\nast = parse_one(sql, dialect=MySQL)\r\nprint(repr(ast))\r\n````\r\n\r\n\r\n**ERROR** \r\n\r\n**sqlglot.errors.ParseError: Invalid expression / Unexpected token. Line 2, Col: 13.**\r\n\r\n\r\n [   link](https://dev.mysql.com/doc/refman/8.4/en/json-search-functions.html) for the documetation \r\n    \r\n    \r\n    \n</issue_description>\n\nCan you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\nI've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\nAlso the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\nYour task is to make the minimal changes to non-test files in the /workspace/tobymao__sqlglot__25.17 directory to ensure the <issue_description> is satisfied.\n\nFollow these phases to resolve the issue:\n\nPhase 1. READING: read the problem and reword it in clearer terms\n   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.\n   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.\n   1.3 Explain the problem in clear terms.\n   1.4 Enumerate the steps to reproduce the problem.\n   1.5 Hightlight any best practices to take into account when testing and fixing the issue\n\nPhase 2. RUNNING: install and run the tests on the repository\n   2.1 Follow the readme\n   2.2 Install the environment and anything needed\n   2.2 Iterate and figure out how to run the tests\n\nPhase 3. EXPLORATION: find the files that are related to the problem and possible solutions\n   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.\n   3.2 Identify all files related to the problem statement.\n   3.3 Propose the methods and files to fix the issue and explain why.\n   3.4 From the possible file locations, select the most likely location to fix the issue.\n\nPhase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.\n   4.1 Look at existing test files in the repository to understand the test format/structure.\n   4.2 Create a minimal reproduction script that reproduces the located issue.\n   4.3 Run the reproduction script to confirm you are reproducing the issue.\n   4.4 Adjust the reproduction script as necessary.\n\nPhase 5. FIX ANALYSIS: state clearly the problem and how to fix it\n   5.1 State clearly what the problem is.\n   5.2 State clearly where the problem is located.\n   5.3 State clearly how the test reproduces the issue.\n   5.4 State clearly the best practices to take into account in the fix.\n   5.5 State clearly how to fix the problem.\n\nPhase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.\n   6.1 Make minimal, focused changes to fix the issue.\n\nPhase 7. VERIFICATION: Test your implementation thoroughly.\n   7.1 Run your reproduction script to verify the fix works.\n   7.2 Add edge cases to your test script to ensure comprehensive coverage.\n   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.\n\n8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit 2f3626a4fc20c46411cd91bf8beda2bdd103ca4a.\n   8.1 Ensure you've fully addressed all requirements.\n   8.2 Run any tests in the repository related to:\n     8.2.1 The issue you are fixing\n     8.2.2 The files you modified\n     8.2.3 The functions you changed\n   8.3 If any tests fail, revise your implementation until all tests pass\n\nBe thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.",
        "type": "text"
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "text": "I'll help you implement the necessary changes to support JSON path syntax in MySQL dialect. Let me start by understanding the problem and working through each phase systematically.\n\n## Phase 1. READING: Understanding the Problem\n\n",
        "type": "text"
      },
      {
        "id": "chatcmpl-tool-b2554c365e5d4379bf4b9039c50851be",
        "type": "tool_call",
        "input": {
          "thought": "Let me analyze the issue description carefully:\n\n1. The problem is that SQLGlot's MySQL dialect doesn't support JSON path syntax functions like JSON_VALUE\n2. Specifically, the following syntax elements are not supported:\n   - RETURNING type clause\n   - on_empty clause: {NULL | ERROR | DEFAULT value} ON EMPTY  \n   - on_error clause: {NULL | ERROR | DEFAULT value} ON ERROR\n3. The example shows JSON_VALUE function with RETURNING DECIMAL(4,2)\n4. Error occurs when trying to parse: ParseError: Invalid expression / Unexpected token. Line 2, Col: 13.\n\nKey technical details:\n- Function name: JSON_VALUE\n- Method involved: parse_one() \n- Dialect: MySQL\n- Error type: sqlglot.errors.ParseError\n- Location: Line 2, Col: 13\n\nThe problem is clear: the MySQL parser doesn't recognize the extended JSON_VALUE syntax with RETURNING, ON EMPTY, and ON ERROR clauses.\n\nSteps to reproduce:\n1. Import MySQL dialect from sqlglot.dialects.mysql\n2. Use parse_one() with a SQL query containing JSON_VALUE with RETURNING clause\n3. Parser fails with ParseError\n\nBest practices for fixing:\n- Need to extend the MySQL dialect parser to handle these additional clauses\n- Should maintain backward compatibility\n- Need to understand how other similar functions are implemented\n- Should follow existing patterns in the codebase"
        },
        "tool_name": "think"
      }
    ]
  },
  {
    "role": "tool",
    "content": [
      {
        "id": "chatcmpl-tool-b2554c365e5d4379bf4b9039c50851be",
        "type": "tool_response",
        "tool_name": "think",
        "tool_response": "Your thought has been logged."
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "text": "### 1.1-1.3 Problem Analysis\n\nThe issue is that **SQLGlot's MySQL dialect parser does not support the full JSON_VALUE function syntax**, specifically:\n- **Method name**: `JSON_VALUE()`\n- **Missing syntax elements**: `RETURNING type`, `ON EMPTY` clause, `ON ERROR` clause\n- **Error message**: `sqlglot.errors.ParseError: Invalid expression / Unexpected token. Line 2, Col: 13.`\n- **File/method involved**: `parse_one()` with `MySQL` dialect\n\n### 1.4 Steps to Reproduce\n1. Import `MySQL` dialect from `sqlglot.dialects.mysql`\n2. Call `parse_one()` with SQL containing `JSON_VALUE(... RETURNING ...)` syntax\n3. Parser throws `ParseError` at the `RETURNING` keyword\n\n### 1.5 Best Practices\n- Extend existing parser grammar rather than rewriting\n- Follow established patterns for function parsing in SQLGlot\n- Maintain backward compatibility with existing JSON functions\n- Test edge cases with various RETURNING types and ON clauses\n\n## Phase 2. RUNNING: Environment Setup and Tests\n\n",
        "type": "text"
      },
      {
        "id": "chatcmpl-tool-e69d876d6e064b32a1d8ad4a1c921754",
        "type": "tool_call",
        "input": {
          "path": "/workspace",
          "command": "view"
        },
        "tool_name": "str_replace_editor"
      }
    ]
  },
  // ... 176 more messages elided — full raw: benchmarks/demo_eval/raw_d6cd3603.json
]
```

</details>

---

## Installation

```bash
pip install episodiq-ai
```

## Development

```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync
source .venv/bin/activate  # then use episodiq directly
# or without activating: uv run episodiq --help
```

## Quick start

**Requirements:** PostgreSQL 17+ with [pgvector](https://github.com/pgvector/pgvector)

```bash
pip install episodiq-ai
episodiq db init    # requires EPISODIQ_DATABASE_URL pointing to your Postgres instance
episodiq up
```

<details>
<summary>Don't have Postgres + pgvector? One-liner:</summary>

```bash
docker run -d --name episodiq-pg \
  -e POSTGRES_PASSWORD=episodiq \
  -p 5432:5432 \
  pgvector/pgvector:pg17

export EPISODIQ_DATABASE_URL=postgresql://postgres:episodiq@localhost:5432/postgres
```

</details>

**OpenAI**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/openai/v1", api_key="...")
```

**Anthropic**

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080/anthropic/v1",
    api_key="...",
)
```

**LangChain (ReAct)**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/openai/v1",
    api_key="...",
)
```

**Trajectory ID**

Each agent run is a trajectory. Pass `X-Trajectory-ID` with a UUID you generate, or omit it and read the ID back from the response header — the proxy creates one and echoes it.

```python
import uuid, httpx

# Option A: you own the ID
trajectory_id = str(uuid.uuid4())
client = OpenAI(
    base_url="http://localhost:8080/openai/v1",
    api_key="...",
    default_headers={"X-Trajectory-ID": trajectory_id},
)

# Option B: proxy generates it — read from first response
response = client.chat.completions.with_raw_response.create(...)
trajectory_id = response.headers.get("x-trajectory-id")
```

---

## How it works

```
Your agent
    │
    │  unchanged API calls
    ▼
┌──────────────────────────────────────────────────────────┐
│                   Episodiq proxy  :8080                  │
│  /openai/v1                                              ├──────► OpenAI / Anthropic / OpenRouter
│  /anthropic/v1                                           │
└───────────┬───────────────────────────────┬──────────────┘
            │                               │
            │ trajectory persistence        │ fire-and-forget
            │ sync, timeout-guarded         │ embedding → online clustering → path update
            │ (on error → forward only)     │
            ▼                               │
     PostgreSQL + pgvector ◄────────────────┘
            │
            ▼
   offline clustering · annotating · analytics
```

**Why a proxy.** One base URL change is all your agent code needs.
Everything else — database management, embedding, clustering, and
analytics pipelines — is handled by Episodiq out-of-band. The proxy
is **failsafe**: if any step errors or times out, the request still
forwards to the upstream provider and your agent never sees a
difference.

**The representation.** Each user/tool message is an **observation**, each assistant message is an **action**, and each agent step is an (action, observation) pair. Both sides are embedded and clustered, then the pairs themselves are re-clustered into a compact **act_obs token vocabulary** (~50–60 tokens). A trajectory is a sequence of these typed tokens.

**The retrieval cascade.** Anchors are placed at evenly spaced center steps along the trajectory; each anchor's surrounding window is hashed into per-window MinHash LSH bands. Lookup runs in four stages, each shrinking the candidate pool: (1) LSH band collisions per anchor → wide candidate set; (2) exact Jaccard rerank against each anchor's signature; (3) cross-anchor aggregation (mean or min-distance) into a single per-candidate score; (4) `agg-shift` Levenshtein rerank with a shift penalty, producing the final top-K. That structural similarity powers fail-prediction, loop detection, and action-variance signals, all computed without runtime LLM calls.

---

## Benchmarks

Evaluated on coding-agent trajectories from the [SWE-rebench-OpenHands-Trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) dataset, filtered to the [`tobymao/sqlglot`](https://github.com/tobymao/sqlglot) repository: **275 trajectories** (178 failures, 97 successes), no two sharing the same PR. A stratified split — `Random(42).shuffle` over instance-id-sorted trajectories, then proportional interleave by status — yields a 100-trajectory tune slice and a 175-trajectory held-out eval slice, both at ~65% failure rate (population mean). Reported 95% confidence intervals are per-trajectory bootstrap (200 draws). Full methodology and tuning guidance for your own agent: [`benchmarks/demo_eval/README.md`](benchmarks/demo_eval/README.md).

### Token efficiency

At 1 000 trajectories, Episodiq annotation uses **~165× fewer tokens** than naive per-message annotation (918k vs 151M tokens, **99% reduction**). On 275 sqlglot trajectories specifically: 918k vs 41.5M = **98% reduction**.

Methodology: cluster all messages, contrastively annotate the centroids once via [`episodiq annotate`](episodiq/cli/annotate); naive baseline samples 30 messages, runs them individually through the same summarize→annotate path, and extrapolates linearly to the full corpus. Both sides include a summarizer step (Haiku) for long inputs; on sqlglot most messages were short enough that Episodiq's summarizer didn't trigger. See [`benchmarks/demo_eval/token_efficiency.py`](benchmarks/demo_eval/token_efficiency.py).

### Pattern retrieval vs Basic RAG

`cummean` (running-mean of per-snapshot fail-similarity) is the headline metric for both systems. Of the three aggregations Episodiq computes (`cummax`, `cummean`, `cummeanmax`), `cummean` is the most robust to noise — averaging dilutes single lucky hits, so a high `cummean` requires a *sustained* similarity signal. With only 275 trajectories split into a 100-traj tune slice and a 175-traj eval slice, single-spike metrics turn noisy, so locking the headline to the robust aggregation is the honest call.

**Headline**:

| Method | tune AUC | eval AUC@s50 (cummean) | 95% CI |
|---|---|---|---|
| **Cascade retrieval** (per-window MinHash LSH + agg-shift Lev rerank) | 0.6440 | **0.6983** | [0.609, 0.782] |
| Basic RAG (qwen3-embedding-8b cosine) | 0.5526 | 0.4820 | [0.372, 0.577] |

For reference — Basic RAG's strongest shot, cherry-picking its own best config per metric:

| Metric | Basic eval | 95% CI |
|---|---|---|
| cummax | 0.5764 | [0.461, 0.670] |
| cummean | 0.4820 | [0.372, 0.577] |
| cummeanmax | 0.4565 | [0.345, 0.561] |

Every Basic RAG CI overlaps 0.5 — text-embedding cosine over qwen3 has no usable structural-prediction signal on this corpus, even when allowed to pick the best aggregation. Cascade's `cummean` eval CI lower bound is 0.609 (clearly above random), and eval > tune indicates generalisation, not overfit.

**Methodology**:

- **`AUC@s50`** is a time-stratified, weighted ROC AUC that controls for trajectory length. For every step `S ≥ 50`: take the trajectories still alive at `S` (last snapshot ≥ `S`); each contributes ONE score — its cumulative aggregate (`cummax` / `cummean` / `cummeanmax`) over snapshots up to step `S`. The ROC AUC at step `S` is then computed across those trajectories at the same horizon — a long trajectory doesn't inflate its score by having more snapshots to draw from, because at step `S` it's compared only against other alive trajectories using their `[0..S]` slice. Step AUCs are averaged across `S` weighted by `n_active(S)` so the early steps (where most trajectories are still alive) carry the most weight.
- Both sides tune on the same 100-traj slice and evaluate on the same 175-traj held-out slice. Split is `Random(42).shuffle` over instance-id-sorted trajectories, then proportional interleave by status — both slices land at ~65% failure rate (population mean). Eval queries see the full corpus (the split lives in hyperparameter selection, not the data).
- Cascade champion: `W=10  agg=min_distance  prefetch_n_uniq=220  jaccard_n_uniq=140  top_k=10  penalty_shape=lin  lam=1.2  sigma=0.5  gap_open=2.8  gap_extend=1.2` (LSH layout B=64, R=1; see step 7 in the [setup guide](#setup-guide) for why bench overrides the prod default).

Results are still **preliminary** — single-repo eval, wide bootstrap CIs. Will be confirmed on larger corpora.

Reproduce: [`benchmarks/demo_eval/`](benchmarks/demo_eval/) (steps 01–11) for the cascade pipeline, [`benchmarks/basic.py`](benchmarks/basic.py) for the Basic RAG baseline. Both consume the same `--shuffle-seed 42 --tune-limit 100 --stratify` split.

---

## Important notes

- Episodiq works best for a single-domain agent operating over a pool of similar or related tasks — the token vocabulary and retrieval signals only become meaningful once the agent has revisited comparable situations often enough for recurring patterns to surface. Current tested setup is a coding agent operating on a single medium-size repo. Each distinct agent should live in its own database.
- Multi-turn seeding is not supported — injecting a pre-existing conversation history into the first request will not be tracked correctly.

---

## Setup guide

**1. Start the proxy and route your agent through Episodiq**

```bash
episodiq up
```

Point your agent at `http://localhost:8080/openai/v1` (or `/anthropic/v1` — both providers are supported) and pass `X-Trajectory-ID` per run. The proxy captures every turn transparently.

**2. Accumulate trajectories**

Run your agent on a representative pool of tasks. Clustering needs enough data to form meaningful groups — a few hundred trajectories is a practical minimum.

Label each trajectory outcome so analytics can distinguish successes from failures:

```http
PATCH /episodiq/trajectories/{trajectory_id}
{"status": "success" | "failure"}
```

**3. Cluster messages**

```bash
episodiq cluster grid-search   # finds best UMAP+HDBSCAN params per (type, category) — saved as CSV
episodiq cluster run           # apply chosen params to all messages
```

Picking clustering params from the grid CSV: filter `noise_ratio ≤ 0.25`, then pick the row with the highest `dbcv` and the smallest reasonable `n_clusters`. **Some noise is healthy — 10–20% means HDBSCAN is being honest about borderline points**; 0% with very few clusters means UMAP collapsed the manifold and you've lost granularity; >30% means params are too strict. Tune each (type, category) pair independently — `execute_bash` observations have a different shape than `think` actions. Worked example: [`benchmarks/demo_eval/README.md#picking-clustering-parameters`](benchmarks/demo_eval/README.md#picking-clustering-parameters).

**4. Annotate clusters**

```bash
episodiq annotate run
```

Short human-readable labels per cluster (contrastive annotation). The annotator/summarizer use the same OpenAI- or Anthropic-compatible endpoint as the proxy — set **both** the base URL and the API key: `EPISODIQ_OPENAI_BASE_URL` + `EPISODIQ_OPENAI_API_KEY`, or `EPISODIQ_ANTHROPIC_BASE_URL` + `EPISODIQ_ANTHROPIC_API_KEY`.

> Note: annotations drive a label-aware merge inside the AO tokenizer; on the demo corpus this gives a noticeable eval-AUC uplift (~3–4 points) over running the tokenizer on raw clustering.

**5. Build paths**

```bash
episodiq cluster build-paths
```

Reconstructs trajectory paths from cluster labels (one path per agent step).

**6. Tokenize act_obs pairs**

```bash
episodiq cluster tokenize-grid   # grid-search the AO tokenizer (no DB writes)
episodiq cluster tokenize        # apply chosen params; same picking heuristic as step 3, aim for ~50–60 tokens
```

Clusters (action, observation) pair embeddings into a compact AO token vocabulary — the alphabet over which the cascade retrieval indexes per-window MinHash LSH bands.

**7. Build the MinHash retrieval index**

```bash
episodiq index build
```

Materializes `trace_tokens` + per-window MinHash LSH bands per `trajectory_path` from the AO token mapping. Required before `tune` and the report's pattern-retrieval signal.

LSH layout — `EPISODIQ_WMH_SIG_SIZE` × `EPISODIQ_WMH_NUM_BANDS` defines `(rows_per_band, num_bands)` via `sig_size / num_bands`:

- **Prod default** `64 / 32` (B=32, R=2; candidate-Jaccard threshold ≈ 0.18) — narrower prefilter, less recall noise.
- **Bench used** `64 / 64` (B=64, R=1; threshold ≈ 0.015) — much looser; on the 275-traj demo the corpus is too small for the narrower band layout to surface enough candidates, so the wider one wins eval AUC.

Other knobs: `EPISODIQ_RETRIEVAL_WINDOW` (W = 2w token window, default 10), `EPISODIQ_WMH_QGRAM` (default 2).

**8. Tune the retrieval cascade**

```bash
episodiq tune retrieval-sweep \
  --shuffle-seed 42 --stratify-field status --tune-limit 100 \
  --w-grid 10,14 --agg-grid mean,min_distance \
  --objective-metric cummean --save-trials trials.csv
```

`retrieval-sweep` writes every trial to the CSV; copy the picked row's params into your runtime environment (`EPISODIQ_RETRIEVAL_WINDOW`, `EPISODIQ_CASCADE_*`, `EPISODIQ_AS_*`). `--stratify-field status` keeps the tune/eval split's fail-rate at the population mean regardless of seed.

`--objective-metric` controls what TPE optimises:

- **Single metric** (e.g. `cummean`) — locks the sampler to one objective. Recommended on small corpora where multi-metric sampling lets the picker chase noise (see the [benchmarks section](#pattern-retrieval-vs-basic-rag)).
- **Omit the flag** — TPE samples `metric` as a categorical knob per trial; the picker reports the best `(W, agg, metric)` combo overall. Recommended on larger corpora where sharing samples across metrics accelerates convergence and the picker noise is small.

**9. Tune the path-frequency thresholds (optional)**

```bash
episodiq tune path-freq          # → EPISODIQ_LOW_ENTROPY / EPISODIQ_HIGH_ENTROPY
```

Samples completed paths, runs the cascade retrieval on each, and folds the resulting **vote-entropy** distribution into two percentile thresholds (default 10th / 90th). These thresholds power the report's *action-variance* bucket — paths with low entropy collapse the next action onto a single dominant candidate ("confident consensus"), high entropy means the agent could plausibly branch many ways. Optional because the retrieval signal stands on its own; pull this in when you want to surface action-predictability alongside fail-similarity. Set the suggested values via `EPISODIQ_LOW_ENTROPY` / `EPISODIQ_HIGH_ENTROPY` in your runtime environment so reports pick them up.

---

## Environment variables

See [`.env.example`](.env.example).

---

## Getting help

- Bug or feature request → [GitHub Issues](../../issues)
- Usage question → [GitHub Discussions](../../discussions)
- Feedback, help → ping me on [Telegram](https://t.me/wdambb) or email [skaiscan@gmail.com](mailto:skaiscan@gmail.com)
