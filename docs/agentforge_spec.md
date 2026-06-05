# AgentForge MVP — Project Specification

> Handoff document for code generation. Read this in full before writing any code. Constraints in sections 4, 14, and 17 are load-bearing — don't relax them without explicit approval.

---

## 1. One-line summary

A system that learns which of three fixed agentic workflows to deploy for a given text-to-SQL task, using an ε-greedy multi-armed bandit over binary execution-correctness reward, demonstrated on the BIRD-SQL benchmark.

## 2. Project context

**Course:** Stanford CS 153 (Frontier Systems), Spring 2026. Solo project. Already-approved proposal (full marks).

**Key milestones (today is May 11, 2026):**
- May 15, 2026 — Week 7 progress check (5% of grade)
- May 29, 2026 — Final video + GitHub submission
- June 4, 2026 — Peer reviews

**TA framing (locked in):** focus on one domain, demonstrate a clean MVP with fixed workflows + reward signals, show the system learning to pick better workflows over time before expanding.

**Broader vision (NOT part of MVP):** workflows as learnable objects that can be discovered/improved via process supervision from expert demonstrations. The MVP is a proof-of-concept for this, not the full system.

## 3. The problem being solved

Agents built on LLMs are stochastic: the same prompt yields different trajectories across runs, and different task instances benefit from different workflow strategies. Today there's no principled way to select *which workflow* to run for a given task — practitioners hand-tune one workflow per use case. AgentForge treats workflow selection as a bandit problem: given a fixed library of workflows, learn online which workflow tends to succeed on which kind of task.

**Why this is bandit-shaped, not RL-shaped:** each task is one independent decision (pick a workflow, get a binary reward). No state transitions, no credit assignment over time. RL framing would only become relevant if learning workflow internals later.

## 4. Locked-in technical choices (do not relax)

| Choice | Value | Rationale |
|---|---|---|
| Domain | BIRD-SQL (LiveCodeBench backup) | Automatic binary correctness signal; workflow strategies map cleanly to SQL |
| Number of workflows (arms) | 3 | Direct / Schema-explore / Decompose — meaningfully differentiated |
| Bandit algorithm | ε-greedy, ε=0.1 | Simplest convergent algorithm; defensible default |
| Episode count | 50–100 per run | Enough to show learning curve, small enough to fit budget |
| Random seeds | 3–5 | Average curves across seeds for variance |
| LLM temperature | 0 | Reduce stochasticity to isolate workflow effects |
| Stack | Plain Python + Anthropic OR OpenAI API + SQLite + JSONL + matplotlib | No framework dependencies |
| Frameworks | NONE for MVP | No LangGraph, no DSPy, no LangChain, no LangSmith |

## 5. Architecture

```
                ┌─────────────────────────────────────────────────┐
                │                  Episode Loop                    │
                │                                                  │
   BIRD task    │                                                  │
   ──────────► ┌──────────┐    ┌─────────┐                        │
               │  Bandit  ├───►│Workflow │                         │
               │ (ε-grdy) │    │ Library │                         │
               └────▲─────┘    └────┬────┘                         │
                    │               │                              │
                    │               ▼                              │
                    │         ┌───────────┐    ┌───────────┐      │
                    │         │ LLM call(s)├──►│Generated  │      │
                    │         │ (+ schema  │   │   SQL     │      │
                    │         │  inspect)  │   └─────┬─────┘      │
                    │         └───────────┘         │              │
                    │                               ▼              │
                    │                       ┌──────────────┐       │
                    │                       │   Executor   │       │
                    │                       │   (SQLite)   │       │
                    │                       └──────┬───────┘       │
                    │                              │               │
                    │  reward update         ┌─────▼─────┐         │
                    └───────────────────────►│  Reward   │         │
                                             │ (binary)  │         │
                                             └─────┬─────┘         │
                                                   │               │
                                                   ▼               │
                                             ┌───────────┐         │
                                             │  Logger   │         │
                                             │  (JSONL)  │         │
                                             └───────────┘         │
                └─────────────────────────────────────────────────┘
                                       │
                                       ▼
                                 ┌──────────┐
                                 │ Evaluator│  matplotlib
                                 │  / Plots │  cumulative reward, arm freq, regret
                                 └──────────┘
```

**Data flow per episode:**
1. Sample task from BIRD dev subset (IID, without replacement within a run)
2. Bandit selects arm: ε-greedy over running mean rewards
3. Selected workflow generates SQL via 1+ LLM calls (and possibly schema inspection tool calls)
4. Executor runs generated SQL on task's SQLite DB
5. Executor runs gold SQL on same DB
6. Reward = 1 if result sets match (order-insensitive), 0 otherwise (including on errors/timeouts)
7. Bandit updates running mean for selected arm
8. Logger writes JSONL line with full episode trace

## 6. Data: BIRD-SQL

**Source:** https://bird-bench.github.io/

**What it is:** 12,751 question–SQL pairs across 95 databases (33.4 GB total) spanning 37+ professional domains. Each task contains a natural-language question, a SQLite database, and a gold SQL query.

**MVP-relevant subset:** start with the BIRD **dev set** (~1,534 examples across 11 databases). For the first runs, use a 5–10 database subset to keep download/runtime manageable. Full dev set later.

**Task data model:**
```python
@dataclass
class Task:
    task_id: str            # BIRD's "question_id"
    db_id: str              # which database
    db_path: str            # path to .sqlite file
    question: str           # natural language question
    evidence: str           # BIRD's "external knowledge" hint
    gold_sql: str           # ground truth SQL
    difficulty: str         # BIRD's labeled difficulty (easy/medium/challenging)
```

**Known caveat:** BIRD's gold SQL has documented noise (~some "correct" answers are actually wrong, per Wretblad et al. 2024). This adds variance to the reward signal. Acceptable for MVP; flag in writeup.

## 7. The three workflow templates

All three workflows take `(question, db_schema, evidence)` and return a SQL string. They differ in *strategy*.

### 7.1 Direct
Single LLM call. Prompt includes full schema, question, evidence. Asks for SQL directly.

**Pseudocode:**
```python
def workflow_direct(task: Task, llm) -> str:
    schema = load_schema(task.db_path)
    prompt = DIRECT_PROMPT.format(
        schema=schema,
        question=task.question,
        evidence=task.evidence,
    )
    sql = llm.complete(prompt, temperature=0)
    return extract_sql(sql)
```

**Prompt template (DIRECT_PROMPT):**
```
You are an expert SQL writer. Given the following SQLite database schema and question, write a single SQL query that answers the question.

Schema:
{schema}

External knowledge: {evidence}

Question: {question}

Return only the SQL query, no explanation. Wrap it in ```sql ... ``` fences.
```

### 7.2 Schema-Explore-Then-Write
Two-step. First LLM call identifies relevant tables/columns from the full schema. Second LLM call writes SQL using only the relevant subset.

**Pseudocode:**
```python
def workflow_explore(task: Task, llm) -> str:
    schema = load_schema(task.db_path)
    relevant = llm.complete(
        EXPLORE_PROMPT.format(schema=schema, question=task.question, evidence=task.evidence),
        temperature=0,
    )
    relevant_schema = filter_schema(schema, parse_relevant_tables(relevant))
    sql = llm.complete(
        WRITE_PROMPT.format(schema=relevant_schema, question=task.question, evidence=task.evidence),
        temperature=0,
    )
    return extract_sql(sql)
```

**Hypothesis:** wins on tasks with large/messy schemas where the model is distracted by irrelevant tables.

### 7.3 Decompose-Then-Compose
Three-step. First LLM call decomposes the question into sub-questions. Second LLM call writes sub-SQL for each. Third LLM call composes them into a final SQL query.

**Pseudocode:**
```python
def workflow_decompose(task: Task, llm) -> str:
    schema = load_schema(task.db_path)
    sub_questions = llm.complete(
        DECOMPOSE_PROMPT.format(question=task.question, evidence=task.evidence),
        temperature=0,
    )
    sub_sqls = [
        llm.complete(SUB_SQL_PROMPT.format(schema=schema, sub_q=sq), temperature=0)
        for sq in parse_sub_questions(sub_questions)
    ]
    final_sql = llm.complete(
        COMPOSE_PROMPT.format(
            schema=schema,
            question=task.question,
            sub_sqls="\n\n".join(sub_sqls),
        ),
        temperature=0,
    )
    return extract_sql(final_sql)
```

**Hypothesis:** wins on multi-join, multi-aggregate, nested queries where one-shot generation fails.

### Shared workflow interface
```python
class Workflow(ABC):
    name: str

    @abstractmethod
    def run(self, task: Task, llm: LLMClient) -> WorkflowResult:
        ...

@dataclass
class WorkflowResult:
    sql: str
    llm_calls: int
    tokens_used: int
    latency_ms: int
    intermediate_outputs: dict  # for debugging
```

## 8. Bandit algorithm — ε-greedy

```python
class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.means = [0.0] * n_arms
        self.counts = [0] * n_arms
        self.rng = random.Random(seed)

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_arms)
        # break ties uniformly at random
        max_val = max(self.means)
        candidates = [i for i, m in enumerate(self.means) if m == max_val]
        return self.rng.choice(candidates)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n

    def state(self) -> dict:
        return {"means": list(self.means), "counts": list(self.counts)}
```

**Notes:**
- Tie-breaking is uniformly random among tied arms (important early when all arms have mean 0).
- Optional but recommended: initialize each arm with one forced pull before bandit takes over (so means aren't all 0 at start). Document whichever choice is made.

## 9. Reward signal

Binary execution match. Result sets compared order-insensitively.

```python
def compute_reward(generated_sql: str, gold_sql: str, db_path: str, timeout_s: int = 30) -> tuple[int, str | None]:
    """Returns (reward, error_message_or_None)."""
    try:
        gen_rows = execute_sql(generated_sql, db_path, timeout_s)
    except Exception as e:
        return 0, f"generated_exec_error: {e}"
    try:
        gold_rows = execute_sql(gold_sql, db_path, timeout_s)
    except Exception as e:
        return 0, f"gold_exec_error: {e}"
    if set(map(tuple, gen_rows)) == set(map(tuple, gold_rows)):
        return 1, None
    return 0, None
```

**Edge cases to handle:**
- SQL syntax errors → reward 0, log error
- Timeouts → reward 0, log error
- Empty result sets matching → reward 1 (valid case)
- NULL handling in tuples → cast everything to string before set comparison
- Floating point: BIRD typically returns ints/strings; if floats appear, round to 4 decimals before comparison

## 10. Logging — JSONL schema

One line per episode. Append-only.

```json
{
  "run_id": "exp_2026_05_12_seed0",
  "episode_idx": 17,
  "timestamp": "2026-05-12T14:32:11Z",
  "seed": 0,
  "task_id": "bird_dev_493",
  "db_id": "california_schools",
  "question": "List all schools in...",
  "difficulty": "medium",
  "selected_arm": 1,
  "workflow_name": "schema_explore",
  "generated_sql": "SELECT ...",
  "gold_sql": "SELECT ...",
  "reward": 1,
  "execution_error": null,
  "latency_ms": 4821,
  "llm_calls": 2,
  "tokens_used": 3104,
  "arm_means_after": [0.41, 0.52, 0.33],
  "arm_counts_after": [6, 8, 3],
  "epsilon": 0.1,
  "model": "claude-sonnet-4-6"
}
```

Separate file per run: `logs/{run_id}.jsonl`.

## 11. Experimental design

### 11.1 Pre-flight smoke test (BEFORE running the bandit)

Run each workflow on a fixed set of 15–20 BIRD dev tasks, temperature 0, deterministic. Compute per-workflow success rate.

**Pass criterion:** at least 5–10 percentage points spread between best and worst workflow, AND no two workflows are within 2 points of each other on the same task subset. If workflows are too similar in success profile, the bandit has nothing to learn — redesign workflows before proceeding.

### 11.2 Main runs

For each of N=3–5 seeds:
1. Initialize bandit with seed
2. Shuffle BIRD dev subset deterministically with seed
3. Run 50–100 episodes (one task each, without replacement)
4. Write JSONL log

### 11.3 Baselines (same task ordering per seed for fair comparison)

- **Random:** uniform-random arm selection (no learning)
- **Single-arm:** always-pick-direct, always-pick-explore, always-pick-decompose (three baselines)
- **Oracle:** post-hoc, for each task, pick the workflow that *would have* succeeded. Compute by running all three workflows offline on the test tasks. Upper bound the bandit cannot beat.

## 12. Evaluation & plots

Produce four plots (matplotlib, saved to `plots/`):

1. **Cumulative reward over episodes** — line plot, x=episode, y=cumulative reward. One line per condition (bandit, random, each single-arm, oracle). For bandit and random, show mean ± std across seeds as shaded band.

2. **Arm pull frequency over time** — stacked area chart, x=episode, y=fraction of pulls each arm has received in the rolling window so far. Bandit should converge to favoring the best arm.

3. **Cumulative regret vs. oracle** — line plot, x=episode, y=cumulative (oracle_reward − bandit_reward). Should grow sub-linearly for a learning bandit.

4. **Per-arm running mean reward** — three lines (one per arm), x=episode, y=current empirical mean. Shows what the bandit "thinks" of each arm over time.

Summary table: per arm, total pulls, success rate, success rate confidence interval (Wilson score).

## 13. File / module structure

```
agentforge/
├── README.md
├── requirements.txt        # python deps, pinned
├── .env.example            # ANTHROPIC_API_KEY or OPENAI_API_KEY
├── data/
│   └── bird/               # download target, gitignored
├── src/
│   ├── __init__.py
│   ├── config.py           # constants (epsilon, n_episodes, model name, paths)
│   ├── tasks.py            # BIRD loader, Task dataclass
│   ├── schema.py           # load + format SQLite schema for prompts
│   ├── llm.py              # thin client wrapper (Anthropic or OpenAI)
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── base.py         # Workflow ABC, WorkflowResult
│   │   ├── direct.py
│   │   ├── explore.py
│   │   ├── decompose.py
│   │   └── prompts.py      # all prompt templates as string constants
│   ├── bandit.py           # EpsilonGreedyBandit
│   ├── executor.py         # safe SQL execution with timeout
│   ├── reward.py           # compute_reward
│   ├── logger.py           # JSONL writer
│   └── episode.py          # run_episode(task, bandit, workflows, llm, logger)
├── scripts/
│   ├── download_bird.py    # fetch + unpack BIRD dev set
│   ├── smoke_test.py       # section 11.1
│   ├── run_experiment.py   # main bandit run, takes --seed --n-episodes
│   ├── run_baselines.py    # random + single-arm + oracle
│   └── plot_results.py     # produces all 4 plots from logs/
├── logs/                   # JSONL output, gitignored except .gitkeep
└── plots/                  # PNG output, gitignored except .gitkeep
```

## 14. Out of scope for MVP (do not build)

- LangGraph, DSPy, LangChain, LangSmith, or any agent framework
- Contextual bandits (task features as state)
- Multi-domain expansion (BIRD only)
- Workflow improvement, generation, or learning
- Web UI / frontend
- Fine-tuning of LLMs
- More than 3 workflows
- More than 1 LLM provider per run (pick Anthropic OR OpenAI for a given experiment; can support both in `llm.py` but don't mix in one run)
- W&B, MLflow, or other experiment tracking — JSONL only
- Async / parallel episode execution — sequential only for MVP, reproducibility matters more than wall-clock speed

## 15. Milestone deliverables

### May 15 — Week 7 progress check
- [ ] Pipeline runs end-to-end on at least 30 episodes
- [ ] Smoke test results (workflow differentiation verified)
- [ ] At least one of the four plots produced
- [ ] Written progress summary (1–2 pages)

### May 29 — Final submission
- [ ] Full 50–100 episode runs across 3+ seeds
- [ ] All four plots, polished
- [ ] All baselines (random, three single-arm, oracle)
- [ ] Summary table with confidence intervals
- [ ] Video walkthrough demonstrating the system + showing the learning curve
- [ ] Public GitHub repo with README, reproducible setup

## 16. Implementation order (recommended)

1. `download_bird.py`, `tasks.py`, `schema.py` — get data loading working, verify Task objects
2. `executor.py`, `reward.py` — verify reward function works on a few known examples (run gold SQL on gold task, should get reward 1)
3. `llm.py` — thin wrapper, get one Anthropic or OpenAI call working
4. `workflows/direct.py` — simplest workflow, get it producing valid SQL on a few tasks
5. `episode.py` + `logger.py` — wire one workflow into the episode loop, verify JSONL output
6. **Smoke test gate:** implement the other two workflows, run smoke test, verify differentiation. Do not proceed past this point if workflows are too similar.
7. `bandit.py` — implement ε-greedy
8. `run_experiment.py` — full bandit loop
9. `run_baselines.py` — random + single-arm + oracle
10. `plot_results.py` — all four plots from collected logs

## 17. Known risks

| Risk | Mitigation |
|---|---|
| Workflows aren't sufficiently differentiated | Smoke test in step 6; if fails, redesign workflows before bandit |
| 50–100 episodes is small; curve will be noisy | Run 3–5 seeds, plot mean with std band |
| LLM API cost / rate limits | Cache LLM responses by `(prompt_hash, model)` in a local SQLite cache to avoid re-paying for re-runs |
| BIRD download size (33 GB full, ~3 GB dev) | Start with 5–10 DB subset; full dev set later |
| Gold SQL has documented noise | Accept; flag in writeup; not a blocker for MVP |
| Stochasticity in LLM despite temperature=0 | Some non-determinism remains in API; same prompt may give different output across days. Cache responses; document in writeup |
| Decompose workflow may produce broken composite SQL | Add a try/except fallback in workflow to return the last sub-SQL if composition fails; log as `workflow_error` field |

## 18. Vision context (for the writeup / video, not for code)

This MVP is bucket-one of a two-bucket framing:

- **Bucket 1 (this MVP):** Learn to *select* among fixed workflows. Workflows are static; selection policy improves.
- **Bucket 2 (future work):** Workflows themselves become learnable objects, improved via process supervision from expert demonstrations (related literature: AgentPRM, InversePRM, DSPy, IRL). Not part of MVP.

The MVP proves the bandit-selection layer works. The bucket-2 work would sit *on top of* this layer: a slow outer loop that discovers/improves workflows offline, with the bandit selecting among the current library online.

---

*End of spec. When implementing, refer back to sections 4, 14, and 17 frequently to keep scope tight.*
