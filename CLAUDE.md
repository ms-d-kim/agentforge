# CLAUDE.md — AgentForge

> Project memory for Claude Code. Drop this at the repo root (`./CLAUDE.md`); it auto-loads every session.
> Repo: `ms-d-kim/agentforge` · CS 153: Frontier Systems, Stanford, Spring 2026 · solo project.

## Project overview

**AgentForge: A Self-Improving System for Learning Expert-Level Agentic Workflows.**

AgentForge is a **self-improving workflow selector** for text-to-SQL. An ε-greedy multi-armed
bandit learns, across repeated episodes, which of a small set of fixed SQL-generation workflow
templates to apply to a given task. The contribution is a *learned policy at the
workflow-selection layer* of an agent system — a layer where selection is usually hardcoded or
heuristic rather than learned.

Track: Automation / Agent Systems. Proposal scored 1/1.

## Framing rules (load-bearing — keep these straight in code, docstrings, README, demo)

- **Selector, not a self-rewriting agent.** The bandit makes **one discrete selection decision
  per task**. It does not modify workflow internals.
- **Bandit, not RL.** Selection is a one-step decision with no sequential credit assignment.
  Full RL only becomes relevant if/when learning *workflow internals* — deferred, out of MVP scope.
- **Composition-novel, not algorithm-novel.** ε-greedy is textbook; the contribution is a learned
  selector over *multi-step workflow compositions*. Closest commercial analogs are LLM routers
  (RouteLLM, Martian, Not Diamond); the distinguishing claim is selecting among multi-step
  workflows, not swapping a single model.

## Architecture (MVP)

Three fixed workflow arms, one bandit, one reward signal, one benchmark.

**Workflow arms (the actions):**
1. **Direct** — single-shot: prompt the model to emit SQL directly from question + schema.
2. **Schema-explore-then-write** — inspect/summarize relevant schema (tables, columns, keys),
   then generate SQL conditioned on that exploration.
3. **Decompose-then-compose** — break the question into sub-questions, solve each, compose the
   final SQL.

**Selector (the policy):** ε-greedy multi-armed bandit, **ε = 0.1**. Per-arm value estimates
(mean reward); with prob. ε explore a random arm, else exploit the current best. Updated online
over **50–100 episodes**.

**Reward:** **binary execution-match** on BIRD-SQL. Execute candidate SQL and gold SQL against the
SQLite DB; reward = 1 if result sets match (order-insensitive unless `ORDER BY`), else 0.
Execution accuracy — not string match.

**Benchmark:** BIRD-SQL (primary). LiveCodeBench subset is the backup, unused unless BIRD is
unavailable.

## Per-episode data flow

1. Sample a BIRD-SQL task (question + DB schema + gold SQL).
2. Bandit selects an arm (explore/exploit).
3. Selected workflow runs → candidate SQL (one or more model calls).
4. Execute candidate vs. gold on SQLite → binary reward.
5. Update the selected arm's value estimate.
6. Append one structured record to the JSONL log.

## Tech stack

- **Language:** plain Python — no agent framework.
- **Models:** Anthropic / OpenAI API, called directly (no orchestration layer).
- **DB / execution:** SQLite (BIRD databases).
- **Logging:** JSONL, one line per episode.
- **Plotting:** matplotlib.
- **Deferred:** LangGraph — expansion phase only. Do **not** introduce it into the MVP.

## Repository structure (as built)

Source under `src/`; entrypoints under `scripts/` (run as `python -m scripts.<name>` from repo
root). Full handoff spec at `docs/agentforge_spec.md`.

Reusable SDK (the generalizable takeaway): `src/selector.py` `WorkflowSelector` — domain-agnostic,
bring-your-own arms + reward, online learning, persistence. Policy ladder: epsilon-greedy | ucb1 |
thompson | linucb (contextual) → REINFORCE (multi-step, `src/rl/`). Exposed via top-level
`agentforge/` package (`pip install -e .` → `from agentforge import WorkflowSelector`). Four domains
under `src/domains/` (code, search, compaction) + BIRD (`src/workflows/`). Positioning vs other agent
SDKs in `docs/POSITIONING.md`; adversarial verification in `docs/VERIFICATION.md`.

```
src/
  selector.py           # WorkflowSelector SDK (domain-agnostic; the reusable product)
  bandit.py             # ε-greedy / UCB1 / Thompson policies
  contextual.py         # LinUCB contextual bandit + default featurizer (per-task selection)
  rl/                   # multi-step REINFORCE on a multi-hop iterate-or-stop env
  domains/              # code (tests-pass) / search (file-match) / compaction (downstream-QA)
  config.py tasks.py schema.py llm.py executor.py reward.py logger.py episode.py experiment.py metrics.py
  workflows/            # BIRD arms: base, prompts, direct, explore, decompose
scripts/                # download_bird, make_fixture, dry_run, smoke_test, run_experiment,
                        # run_baselines, run_real, plot_results, sweep
examples/               # byo_agent, code_domain, search_domain, compaction_domain, rl_multihop
tests/                  # offline suite (64 tests): python -m unittest discover -s tests
data/  logs/  plots/  results/  docs/
```

Offline verification (no API key, no BIRD download): `python -m scripts.dry_run` then
`python -m scripts.plot_results --prefix dry`. Arm indices are stable: 0=direct, 1=schema_explore,
2=decompose.

## Running experiments

Confirm the entrypoint and flags against the real repo before relying on these.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...    # and/or OPENAI_API_KEY
python run.py                   # or `python -m agentforge.run`
```

## Logging format (JSONL)

One JSON object per episode. Keep field names stable once chosen — the plots depend on them.

- `episode`, `task_id`, `db_id`
- `arm` (selected workflow name/index), `selection_reason` (`"explore"` | `"exploit"`)
- `reward` (0/1), `exec_match` (bool), `arm_value_after` (updated estimate)
- `n_model_calls`, `latency_s` (and `tokens` / `cost` if tracked)
- `predicted_sql`, `gold_sql`

## Metrics & plots (the evidence the demo needs)

The demo must show the system **learning to pick better workflows over time**.

- **Rolling accuracy by episode** — the learning curve (headline plot).
- **Per-workflow accuracy** — which arm is actually best.
- **Selection distribution over time** — show probability mass shifting toward the better arm(s).
- **Cumulative regret** — vs. an always-best-arm oracle.

## Scope discipline (immutable for the MVP)

The single most important project principle. Do **not** expand the MVP mid-stream. When a new
idea appears mid-task, default to *cite in related work or leave a stub*, not *absorb
architecturally*. Explicitly deferred / out of scope:

- **4th "iterative refinement / critic loop" arm** — possible later arm, not a separate product
  layer, not in MVP.
- **Cost-adjusted reward** (accuracy per dollar) — highest-ROI extension, but post-MVP.
- **Contextual bandit with task embeddings** — natural follow-on, post-MVP.
- **Second domain** — ship only a **config stub** to defuse the "overfitting to BIRD-SQL"
  critique; do not actually build a second domain for the MVP.
- **LangGraph, RL over workflow internals, self-evolving harnesses, workflow compilation** —
  related-work citations only.

## Conventions

- Keep all arms behind a **uniform interface** (same signature: `task -> candidate_sql`) so the
  bandit treats them as interchangeable actions.
- Keep arm names/indices **stable** across code, logs, and plots.
- **Seed** the bandit RNG and log the seed; determinism where it matters.
- **No secrets in the repo.** API keys via environment variables only.
- **AI-use disclosure:** the README must state how/where AI tools (including Claude Code) were
  used — this is a course requirement.

## Literature anchors (for README related-work)

AFlow, DSPy, Voyager, Reflexion, Self-Refine, DIN-SQL, DAIL-SQL, RouteLLM.
