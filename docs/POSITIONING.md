# Positioning: a self-improving agent SDK

> Competitive analysis backing AgentForge's core claim. Researched against primary sources,
> 2025–2026. TL;DR: every agent SDK lets you **write** how an agent acts and some let you **measure**
> it — but **none learns, online, which way of acting works for a given task, from your own reward,
> and persists that across runs.** AgentForge is that missing decision layer.

## The thesis

Agents are stochastic and tasks are heterogeneous: the right *strategy* (a direct prompt vs.
chain-of-thought vs. decompose vs. tool-use loop) differs per task. Today you hand-pick one strategy
and freeze it; "improvement" means a human reads traces and edits prompts between runs. AgentForge
makes the choice **learnable**: register your candidate workflows + a reward signal, and an online
bandit (→ contextual bandit → RL) learns which to run per task and **keeps improving across runs.**

## The white space (a four-way intersection that — to our knowledge — is unoccupied)

**To our knowledge (June 2026)** — this is a best-effort survey, not a maintained matrix — no shipped
system does all four at once:

1. **Online** — learns at *runtime* from *your own* reward (not offline preference labels, not a fixed
   trainset, not just accumulated memory text).
2. **A policy over multi-step *workflows*** — an arm is an entire procedure (tools, control flow,
   budget), not a single prompt or a single model on a single turn.
3. **Persistence across runs** — the dispatch policy compounds over time and survives restarts.
4. **A domain-agnostic SDK** — `register workflows + a reward callback`, like TF-Agents is for generic
   bandits, but native to LLM-agent workflows.

The ingredients each exist in isolation; the *combination* does not. **Caveat:** the landscape moves
fast, and the "model routers are mostly offline" row is the softest — newer online adaptive routers
(e.g. ParetoBandit, arXiv:2604.00136) adapt at runtime, but still pick a *single model per turn*, not a
multi-step *workflow*. The four-way claim is the honest scope; counterexamples welcome (open an issue).

## Landscape

| System | Online + persists? | Policy over multi-step **workflows**? | Reward/signal | Shipped SDK? |
|---|---|---|---|---|
| **Google ADK** | No | No (deterministic workflows) | Offline golden-trajectory eval (`final_response_match_v2`, `tool_trajectory_avg_score`) | Yes |
| **OpenAI Agents SDK** | No (*"do not modify config during the run"*) | No | Trace→feedback→evals→**human-approved diff** | Yes |
| **MS Agent Framework** (AutoGen+SK) | No (Magentic adapts *within* a run, forgets after) | No | Magentic progress-ledger + stall-counter (in-run heuristic) | Yes |
| **NVIDIA NeMo Agent Toolkit** | No (offline RL/DPO → **model artifact**) | No (optimizes one workflow) | Offline finetuning harness + profiler; online RL is *roadmap* | Yes |
| **LangGraph + LangSmith** | No (memory = data store) | No | LangSmith feedback-scores / online evals — loop stays **open** | Yes |
| **CrewAI / LlamaIndex** | No | No | Tracing + memory; eval external | Yes |
| **DSPy / MIPROv2 / GEPA** | No (compile-time) | No (optimizes prompt *text* of one program) | scalar metric (+ text in GEPA) | Yes |
| **TextGrad / Trace (OptoPrime)** | Offline loop (Trace can take live feedback) | No (optimizes *params of one agent*) | textual gradient / numeric / compiler errors | Yes |
| **RouteLLM / Not Diamond / Martian** | Mostly offline classifiers | No (picks one **model**, single-turn) | offline preference / cost | Yes (product) |
| **PILOT** (adaptive routing) | **Yes** (LinUCB online) | No (one **model**, single-turn) | bandit reward + budget | No (paper) |
| **ARC** (Configure Agentic Systems) | **No** (PPO+SFT offline, then frozen) | **Yes** (SMDP over workflow+tools+budget) | α·correct − cost | No (research code) |
| **Letta/MemGPT, Voyager** | Yes (persists *memory/skills*) | No (no reward-driven policy) | none / env feedback | Yes / research |
| **AgentForge (this repo)** | **Yes** | **Yes** | **your callback** (tests / eval / 👍 / exec-match) | **Yes** |

## Closest relatives — and exactly how we differ

- **ARC** — *"Learning to Configure Agentic AI Systems"* (arXiv:2602.11574). The only work that frames
  the problem as we do: an SMDP where each configuration is a temporally-extended option and a policy
  selects (workflow, tools, budget) per query. **But it is offline** (PPO + SFT on a fixed trainset,
  then frozen) and **research code, not an SDK.** AgentForge is *"ARC, but online, persistent, and
  productized."*
- **PILOT** (arXiv:2508.21141) — genuinely online (LinUCB contextual bandit adapting from live
  feedback under a budget). **But its arms are single models on single-turn queries.** AgentForge is
  *"PILOT's online adaptation, but the arms are full multi-step workflows, shipped as an SDK with
  cross-run persistence."*
- **Trace / OptoPrime** (microsoft/Trace) — the closest *usable SDK with a feedback loop* — but it
  **optimizes the parameters of one agent in place**, not a *policy that dispatches among workflows*.
- **DSPy / GEPA** — **complementary, not competing**: they offline-optimize the prompts of *one*
  program. AgentForge selects *among* programs online. Natural stack: GEPA-optimize each arm offline,
  then let AgentForge learn which arm to dispatch online.
- **Model routers** answer "which model for this prompt?" (a flat set, single turn). **Workflow
  selection** commits to a variable-length, variable-cost *trajectory* whose reward arrives only after
  the whole procedure runs — a harder credit-assignment/exploration problem (an SMDP, not a one-shot
  bandit).
- **Orchestration SDKs** (ADK, OpenAI, MS, LangGraph, CrewAI, LlamaIndex) are where you *define an arm*
  and *capture a reward* — but the loop is deliberately **open**: a human (or a coding assistant) edits
  prompts between runs. AgentForge **closes the loop automatically.**

## What we borrow (good ideas, credited)

- **Google ADK** golden-trajectory eval metrics → ready-made **reward functions** (`exec-match`,
  `response-match`, `tool-trajectory`).
- **OpenAI Agents SDK** `@function_tool` decorator ergonomics; the trace→feedback→eval pipeline —
  we swap the human-approved diff for an automatic policy update.
- **LangSmith** feedback-score-on-run schema `(run_id, key, score, comment)` → our **reward-ingestion
  contract**, so reward capture is tool-agnostic.
- **NVIDIA NeMo** trajectory abstraction + profiler → **cost-aware reward shaping** (`reward =
  success − λ·cost`) and framework-agnostic wrapping.
- **MS Magentic** progress-ledger + stall-counter → **intermediate reward / early-abort** signal.
- **LangGraph** checkpointer threads → **persistence + resumability** of policy + per-episode context.
- **GEPA** `ScoreWithFeedback {score, feedback}` → reward can carry **text**, not just a scalar.
- **CrewAI** Crews-vs-Flows → arms may be **autonomous or scripted** under one interface.

## Honest framing for the writeup / video

> AgentForge is the online, reward-driven, cross-run-persistent **selection layer** that the agent-SDK
> ecosystem is missing. Everyone else optimizes *inside* a fixed strategy (better prompts, better
> weights, faster execution) or measures it (eval, tracing). AgentForge optimizes *over a portfolio of
> strategies* and routes traffic to winners online — an orthogonal axis that, to our knowledge, no
> shipped agent SDK occupies.

## Sources

ADK: github.com/google/adk-python · google.github.io/adk-docs · codelabs.developers.google.com/adk-eval.
OpenAI Agents SDK: openai.github.io/openai-agents-python · developers.openai.com/cookbook/examples/agents_sdk/agent_improvement_loop.
MS Agent Framework: learn.microsoft.com/en-us/agent-framework · .../orchestrations/magentic.
NVIDIA NeMo Agent Toolkit: github.com/NVIDIA/NeMo-Agent-Toolkit · docs.nvidia.com/nemo/agent-toolkit.
LangGraph/LangSmith: github.com/langchain-ai/langgraph · langchain.com/langsmith.
CrewAI: github.com/crewAIInc/crewAI. LlamaIndex Workflows: llamaindex.ai/workflows.
DSPy/GEPA: github.com/stanfordnlp/dspy · arXiv:2507.19457. TextGrad: arXiv:2406.07496.
Trace/OptoPrime: arXiv:2406.16218 · microsoft.github.io/Trace.
RouteLLM: arXiv:2406.18665. PILOT: arXiv:2508.21141. ParetoBandit (online model routing): arXiv:2604.00136.
ARC: arXiv:2602.11574 · github.com/somsagar07/Context_Optimization.
Bandits+LLMs survey: arXiv:2505.13355. AFlow: arXiv:2410.10762. Agent Lightning: arXiv:2508.03680.
Letta/MemGPT: github.com/letta-ai/letta. Reflexion: arXiv:2303.11366. Voyager: arXiv:2305.16291.
AutoGen AgentOptimizer: arXiv:2402.11359.
