"""Tasks for the agentic-search domain: locate a target in the AgentForge corpus.

Each :class:`SearchTask` asks *where* something lives and records the **real**
location in this repo's ``src/`` tree (verified gold ``file`` + ``symbol``). The
set is deliberately a mix of difficulties so the arms differentiate:

* ``exact``  — the question names the symbol almost verbatim (e.g. "EpsilonGreedy
  bandit"). A single keyword grep on the top hit usually nails these.
* ``concept`` — the question is phrased conceptually ("where is the reward
  computed", "which file caches responses"); the literal keyword is ambiguous or
  spread across files, so a follow-up "which file answers this?" step wins.
* ``definition`` — the answer is a *class/def definition* whose name also appears
  (as an import / call) in several other files; anchoring on ``^class ``/``^def ``
  inside the candidate files is what disambiguates.

These difficulty tags are the natural feature for a contextual policy later, but
the selector treats tasks opaquely here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SearchTask:
    """A 'where does X live?' query with its verified gold location in the corpus."""

    task_id: str
    question: str
    gold_file: str  # repo-relative POSIX path, e.g. "src/bandit.py"
    gold_symbol: str  # the class/def the question is really about
    difficulty: str  # "exact" | "concept" | "definition"


# Gold locations confirmed against the live repo (grep). Keep these in sync if the
# corpus moves; the offline test asserts a couple of them directly.
TASKS: List[SearchTask] = [
    SearchTask(
        task_id="bandit_eps",
        question="Where is the epsilon-greedy bandit defined?",
        gold_file="src/bandit.py",
        gold_symbol="EpsilonGreedyBandit",
        difficulty="exact",
    ),
    SearchTask(
        task_id="fake_llm",
        question="Where is the fake LLM client?",
        gold_file="src/llm.py",
        gold_symbol="FakeLLMClient",
        difficulty="exact",
    ),
    SearchTask(
        task_id="contextual_bandit",
        question="Where is the contextual bandit?",
        gold_file="src/contextual.py",
        gold_symbol="LinUCBBandit",
        difficulty="exact",
    ),
    SearchTask(
        task_id="reward_bird",
        question="Where is the reward computed for BIRD tasks?",
        gold_file="src/reward.py",
        gold_symbol="compute_reward",
        difficulty="concept",
    ),
    SearchTask(
        task_id="response_cache",
        question="Which file caches model responses?",
        gold_file="src/llm.py",
        gold_symbol="ResponseCache",
        difficulty="concept",
    ),
    SearchTask(
        task_id="selector_persist",
        question="Where does the selector persist its policy?",
        gold_file="src/selector.py",
        gold_symbol="WorkflowSelector",
        difficulty="concept",
    ),
    SearchTask(
        task_id="prompt_templates",
        question="Where are the workflow prompt templates?",
        gold_file="src/workflows/prompts.py",
        gold_symbol="DIRECT_PROMPT",
        difficulty="concept",
    ),
    SearchTask(
        task_id="safe_executor",
        question="Where is the safe SQL executor?",
        gold_file="src/executor.py",
        gold_symbol="execute_sql",
        difficulty="definition",
    ),
    SearchTask(
        task_id="ucb1_bandit",
        question="Where is the UCB1 bandit policy defined?",
        gold_file="src/bandit.py",
        gold_symbol="UCB1Bandit",
        difficulty="definition",
    ),
    SearchTask(
        task_id="query_timeout",
        question="Where is the SQL query timeout exception defined?",
        gold_file="src/executor.py",
        gold_symbol="QueryTimeout",
        difficulty="definition",
    ),
]


def get_tasks() -> List[SearchTask]:
    """Return the embedded benchmark task list (a fresh list copy)."""
    return list(TASKS)
