"""Arms + reward + offline fake for the agentic-search domain.

Three strategies ("arms") locate a target file in a code corpus. They share one
uniform interface — ``task -> retrieved_file_path`` (a repo-relative string) — so
the :class:`~src.selector.WorkflowSelector` treats them as interchangeable actions
and learns online which one wins on these tasks. Each arm issues **real**
``grep`` calls (src/domains/search/corpus.py); only the *decisions* (what keyword
to grep, which candidate file answers the question) come from the LLM, so swapping
in :class:`~src.llm.FakeLLMClient` makes the whole thing run offline.

The three arms span a difficulty gradient on purpose:

* ``simple``            — one keyword, one grep, take the top hit's file. Fast and
  right when the question names the symbol; brittle on conceptual questions.
* ``iterative``         — keyword grep, then if the hits span several files, ask
  the model "which of these files most likely answers the question?" (up to 3
  rounds, refining the keyword). Resolves conceptual questions the single grep
  can't.
* ``broad_then_narrow`` — grep a broad concept, collect the candidate files, then
  grep *within* them for a definition (``^class ``/``^def ``) and pick the best
  defining file. Strong when the answer is a class/def whose name also shows up as
  imports/calls elsewhere.

================================================================================
HOW THIS EXTENDS TO A MULTI-STEP RL AGENT  (design note — NOT implemented here)
================================================================================
Today each arm is a **single selection decision**: the bandit picks one arm, the
arm runs to completion, we score the final file. That matches the AgentForge
framing (a selector, one discrete choice per task — see CLAUDE.md).

The same machinery is one step away from a *sequential* agent, and this domain was
shaped to be that RL testbed:

* **State**  = the search history so far: the question + the list of
  ``(action, observation)`` pairs, where each observation is a set of ``GrepHit``s
  already seen (which files/lines, how ambiguous).
* **Actions** = the primitive tool moves the arms already make internally —
  ``grep(keyword)``, ``narrow(scope, pattern)``, ``answer(file)`` — i.e. *choose
  the next query* rather than *choose a whole pre-baked strategy*.
* **Transition** = run the chosen grep, append its hits to the history.
* **Reward**  = :func:`compute_search_reward` on ``answer(file)`` (terminal), with
  an optional per-step cost so the agent learns to stop searching once confident.
* **Episode** = repeat select-action → observe until ``answer`` or a step budget.

So ``iterative``'s "grep → inspect → maybe re-grep → answer" loop is exactly a
hand-coded rollout of that MDP. A later phase swaps the bandit-over-arms for a
policy-over-next-actions (the ``next-action choice`` conditioned on the search
history + observations) and keeps the corpus, reward, and fake responder. We keep
the arms single-decision here on purpose; the RL is out of scope for this file.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional

from src.llm import BaseLLMClient

from .corpus import Corpus, GrepHit
from .tasks import TASKS, SearchTask

# ---------------------------------------------------------------------------- #
# Prompts (kept as constants, house style per src/workflows/prompts.py).
# ---------------------------------------------------------------------------- #
# Each prompt carries a stable ``[step:NAME]`` marker on the first line so the
# offline fake can route by step without depending on the surrounding prose (and a
# real model simply ignores it). The four markers below are the only contract
# between the prompts and make_fake_responder().
STEP_KEYWORD = "[step:keyword]"
STEP_REFINE = "[step:refine_keyword]"
STEP_CHOOSE = "[step:choose_file]"
STEP_BROAD = "[step:broad_concept]"

KEYWORD_PROMPT = (
    STEP_KEYWORD
    + """
You are helping locate code in a Python repository by grepping.
Extract the SINGLE best search keyword (one identifier, no spaces) to grep for to
answer this question. Reply with ONLY the keyword.

Question: {question}
"""
)

REFINE_KEYWORD_PROMPT = (
    STEP_REFINE
    + """
Grepping for "{keyword}" was too ambiguous (it matched many files). Give ONE
different, more specific search keyword (one identifier, no spaces) to locate the
answer. Reply with ONLY the keyword.

Question: {question}
Files matched so far: {files}
"""
)

CHOOSE_FILE_PROMPT = (
    STEP_CHOOSE
    + """
Several files matched the search. Which ONE file most likely answers the question?
Reply with ONLY the file path, exactly as listed.

Question: {question}
Candidates:
{candidates}
"""
)

BROAD_CONCEPT_PROMPT = (
    STEP_BROAD
    + """
To locate where something is *defined* in a Python repo, give ONE broad concept
keyword (one identifier, no spaces) to grep for first. Reply with ONLY the keyword.

Question: {question}
"""
)

# Definition-line anchors used by the narrowing step. grep basic-regex: '^class '.
_DEF_PATTERNS = ["^class ", "^def "]


# ---------------------------------------------------------------------------- #
# Small helpers shared by the arms.
# ---------------------------------------------------------------------------- #
def _clean_keyword(raw: str) -> str:
    """Normalize the model's keyword into a grep pattern.

    Most replies are a single identifier (``ResponseCache``). We also let a
    *definition pattern* through verbatim — ``class Foo`` / ``def foo`` (optionally
    ``^``-anchored) — because "grep for the symbol's definition" is a legitimate,
    more-precise single query a searcher would issue. Anything else collapses to
    its first identifier token.
    """
    raw = (raw or "").strip()
    m = re.match(r"\^?(?:class|def)\s+[A-Za-z_][A-Za-z0-9_]*", raw)
    if m:
        return m.group(0)
    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", raw)
    return m.group(0) if m else raw


def _files_in_order(hits: List[GrepHit]) -> List[str]:
    """Distinct files in first-seen order (grep order)."""
    seen: List[str] = []
    for f, _line, _text in hits:
        if f not in seen:
            seen.append(f)
    return seen


def _top_file(hits: List[GrepHit]) -> Optional[str]:
    """The single best file: most-frequent match, ties broken by grep order."""
    if not hits:
        return None
    counts = Counter(f for f, _l, _t in hits)
    order = _files_in_order(hits)
    return max(order, key=lambda f: (counts[f], -order.index(f)))


def _snippet_for(hits: List[GrepHit], file: str) -> str:
    """A representative line for ``file`` (first hit), trimmed for the prompt."""
    for f, _line, text in hits:
        if f == file:
            return text.strip()[:120]
    return ""


# ---------------------------------------------------------------------------- #
# Arm 1: simple — one keyword, one grep, top hit.
# ---------------------------------------------------------------------------- #
def simple_search(llm: BaseLLMClient, corpus: Corpus, task: SearchTask) -> str:
    """Extract one keyword, grep once, return the file of the top hit."""
    keyword = _clean_keyword(llm.complete(KEYWORD_PROMPT.format(question=task.question)))
    hits = corpus.grep(keyword)
    return _top_file(hits) or ""


# ---------------------------------------------------------------------------- #
# Arm 2: iterative — grep, disambiguate via the LLM, refine up to 3 rounds.
# ---------------------------------------------------------------------------- #
def iterative_search(
    llm: BaseLLMClient, corpus: Corpus, task: SearchTask, max_rounds: int = 3
) -> str:
    """Keyword grep → if ambiguous, ask which file answers it; refine up to 3x."""
    keyword = _clean_keyword(llm.complete(KEYWORD_PROMPT.format(question=task.question)))
    best = ""
    for _round in range(max_rounds):
        hits = corpus.grep(keyword)
        files = _files_in_order(hits)
        if not files:
            # Dead end: try a fresh, more specific keyword.
            keyword = _clean_keyword(
                llm.complete(
                    REFINE_KEYWORD_PROMPT.format(
                        keyword=keyword, question=task.question, files="(none)"
                    )
                )
            )
            continue
        if len(files) == 1:
            return files[0]
        # Ambiguous: let the model pick the file that answers the question.
        candidates = "\n".join(f"- {f}: {_snippet_for(hits, f)}" for f in files)
        choice = (llm.complete(
            CHOOSE_FILE_PROMPT.format(question=task.question, candidates=candidates)
        ) or "").strip()
        picked = next((f for f in files if f in choice or choice in f), None)
        best = picked or _top_file(hits) or best
        if picked:
            return picked
        # No confident pick — refine the keyword and try once more.
        keyword = _clean_keyword(
            llm.complete(
                REFINE_KEYWORD_PROMPT.format(
                    keyword=keyword, question=task.question, files=", ".join(files)
                )
            )
        )
    return best


# ---------------------------------------------------------------------------- #
# Arm 3: broad_then_narrow — broad grep, then narrow to a definition.
# ---------------------------------------------------------------------------- #
def broad_then_narrow_search(
    llm: BaseLLMClient, corpus: Corpus, task: SearchTask
) -> str:
    """Broad grep → candidate files → narrow grep for ``^class``/``^def`` → best."""
    concept = _clean_keyword(llm.complete(BROAD_CONCEPT_PROMPT.format(question=task.question)))
    broad_hits = corpus.grep(concept)
    candidates = _files_in_order(broad_hits)
    if not candidates:
        return ""

    # Narrow: search the candidate files for an actual definition line, and prefer
    # the definition whose name matches the broad concept keyword.
    def_hits: List[GrepHit] = []
    for pattern in _DEF_PATTERNS:
        def_hits.extend(corpus.grep(pattern, scope=candidates))

    if def_hits:
        concept_lc = concept.lower()
        named = [h for h in def_hits if concept_lc in h[2].lower()]
        pool = named or def_hits
        return _top_file(pool) or candidates[0]

    # No definitions inside the candidates — fall back to the broad top hit.
    return _top_file(broad_hits) or candidates[0]


# ---------------------------------------------------------------------------- #
# Reward.
# ---------------------------------------------------------------------------- #
def _posix(path: str) -> str:
    return (path or "").replace("\\", "/")


def compute_search_reward(task: SearchTask, retrieved_file: str):
    """Return ``(reward, error_or_None)`` for a retrieved file vs. the gold file.

    * ``1.0`` — exact match (same repo-relative POSIX path, or same basename).
    * ``0.5`` — wrong file but in the **same directory** as the gold file.
    * ``0.0`` — different directory (or nothing retrieved).
    """
    got = _posix(retrieved_file)
    gold = _posix(task.gold_file)
    if not got:
        return 0.0, "no_file_retrieved"
    if got == gold or got.rsplit("/", 1)[-1] == gold.rsplit("/", 1)[-1]:
        return 1.0, None
    got_dir = got.rsplit("/", 1)[0] if "/" in got else ""
    gold_dir = gold.rsplit("/", 1)[0] if "/" in gold else ""
    if got_dir == gold_dir:
        return 0.5, f"same_dir_wrong_file: {got}"
    return 0.0, f"wrong_file: {got}"


# ---------------------------------------------------------------------------- #
# Arm factory — bind an llm + corpus into the uniform ``task -> file`` interface.
# ---------------------------------------------------------------------------- #
def make_arms(
    llm: BaseLLMClient, corpus: Optional[Corpus] = None
) -> Dict[str, Callable[[SearchTask], str]]:
    """Build the three arms as ``task -> retrieved_file_path`` callables."""
    corpus = corpus or Corpus()
    return {
        "simple": lambda task: simple_search(llm, corpus, task),
        "iterative": lambda task: iterative_search(llm, corpus, task),
        "broad_then_narrow": lambda task: broad_then_narrow_search(llm, corpus, task),
    }


# ---------------------------------------------------------------------------- #
# Offline fake responder — designed so the arms DIFFERENTIATE.
# ---------------------------------------------------------------------------- #
# The fake inspects the prompt to recover (a) which task it is (by matching the
# question text) and (b) which step is being asked (keyword? refine? choose-file?
# broad-concept?), then returns decisions that give each arm a distinct competence
# profile under deterministic, network-free conditions:
#
#   * EXACT questions     -> ``simple`` already greps the named symbol and the top
#     hit is the gold file: it wins. ``iterative`` also wins (single file → return).
#   * CONCEPT questions   -> the keyword the fake hands every arm is AMBIGUOUS (it
#     matches many files), so ``simple`` takes a wrong top hit and misses; only the
#     ``choose-file`` follow-up (which ``iterative`` issues, ``simple`` does not)
#     recovers the gold file. So ``iterative`` wins, ``simple`` loses.
#   * DEFINITION questions-> the broad keyword is ambiguous *and* the name appears
#     as imports/calls in several files, so neither ``simple`` nor the file-choice
#     heuristic lands it; but ``broad_then_narrow`` re-greps the candidates for
#     ``^class``/``^def`` and picks the defining file: it wins.
#
# Concretely: the fake gives CONCEPT/DEFINITION tasks a deliberately broad keyword
# so the single-grep arm is misled, and supplies the correct file ONLY through the
# follow-up "which file?" answer (used by iterative) — except DEFINITION tasks,
# where even the follow-up is withheld so only the narrowing arm succeeds.

# Per-task fake decisions, keyed by task_id.
#   ambiguous_keyword : what every arm greps first (broad on concept/definition).
#   file_choice       : the answer the fake gives to the "which file?" step
#                       (None = the fake refuses to disambiguate for this task).
_FAKE: Dict[str, Dict[str, Optional[str]]] = {
    # EXACT — the question names the symbol; grepping its *definition* pins the
    # single gold file, so simple (and iterative) land it immediately.
    "bandit_eps": {"ambiguous_keyword": "class EpsilonGreedyBandit", "file_choice": "src/bandit.py"},
    "fake_llm": {"ambiguous_keyword": "class FakeLLMClient", "file_choice": "src/llm.py"},
    "contextual_bandit": {"ambiguous_keyword": "class LinUCBBandit", "file_choice": "src/contextual.py"},
    # CONCEPT — broad keyword (top hit is a NON-gold file, so ``simple`` misses);
    # only ``iterative``'s follow-up "which file?" step recovers the gold.
    "reward_bird": {"ambiguous_keyword": "reward", "file_choice": "src/reward.py"},
    "response_cache": {"ambiguous_keyword": "sqlite", "file_choice": "src/llm.py"},
    "selector_persist": {"ambiguous_keyword": "json", "file_choice": "src/selector.py"},
    "prompt_templates": {"ambiguous_keyword": "prompt", "file_choice": "src/workflows/prompts.py"},
    # DEFINITION — broad keyword AND no follow-up (file_choice=None): only
    # broad_then_narrow's ^class/^def step finds the definition.
    "safe_executor": {"ambiguous_keyword": "execute", "file_choice": None},
    "ucb1_bandit": {"ambiguous_keyword": "UCB1", "file_choice": None},
    "query_timeout": {"ambiguous_keyword": "sqlite3", "file_choice": None},
}

# broad_then_narrow's first ("broad concept") keyword, per task.
#   * DEFINITION tasks: a concept whose name appears on the gold ``^class``/``^def``
#     line, so the named-definition preference lands the gold file.
#   * CONCEPT tasks: a concept with NO matching definition line, so the narrowing
#     step finds an unrelated definition and MISSES — only ``iterative`` wins these.
# (EXACT tasks fall through to the per-task ``ambiguous_keyword`` = ``class <Sym>``,
# which the narrow step pins straight to the defining gold file.)
_FAKE_BROAD_CONCEPT = {
    # definition: name is on the gold definition line.
    "safe_executor": "execute",  # -> ^def execute_sql in src/executor.py
    "ucb1_bandit": "UCB1",       # -> ^class UCB1Bandit in src/bandit.py
    "query_timeout": "Timeout",  # -> ^class QueryTimeout in src/executor.py
    # concept: a broad word with no ^class/^def of that name → narrowing misses.
    "reward_bird": "reward",
    "response_cache": "cache",
    "selector_persist": "policy",
    "prompt_templates": "prompt",
}


def _match_task(prompt: str) -> Optional[SearchTask]:
    """Recover which task a prompt is about by matching its question text."""
    for t in TASKS:
        if t.question in prompt:
            return t
    return None


def make_fake_responder():
    """Offline ``responder(prompt, system) -> str`` driving the three arms.

    Deterministic and network-free. See the module note above for the intended
    per-arm competence profile (simple↦exact, iterative↦concept,
    broad_then_narrow↦definition).
    """

    def responder(prompt: str, system: Optional[str] = None) -> str:
        task = _match_task(prompt)
        if task is None:
            return "agentforge"  # harmless generic keyword

        spec = _FAKE.get(task.task_id, {})

        # Step routing by the stable [step:*] marker each prompt carries.
        if STEP_CHOOSE in prompt:
            # iterative's disambiguation step. Withhold the answer on definition
            # tasks (file_choice=None) so only broad_then_narrow can win them.
            choice = spec.get("file_choice")
            return choice if choice else "unsure"

        if STEP_BROAD in prompt:
            # broad_then_narrow's first step.
            return _FAKE_BROAD_CONCEPT.get(task.task_id, spec.get("ambiguous_keyword", "agentforge"))

        if STEP_REFINE in prompt:
            # iterative's refine step — keep it broad so the file-choice step is
            # what actually resolves the task (don't accidentally hand it the gold).
            return spec.get("ambiguous_keyword", "agentforge")

        # Default (STEP_KEYWORD): the initial keyword step (simple + iterative).
        return spec.get("ambiguous_keyword", "agentforge")

    return responder
