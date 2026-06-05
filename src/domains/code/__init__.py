"""Code-generation domain for the AgentForge selector.

A second instantiation of the domain-agnostic ``WorkflowSelector``
(src/selector.py): the "arms" are three Python-codegen strategies (``direct``,
``iterative``, ``decompose``) and the reward is "do the unit tests pass?" — a
cheap, fully automatic 0/1 signal. The selector learns online which strategy wins
on these problems and concentrates pulls there, exactly as the BIRD text-to-SQL
experiment does with SQL arms + execution-match reward.

Public surface::

    from src.domains.code import (
        TASKS, CodegenTask,          # the problem fixture
        make_arms,                   # {name: task -> code} for the selector
        compute_code_reward,         # (task, code) -> (reward, error|None)
        make_fake_responder,         # offline LLM stand-in (no API key)
    )

``make_fake_responder`` is what lets the example + tests run with zero API calls:
it inspects each prompt to tell which task and which *step* (direct vs repair vs
decompose) it is serving, and returns code engineered so the arms differentiate —
see its docstring for the exact success profile.
"""
from __future__ import annotations

from typing import Callable

from .executor import compute_code_reward, execute_python
from .prompts import extract_code
from .tasks import TASKS, TASKS_BY_ID, CodegenTask
from .workflows import decompose_arm, direct_arm, iterative_arm, make_arms

__all__ = [
    "TASKS",
    "TASKS_BY_ID",
    "CodegenTask",
    "make_arms",
    "direct_arm",
    "iterative_arm",
    "decompose_arm",
    "compute_code_reward",
    "execute_python",
    "extract_code",
    "make_fake_responder",
]


# --------------------------------------------------------------------------- #
# Offline fake LLM responder.
# --------------------------------------------------------------------------- #
# A buggy variant per task that FAILS at least one assert but is a plausible
# "first draft" — small enough that the model could repair it. Keyed by task_id.
# These are used by the fake to make the three arms differentiate:
#
#   * direct    : returns the correct reference impl ONLY for easy tasks; for
#                 medium/hard it returns the buggy draft -> fails.
#   * iterative : same buggy first draft, BUT the repair step returns the correct
#                 reference impl, so it recovers on medium/hard (its edge).
#   * decompose : returns the correct impl for easy AND medium, buggy for hard ->
#                 the middling arm.
#
# Net success profile under the fake (10 tasks: 3 easy, 3 medium, 4 hard):
#   direct    -> easy only            = 3/10
#   decompose -> easy + medium        = 6/10
#   iterative -> easy + medium + hard = 10/10   (the best arm)
_BUGGY_IMPLS = {
    # off-by-one: starts the accumulator at 1 instead of 0.
    "sum_of_list": (
        "def sum_of_list(nums):\n"
        "    total = 1\n"
        "    for n in nums:\n"
        "        total += n\n"
        "    return total\n"
    ),
    # forgets case-insensitivity -> misses uppercase vowels.
    "count_vowels": (
        "def count_vowels(s):\n"
        "    return sum(1 for c in s if c in 'aeiou')\n"
    ),
    # does not dedupe at all.
    "remove_duplicates": (
        "def remove_duplicates(items):\n"
        "    return list(items)\n"
    ),
    # off-by-one in the index -> returns fibonacci(n+1).
    "fibonacci": (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    ),
    # concatenates then forgets to sort.
    "merge_sorted_lists": (
        "def merge_sorted_lists(a, b):\n"
        "    return list(a) + list(b)\n"
    ),
    # uses k directly as an index (off-by-one: ignores 1-indexing).
    "kth_largest": (
        "def kth_largest(nums, k):\n"
        "    return sorted(nums, reverse=True)[k]\n"
    ),
    # ignores bracket *type* — only balances counts.
    "is_balanced_parens": (
        "def is_balanced_parens(s):\n"
        "    depth = 0\n"
        "    for c in s:\n"
        "        if c in '([{':\n"
        "            depth += 1\n"
        "        elif c in ')]}':\n"
        "            depth -= 1\n"
        "            if depth < 0:\n"
        "                return False\n"
        "    return depth == 0\n"
    ),
    # only breaks the run on an *adjacent* repeat -> overcounts e.g. 'abcabcbb'.
    "longest_substring_no_repeat": (
        "def longest_substring_no_repeat(s):\n"
        "    best = 0\n"
        "    cur = 0\n"
        "    prev = None\n"
        "    for c in s:\n"
        "        if c == prev:\n"
        "            cur = 1\n"
        "        else:\n"
        "            cur += 1\n"
        "        prev = c\n"
        "        best = max(best, cur)\n"
        "    return best\n"
    ),
    # only enforces pattern->word, not the reverse direction (not a bijection).
    "word_pattern_match": (
        "def word_pattern_match(pattern, s):\n"
        "    words = s.split()\n"
        "    if len(pattern) != len(words):\n"
        "        return False\n"
        "    p2w = {}\n"
        "    for p, w in zip(pattern, words):\n"
        "        if p in p2w and p2w[p] != w:\n"
        "            return False\n"
        "        p2w[p] = w\n"
        "    return True\n"
    ),
    # counts only substitutions (Hamming-style) -> wrong for unequal lengths.
    "edit_distance": (
        "def edit_distance(a, b):\n"
        "    if len(a) != len(b):\n"
        "        return abs(len(a) - len(b))\n"
        "    return sum(1 for x, y in zip(a, b) if x != y)\n"
    ),
}


def _identify_task(prompt: str):
    """Find which CodegenTask a prompt is about (by embedded function name).

    Function names are unique across the fixture and every template includes the
    target ``function_name``, so this is an unambiguous match.
    """
    # Prefer the longest matching name so e.g. a name that is a substring of
    # another can't shadow it (none currently, but keep it robust).
    best = None
    for task in TASKS:
        if task.function_name in (prompt or ""):
            if best is None or len(task.function_name) > len(best.function_name):
                best = task
    return best


def make_fake_responder() -> Callable[[str, object], str]:
    """Build an offline ``responder(prompt, system) -> str`` for ``FakeLLMClient``.

    The responder inspects the prompt to determine (a) which task it is about and
    (b) which workflow STEP is calling it, then returns code chosen so the three
    arms differentiate:

      * Repair step (iterative arm) -> always the correct reference impl. This is
        why ``iterative`` recovers on the hard tasks ``direct`` fails.
      * Plan step (decompose arm)   -> a short JSON plan (no code yet).
      * Implement step (decompose)  -> reference impl for easy/medium, buggy for
        hard. The middling arm.
      * Direct generation           -> reference impl for easy, buggy otherwise.
        Shared by the ``direct`` arm and as the iterative arm's first draft.

    Deterministic and API-free.
    """

    def _fence(code: str) -> str:
        return "```python\n{}```".format(code)

    def responder(prompt: str, system: object) -> str:
        task = _identify_task(prompt)
        if task is None:
            # Unknown task: a harmless stub that won't pass any real test.
            return _fence("def _unknown():\n    return None\n")

        ref = task.reference_impl
        buggy = _BUGGY_IMPLS.get(task.task_id, ref)
        difficulty = task.difficulty

        # 1) Repair step (iterative arm) — recognizable by the failure section.
        if "Test failure / error:" in prompt or "Current code:" in prompt:
            return _fence(ref)

        # 2) Plan step (decompose arm) — asks for a JSON array plan, no code.
        if "list the helper functions" in prompt or "as a JSON array of strings" in prompt:
            return (
                '["parse the input", "compute the core result with a helper", '
                '"return the value from {}"]'.format(task.function_name)
            )

        # 3) Implement step (decompose arm) — follows a plan.
        if "Implement the solution following the plan" in prompt or "Plan:" in prompt:
            # Decompose lands easy + medium, misses hard.
            return _fence(ref if difficulty in ("easy", "medium") else buggy)

        # 4) Plain direct generation (direct arm, and iterative's first draft).
        #    Correct only on easy; buggy on medium/hard so iterative's repair edge
        #    (and decompose's planning edge) can show up.
        return _fence(ref if difficulty == "easy" else buggy)

    return responder
