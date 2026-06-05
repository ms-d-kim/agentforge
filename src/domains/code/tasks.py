"""Code-generation task fixture for the AgentForge selector domain.

Each ``CodegenTask`` is a self-contained Python problem: a natural-language
``prompt``, the ``function_name`` the solution must define, and ``test_code`` made
of inline ``assert`` statements. The executor (executor.py) runs candidate code
plus these asserts; reward is 1.0 iff every assert passes.

Each task also carries a hidden ``reference_impl`` — a known-correct solution.
This is used ONLY by the offline fake responder (prompts.py / workflows.py never
expose it to an arm) so the example and tests can run with zero API calls while
still letting the three arms genuinely differentiate. Difficulty ("easy" /
"medium" / "hard") drives that differentiation: under the fake, ``direct`` only
nails easy tasks, ``iterative`` repairs its way to the hard ones, and
``decompose`` lands the middle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CodegenTask:
    """A single self-contained code-generation problem.

    Attributes:
        task_id: stable identifier (also used by the fake responder to look the
            problem up from a prompt).
        prompt: natural-language problem statement shown to the arms.
        function_name: the function the candidate code must define.
        test_code: inline ``assert`` statements exercising ``function_name``.
        difficulty: one of ``"easy"``, ``"medium"``, ``"hard"``.
        reference_impl: hidden known-correct implementation. Used only by the
            offline fake; never shown to an arm.
    """

    task_id: str
    prompt: str
    function_name: str
    test_code: str
    difficulty: str
    reference_impl: str


# --------------------------------------------------------------------------- #
# Fixture — ten problems spanning easy -> hard.
# Reference impls are deliberately plain; they are graded by the same asserts the
# arms are, so "correct" means "passes the tests", nothing fancier.
# --------------------------------------------------------------------------- #
TASKS: List[CodegenTask] = [
    CodegenTask(
        task_id="sum_of_list",
        prompt=(
            "Write a function `sum_of_list(nums)` that returns the sum of a list "
            "of numbers. The sum of an empty list is 0."
        ),
        function_name="sum_of_list",
        test_code=(
            "assert sum_of_list([]) == 0\n"
            "assert sum_of_list([5]) == 5\n"
            "assert sum_of_list([1, 2, 3, 4]) == 10\n"
            "assert sum_of_list([-1, 1, -2, 2]) == 0\n"
        ),
        difficulty="easy",
        reference_impl=(
            "def sum_of_list(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    return total\n"
        ),
    ),
    CodegenTask(
        task_id="count_vowels",
        prompt=(
            "Write a function `count_vowels(s)` that returns the number of vowels "
            "(a, e, i, o, u, case-insensitive) in the string `s`."
        ),
        function_name="count_vowels",
        test_code=(
            "assert count_vowels('') == 0\n"
            "assert count_vowels('xyz') == 0\n"
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('AEIOU') == 5\n"
            "assert count_vowels('Programming') == 3\n"
        ),
        difficulty="easy",
        reference_impl=(
            "def count_vowels(s):\n"
            "    return sum(1 for c in s.lower() if c in 'aeiou')\n"
        ),
    ),
    CodegenTask(
        task_id="remove_duplicates",
        prompt=(
            "Write a function `remove_duplicates(items)` that returns a new list "
            "with duplicates removed, preserving the order of first appearance."
        ),
        function_name="remove_duplicates",
        test_code=(
            "assert remove_duplicates([]) == []\n"
            "assert remove_duplicates([1, 1, 1]) == [1]\n"
            "assert remove_duplicates([1, 2, 1, 3, 2]) == [1, 2, 3]\n"
            "assert remove_duplicates(['a', 'b', 'a', 'c']) == ['a', 'b', 'c']\n"
        ),
        difficulty="easy",
        reference_impl=(
            "def remove_duplicates(items):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
    ),
    CodegenTask(
        task_id="fibonacci",
        prompt=(
            "Write a function `fibonacci(n)` that returns the n-th Fibonacci "
            "number (0-indexed), where fibonacci(0) == 0 and fibonacci(1) == 1."
        ),
        function_name="fibonacci",
        test_code=(
            "assert fibonacci(0) == 0\n"
            "assert fibonacci(1) == 1\n"
            "assert fibonacci(2) == 1\n"
            "assert fibonacci(7) == 13\n"
            "assert fibonacci(10) == 55\n"
        ),
        difficulty="medium",
        reference_impl=(
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
    ),
    CodegenTask(
        task_id="merge_sorted_lists",
        prompt=(
            "Write a function `merge_sorted_lists(a, b)` that merges two "
            "already-sorted ascending lists into a single sorted ascending list."
        ),
        function_name="merge_sorted_lists",
        test_code=(
            "assert merge_sorted_lists([], []) == []\n"
            "assert merge_sorted_lists([1, 3, 5], []) == [1, 3, 5]\n"
            "assert merge_sorted_lists([1, 4], [2, 3, 5]) == [1, 2, 3, 4, 5]\n"
            "assert merge_sorted_lists([1, 1, 2], [1, 3]) == [1, 1, 1, 2, 3]\n"
        ),
        difficulty="medium",
        reference_impl=(
            "def merge_sorted_lists(a, b):\n"
            "    i = j = 0\n"
            "    out = []\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            out.append(a[i]); i += 1\n"
            "        else:\n"
            "            out.append(b[j]); j += 1\n"
            "    out.extend(a[i:])\n"
            "    out.extend(b[j:])\n"
            "    return out\n"
        ),
    ),
    CodegenTask(
        task_id="kth_largest",
        prompt=(
            "Write a function `kth_largest(nums, k)` that returns the k-th largest "
            "element of `nums` (k is 1-indexed, so k=1 is the maximum). Duplicates "
            "count as distinct positions."
        ),
        function_name="kth_largest",
        test_code=(
            "assert kth_largest([3, 1, 2], 1) == 3\n"
            "assert kth_largest([3, 1, 2], 2) == 2\n"
            "assert kth_largest([5, 5, 4], 2) == 5\n"
            "assert kth_largest([1, 2, 3, 4, 5], 5) == 1\n"
        ),
        difficulty="medium",
        reference_impl=(
            "def kth_largest(nums, k):\n"
            "    return sorted(nums, reverse=True)[k - 1]\n"
        ),
    ),
    CodegenTask(
        task_id="is_balanced_parens",
        prompt=(
            "Write a function `is_balanced_parens(s)` that returns True iff the "
            "brackets in `s` are balanced. Consider three kinds: (), [], {}. "
            "Non-bracket characters are ignored."
        ),
        function_name="is_balanced_parens",
        test_code=(
            "assert is_balanced_parens('') == True\n"
            "assert is_balanced_parens('()') == True\n"
            "assert is_balanced_parens('([{}])') == True\n"
            "assert is_balanced_parens('(]') == False\n"
            "assert is_balanced_parens('([)]') == False\n"
            "assert is_balanced_parens('a(b)c[d]') == True\n"
        ),
        difficulty="hard",
        reference_impl=(
            "def is_balanced_parens(s):\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    stack = []\n"
            "    for c in s:\n"
            "        if c in '([{':\n"
            "            stack.append(c)\n"
            "        elif c in pairs:\n"
            "            if not stack or stack.pop() != pairs[c]:\n"
            "                return False\n"
            "    return not stack\n"
        ),
    ),
    CodegenTask(
        task_id="longest_substring_no_repeat",
        prompt=(
            "Write a function `longest_substring_no_repeat(s)` that returns the "
            "length of the longest substring of `s` with no repeating characters."
        ),
        function_name="longest_substring_no_repeat",
        test_code=(
            "assert longest_substring_no_repeat('') == 0\n"
            "assert longest_substring_no_repeat('abcabcbb') == 3\n"
            "assert longest_substring_no_repeat('bbbbb') == 1\n"
            "assert longest_substring_no_repeat('pwwkew') == 3\n"
            "assert longest_substring_no_repeat('abcdef') == 6\n"
        ),
        difficulty="hard",
        reference_impl=(
            "def longest_substring_no_repeat(s):\n"
            "    last = {}\n"
            "    start = 0\n"
            "    best = 0\n"
            "    for i, c in enumerate(s):\n"
            "        if c in last and last[c] >= start:\n"
            "            start = last[c] + 1\n"
            "        last[c] = i\n"
            "        best = max(best, i - start + 1)\n"
            "    return best\n"
        ),
    ),
    CodegenTask(
        task_id="word_pattern_match",
        prompt=(
            "Write a function `word_pattern_match(pattern, s)` that returns True "
            "iff the space-separated words in `s` follow `pattern` under a "
            "bijection: each pattern letter maps to exactly one word and vice "
            "versa. Example: pattern 'abba', s 'dog cat cat dog' -> True."
        ),
        function_name="word_pattern_match",
        test_code=(
            "assert word_pattern_match('abba', 'dog cat cat dog') == True\n"
            "assert word_pattern_match('abba', 'dog cat cat fish') == False\n"
            "assert word_pattern_match('aaaa', 'dog cat cat dog') == False\n"
            "assert word_pattern_match('abba', 'dog dog dog dog') == False\n"
            "assert word_pattern_match('ab', 'dog cat') == True\n"
            "assert word_pattern_match('a', 'dog cat') == False\n"
        ),
        difficulty="hard",
        reference_impl=(
            "def word_pattern_match(pattern, s):\n"
            "    words = s.split()\n"
            "    if len(pattern) != len(words):\n"
            "        return False\n"
            "    p2w = {}\n"
            "    w2p = {}\n"
            "    for p, w in zip(pattern, words):\n"
            "        if p in p2w and p2w[p] != w:\n"
            "            return False\n"
            "        if w in w2p and w2p[w] != p:\n"
            "            return False\n"
            "        p2w[p] = w\n"
            "        w2p[w] = p\n"
            "    return True\n"
        ),
    ),
    CodegenTask(
        task_id="edit_distance",
        prompt=(
            "Write a function `edit_distance(a, b)` that returns the Levenshtein "
            "edit distance between strings `a` and `b`: the minimum number of "
            "single-character insertions, deletions, or substitutions to turn `a` "
            "into `b`."
        ),
        function_name="edit_distance",
        test_code=(
            "assert edit_distance('', '') == 0\n"
            "assert edit_distance('abc', 'abc') == 0\n"
            "assert edit_distance('', 'abc') == 3\n"
            "assert edit_distance('kitten', 'sitting') == 3\n"
            "assert edit_distance('flaw', 'lawn') == 2\n"
        ),
        difficulty="hard",
        reference_impl=(
            "def edit_distance(a, b):\n"
            "    m, n = len(a), len(b)\n"
            "    dp = list(range(n + 1))\n"
            "    for i in range(1, m + 1):\n"
            "        prev = dp[0]\n"
            "        dp[0] = i\n"
            "        for j in range(1, n + 1):\n"
            "            cur = dp[j]\n"
            "            if a[i - 1] == b[j - 1]:\n"
            "                dp[j] = prev\n"
            "            else:\n"
            "                dp[j] = 1 + min(prev, dp[j], dp[j - 1])\n"
            "            prev = cur\n"
            "    return dp[n]\n"
        ),
    ),
]

# Lookup by id, for the fake responder and tests.
TASKS_BY_ID = {t.task_id: t for t in TASKS}
