"""Prompt templates + output parsing for the code-generation arms (house style
mirrors src/workflows/prompts.py).

Three arms, distinct prompting strategies:
  * ``DIRECT_PROMPT``         — one shot: problem -> code.
  * ``REPAIR_PROMPT``         — feed back failing code + the test error to fix it
                                (used by the iterative arm).
  * ``PLAN_PROMPT`` /
    ``IMPLEMENT_PROMPT``      — plan helper functions, then implement + compose
                                (used by the decompose arm).

``extract_code`` pulls a Python block out of a model response (preferring ```python
fences), the code-domain analog of ``extract_sql``.
"""
from __future__ import annotations

import re
from typing import Optional

_PY_FENCE = re.compile(r"```(?:python|py)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull Python source out of a model response (prefers ```python fences).

    Falls back to any fenced block, then to the raw text. Mirrors
    ``extract_sql`` in src/workflows/base.py.
    """
    if not text:
        return ""
    m = _PY_FENCE.search(text) or _ANY_FENCE.search(text)
    code = (m.group(1) if m else text)
    return code.strip()


DIRECT_PROMPT = """\
You are an expert Python programmer. Write a single, self-contained Python \
function that solves the problem below. Define exactly the function \
`{function_name}` and any helpers it needs. Do not include tests, examples, or \
explanation.

Problem:
{prompt}

Return ONLY the Python code, wrapped in ```python ... ``` fences.
"""

REPAIR_PROMPT = """\
You are an expert Python programmer debugging a failing solution. The function \
`{function_name}` below does not pass its tests. Fix the bug so all tests pass. \
Keep the same function name and signature.

Problem:
{prompt}

Current code:
```python
{code}
```

Test failure / error:
{error}

Return ONLY the corrected Python code, wrapped in ```python ... ``` fences. No \
explanation.
"""

PLAN_PROMPT = """\
You are an expert Python programmer planning a solution. For the problem below, \
list the helper functions you would write (name + one-line purpose) and the order \
in which `{function_name}` would call them. Keep it to at most 4 helpers; if the \
problem is simple, say so and propose a single function.

Problem:
{prompt}

Return ONLY a short plan as a JSON array of strings (one step per element). No \
prose outside the array.
"""

IMPLEMENT_PROMPT = """\
You are an expert Python programmer. Implement the solution following the plan. \
Define exactly the function `{function_name}` plus any helper functions the plan \
calls for, composing them into the final answer. Do not include tests or \
explanation.

Problem:
{prompt}

Plan:
{plan}

Return ONLY the Python code, wrapped in ```python ... ``` fences.
"""
