"""The three code-generation arms (strategies) for the selector.

Each arm is a ``CodegenTask -> code_string`` callable, so the domain-agnostic
``WorkflowSelector`` (src/selector.py) treats them as interchangeable actions.
They are genuinely differentiated:

  * ``direct``    — one LLM call, parse out the code. Cheapest; relies entirely on
                    the model getting it right first try.
  * ``iterative`` — generate, then RUN the tests in the sandbox; if they fail,
                    feed the code + the failure back to the model to repair, up to
                    ``max_rounds`` times. Strictly stronger than ``direct`` on
                    tasks where the first draft is buggy, at the cost of extra
                    calls + executions.
  * ``decompose`` — ask for a helper-function plan, then implement + compose. Adds
                    structure but no execution feedback; lands the middle ground.

``make_arms(llm)`` returns the ``{name: fn}`` dict to hand the selector.

Each arm wraps its LLM calls in the ``track`` context manager (src/workflows/
base.py) for parity with the SQL workflows; ``CodegenTask`` is the unit of work in
place of the SQL ``Task``. Arms never see ``reference_impl`` — only ``prompt`` and
``function_name`` flow into the templates.
"""
from __future__ import annotations

from typing import Callable, Dict

from ...workflows.base import track
from .executor import execute_python
from .prompts import (
    DIRECT_PROMPT,
    IMPLEMENT_PROMPT,
    PLAN_PROMPT,
    REPAIR_PROMPT,
    extract_code,
)

# How many repair rounds the iterative arm gets after the initial generation.
MAX_REPAIR_ROUNDS = 3


def direct_arm(task, llm) -> str:
    """Single-shot generation."""
    with track(llm):
        raw = llm.complete(
            DIRECT_PROMPT.format(function_name=task.function_name, prompt=task.prompt)
        )
    return extract_code(raw)


def iterative_arm(task, llm, max_rounds: int = MAX_REPAIR_ROUNDS) -> str:
    """Generate, run the tests, and repair on failure (up to ``max_rounds``).

    Returns the first candidate that passes; if none do, returns the last
    candidate produced (so a downstream reward still has something to grade).
    """
    with track(llm):
        code = extract_code(
            llm.complete(
                DIRECT_PROMPT.format(
                    function_name=task.function_name, prompt=task.prompt
                )
            )
        )
        passed, output = execute_python(code, task.test_code)
        rounds = 0
        while not passed and rounds < max_rounds:
            rounds += 1
            repaired = llm.complete(
                REPAIR_PROMPT.format(
                    function_name=task.function_name,
                    prompt=task.prompt,
                    code=code,
                    error=output,
                )
            )
            code = extract_code(repaired) or code
            passed, output = execute_python(code, task.test_code)
    return code


def decompose_arm(task, llm) -> str:
    """Plan helper functions, then implement + compose.

    Robustness (mirrors the SQL decompose arm): if implementation parsing yields
    nothing, fall back to a single direct generation rather than returning empty.
    """
    with track(llm):
        plan = llm.complete(
            PLAN_PROMPT.format(function_name=task.function_name, prompt=task.prompt)
        )
        impl_raw = llm.complete(
            IMPLEMENT_PROMPT.format(
                function_name=task.function_name,
                prompt=task.prompt,
                plan=plan.strip(),
            )
        )
        code = extract_code(impl_raw)
        if not code:
            code = extract_code(
                llm.complete(
                    DIRECT_PROMPT.format(
                        function_name=task.function_name, prompt=task.prompt
                    )
                )
            )
    return code


def make_arms(llm) -> Dict[str, Callable[[object], str]]:
    """Build the ``{arm_name: task -> code_string}`` dict for the selector.

    Arm order is stable (direct, iterative, decompose) to match the SQL domain's
    convention of stable arm indices.
    """
    return {
        "direct": lambda task: direct_arm(task, llm),
        "iterative": lambda task: iterative_arm(task, llm),
        "decompose": lambda task: decompose_arm(task, llm),
    }
