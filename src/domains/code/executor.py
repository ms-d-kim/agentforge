"""Safe Python execution with a wall-clock timeout (mirrors src/executor.py).

Candidate code is written to a temp file together with the task's ``assert`` tests
and run in a *separate* Python subprocess. We never ``exec`` model output in this
interpreter. Two guards keep an episode from harming the host or hanging the run:

1. A static deny-list (``forbidden_patterns``) rejects code that reaches for the
   filesystem, the network, the process table, or dynamic ``eval``/``exec``. The
   check is intentionally conservative — a matched pattern means "not passed",
   never a crash.
2. ``subprocess.run(..., timeout=timeout_s)`` enforces a hard wall-clock budget;
   on expiry the child is killed and the task is "not passed".

``execute_python`` returns ``(passed, output)`` and is total: any failure mode
(syntax error, assertion failure, timeout, forbidden pattern, even an unexpected
exception while launching) maps to ``(False, <message>)`` rather than raising.
``compute_code_reward`` wraps it into the ``(reward, error_or_None)`` shape the
selector's reward signal expects.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Substrings that must not appear in candidate code. Matching is done on a
# whitespace-normalized copy so ``import   os`` or ``__import__ (`` still trip.
FORBIDDEN_PATTERNS = (
    "import os",
    "import sys",
    "from os",
    "from sys",
    "subprocess",
    "socket",
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "importlib",
    "shutil",
)


def _find_forbidden(code: str) -> Optional[str]:
    """Return the first forbidden pattern present in ``code``, else None."""
    # Collapse runs of whitespace so ``import   os`` matches ``import os`` and
    # ``open (`` matches ``open(``. Comparison is case-insensitive.
    normalized = re.sub(r"\s+", " ", code or "").lower()
    compact = normalized.replace(" (", "(")
    for pat in FORBIDDEN_PATTERNS:
        needle = pat.lower()
        if needle in normalized or needle in compact:
            return pat
    return None


def execute_python(code: str, tests: str, timeout_s: int = 10) -> Tuple[bool, str]:
    """Run ``code`` + ``tests`` in a sandboxed subprocess.

    Returns ``(passed, output)``:
      * ``passed`` is True only if the subprocess exits 0 (all asserts held).
      * ``output`` is a short human-readable status / captured stderr tail.

    Never raises — every failure mode becomes ``(False, message)``.
    """
    if not code or not code.strip():
        return False, "empty code"

    forbidden = _find_forbidden(code)
    if forbidden is not None:
        return False, "forbidden pattern: {}".format(forbidden)
    # Also screen the tests, defensively — they are trusted fixtures, but this
    # keeps the one execution path uniformly safe.
    forbidden_tests = _find_forbidden(tests)
    if forbidden_tests is not None:
        return False, "forbidden pattern in tests: {}".format(forbidden_tests)

    script = "{}\n\n{}\n".format(code.rstrip(), tests.rstrip())

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            tmp_path = fh.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout after {}s".format(timeout_s)

        if proc.returncode == 0:
            return True, "ok"

        # Non-zero exit: surface a compact tail of stderr (assertion / traceback).
        err = (proc.stderr or proc.stdout or "").strip()
        if len(err) > 600:
            err = "..." + err[-600:]
        return False, err or "exit code {}".format(proc.returncode)
    except Exception as exc:  # noqa: BLE001 — executor must never crash an episode
        return False, "executor error: {}".format(exc)
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


def compute_code_reward(task, code: str) -> Tuple[float, Optional[str]]:
    """Reward for a candidate solution: 1.0 iff all of ``task``'s asserts pass.

    Returns ``(reward, error_or_None)`` to match the SQL domain's reward shape
    (src/reward.py). A non-passing run returns ``(0.0, <message>)`` so callers /
    logs can see *why* it missed.
    """
    passed, output = execute_python(code, task.test_code)
    if passed:
        return 1.0, None
    return 0.0, output
