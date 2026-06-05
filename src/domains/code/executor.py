"""Run model-generated Python against a task's asserts, in a subprocess.

⚠️ NOT A SECURITY SANDBOX. This is a *benchmark harness* that runs model solutions to
known toy tasks in a separate, resource-limited subprocess. It applies best-effort
isolation only:

  * a tripwire deny-list for the obvious cases (NOT a security boundary — trivially
    bypassable, e.g. ``getattr(__builtins__, 'open')``),
  * an isolated-mode interpreter (``python -I -S -B``: ignores env/PYTHON*/user-site,
    writes no bytecode),
  * a throwaway working directory and a minimal environment,
  * best-effort POSIX resource limits (CPU time, address space, zero file-WRITE size)
    — enforcement is OS-dependent: Linux honors these; **macOS does NOT enforce
    RLIMIT_AS / RLIMIT_FSIZE**, so the only reliable cross-platform protection is the
    wall-clock ``timeout``.

These help bound runaway processes but do NOT block file *reads* or network access,
and the tripwire is bypassable. **Do not run adversarial or untrusted code here.** For
untrusted code use an OS-level sandbox (container / gVisor / firejail / seccomp).

``execute_python`` returns ``(passed, output)`` and is total: every failure mode
(syntax error, assertion failure, timeout, tripwire hit, launch error) maps to
``(False, <message>)`` rather than raising. ``compute_code_reward`` wraps it into the
``(reward, error_or_None)`` shape the selector's reward signal expects.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore

# Quick tripwire for the obvious cases. This is a convenience filter, NOT a security
# boundary — see the module docstring. Real protection is the resource-limited,
# isolated subprocess below.
TRIPWIRE_PATTERNS = (
    "import os", "import sys", "from os", "from sys", "subprocess", "socket",
    "open(", "__import__", "eval(", "exec(", "importlib", "shutil",
)


def _tripwire(code: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", code or "").lower()
    compact = normalized.replace(" (", "(")
    for pat in TRIPWIRE_PATTERNS:
        needle = pat.lower()
        if needle in normalized or needle in compact:
            return pat
    return None


def _rlimits(timeout_s: int, mem_mb: int = 1024):
    """preexec_fn (POSIX): cap CPU seconds, address space, and file-write size."""
    def _apply():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 1, timeout_s + 2))
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # block file writes
        except Exception:  # noqa: BLE001 - best effort; never break the child launch
            pass
    return _apply


def execute_python(code: str, tests: str, timeout_s: int = 10) -> Tuple[bool, str]:
    """Run ``code`` + ``tests`` in a resource-limited subprocess. Never raises.

    Returns ``(passed, output)`` — ``passed`` is True only if the child exits 0
    (all asserts held). See the module docstring: this is isolation, not a sandbox.
    """
    if not code or not code.strip():
        return False, "empty code"

    hit = _tripwire(code) or _tripwire(tests)
    if hit is not None:
        return False, "tripwire pattern: {}".format(hit)

    script = "{}\n\n{}\n".format(code.rstrip(), tests.rstrip())
    workdir = tempfile.mkdtemp(prefix="agentforge_exec_")
    try:
        script_path = os.path.join(workdir, "candidate.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(script)

        kwargs = dict(
            capture_output=True, text=True, timeout=timeout_s, cwd=workdir,
            env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir},
        )
        if os.name == "posix" and resource is not None:
            kwargs["preexec_fn"] = _rlimits(timeout_s)
        try:
            proc = subprocess.run([sys.executable, "-I", "-S", "-B", script_path], **kwargs)
        except subprocess.TimeoutExpired:
            return False, "timeout after {}s".format(timeout_s)

        if proc.returncode == 0:
            return True, "ok"
        err = (proc.stderr or proc.stdout or "").strip()
        if len(err) > 600:
            err = "..." + err[-600:]
        return False, err or "exit code {}".format(proc.returncode)
    except Exception as exc:  # noqa: BLE001 — executor must never crash an episode
        return False, "executor error: {}".format(exc)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def compute_code_reward(task, code: str) -> Tuple[float, Optional[str]]:
    """Reward for a candidate solution: 1.0 iff all of ``task``'s asserts pass.

    Returns ``(reward, error_or_None)`` to match the SQL domain's reward shape.
    """
    passed, output = execute_python(code, task.test_code)
    return (1.0, None) if passed else (0.0, output)
