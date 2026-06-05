"""Code corpus for the agentic-search domain: a thin, real ``grep`` over a tree.

The corpus is whatever Python lives under a fixed root (default: this repo's
``src/`` directory). Retrieval is intentionally primitive — a single
``grep -rn --include=*.py`` shell-out — because the *interesting* part of this
domain is the **strategy** that decides what to grep for and how to disambiguate
the hits, not a clever index. That keeps the arms (src/domains/search/workflows.py)
comparable to a future agent that issues the same primitive tool calls.

All hit paths are normalized to be **relative to the repo root** (POSIX
separators) so rewards and gold labels compare cleanly across machines.

Offline note: ``grep`` is a standard POSIX tool, so the whole domain — corpus,
arms (under the FakeLLMClient), reward, tests — runs with **no network and no API
key**. We invoke grep via ``subprocess.run`` with an argument *list* (never
``shell=True``) so the ``--include=*.py`` glob is passed literally to grep instead
of being expanded by the user's shell.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Repo root = three levels up from this file: src/domains/search/corpus.py -> repo.
REPO_ROOT = Path(__file__).resolve().parents[3]

# The corpus we search over. Fixed to the AgentForge source tree per the domain
# spec; every gold answer in tasks.py is a real location under here.
DEFAULT_CORPUS_ROOT = REPO_ROOT / "src"

# Subtrees grep skips by default. We exclude ``domains/`` (the selector's *other*
# domain packages — and this search domain's own source) so the corpus is the
# stable "AgentForge core" the gold answers live in, and the domain never matches
# its own keyword tables / docstrings. ``__pycache__`` is just noise.
DEFAULT_EXCLUDE_DIRS = ("domains", "__pycache__")

# A grep hit: (file_relative_to_repo_root, line_number, line_text).
GrepHit = Tuple[str, int, str]


@dataclass(frozen=True)
class Corpus:
    """A searchable code tree rooted at ``root`` (default: the repo's ``src/``).

    ``exclude_dirs`` names directories grep skips (default: the selector's other
    domain packages + ``__pycache__``), keeping the searched corpus the stable
    AgentForge core where every gold answer lives.
    """

    root: Path = DEFAULT_CORPUS_ROOT
    exclude_dirs: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_EXCLUDE_DIRS)

    def __post_init__(self) -> None:
        # Normalize ``root`` to an absolute path without mutating the frozen field
        # via assignment (object.__setattr__ is the dataclass-frozen escape hatch).
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "exclude_dirs", tuple(self.exclude_dirs))

    # -- file listing ------------------------------------------------------- #
    def files(self) -> List[str]:
        """All ``*.py`` files in the corpus, as repo-relative POSIX paths (sorted)."""
        return sorted(
            self._rel(p)
            for p in self.root.rglob("*.py")
            if not any(part in self.exclude_dirs for part in p.parts)
        )

    # -- search ------------------------------------------------------------- #
    def grep(self, query: str, scope: Optional[List[str]] = None) -> List[GrepHit]:
        """Return ``(file, line, text)`` for every ``*.py`` line matching ``query``.

        ``query`` is a plain (fixed-ish) pattern passed to ``grep`` as a basic
        regular expression — callers in workflows.py use literal keywords and the
        anchored definition patterns ``^class `` / ``^def ``.

        ``scope`` optionally restricts the search to a set of repo-relative files
        (the *narrow* step of the broad-then-narrow arm). Paths in ``scope`` that
        fall outside the corpus root are ignored. An empty ``scope`` list means
        "nothing to search" and yields ``[]`` without invoking grep.
        """
        excludes = [f"--exclude-dir={d}" for d in self.exclude_dirs]
        if scope is not None:
            targets = self._resolve_scope(scope)
            if not targets:
                return []
            cmd = ["grep", "-rn", "--include=*.py", *excludes, query, *targets]
        else:
            cmd = ["grep", "-rn", "--include=*.py", *excludes, query, str(self.root)]

        # grep exit code 1 == "no matches" (not an error); >1 is a real failure.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            return []
        return self._parse(proc.stdout)

    # -- internals ---------------------------------------------------------- #
    def _resolve_scope(self, scope: List[str]) -> List[str]:
        """Map repo-relative (or absolute) scope paths to absolute paths inside root."""
        out: List[str] = []
        for s in scope:
            p = Path(s)
            if not p.is_absolute():
                p = REPO_ROOT / s
            p = p.resolve()
            try:
                p.relative_to(self.root)
            except ValueError:
                continue  # outside the corpus — ignore
            if p.exists():
                out.append(str(p))
        return out

    def _parse(self, stdout: str) -> List[GrepHit]:
        hits: List[GrepHit] = []
        for line in stdout.splitlines():
            # grep -n format: "<path>:<lineno>:<text>"
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            path, lineno, text = parts
            try:
                n = int(lineno)
            except ValueError:
                continue
            hits.append((self._rel(Path(path)), n, text))
        return hits

    @staticmethod
    def _rel(path: Path) -> str:
        """Repo-relative POSIX path (falls back to the absolute path if outside)."""
        p = path.resolve()
        try:
            return p.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return p.as_posix()
