"""Central configuration: constants, paths, model + experiment knobs.

Everything tunable lives here so scripts and modules stay parameter-free at the
call site. Environment variables override the LLM-facing values. Arm names and
indices defined here are STABLE across code, logs, and plots (CLAUDE.md
conventions) — do not reorder.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
BIRD_DIR = DATA_DIR / "bird"          # BIRD dev download target (gitignored)
FIXTURE_DIR = DATA_DIR / "fixture"    # tiny synthetic BIRD-like fixture for offline tests
LOGS_DIR = REPO_ROOT / "logs"         # JSONL episode logs, one file per run
PLOTS_DIR = REPO_ROOT / "plots"       # matplotlib PNG output
LLM_CACHE_PATH = REPO_ROOT / "llm_cache.sqlite"  # (prompt_hash, model) -> response

# Load .env (OPENROUTER_API_KEY / ANTHROPIC_API_KEY / overrides) if python-dotenv is present.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # noqa: BLE001 — dotenv is optional; env vars still work without it
    pass

# --------------------------------------------------------------------------- #
# Bandit (spec §4, §8) — STABLE arm order
# --------------------------------------------------------------------------- #
ARM_NAMES = ["direct", "schema_explore", "decompose"]
N_ARMS = len(ARM_NAMES)
EPSILON = 0.1
# If True, force one pull of each arm before the bandit takes over, so initial
# means aren't all 0 (spec §8 note). This is forced exploration, not optimistic
# value initialization.
FORCED_INIT = True
DEFAULT_SEED = 0
SEEDS = [0, 1, 2]  # multi-seed default (spec §4: 3–5 seeds)

# --------------------------------------------------------------------------- #
# Experiment (spec §4, §11)
# --------------------------------------------------------------------------- #
N_EPISODES = 60        # within spec's 50–100 per run
SMOKE_N_TASKS = 20     # pre-flight differentiation check (spec §11.1)

# --------------------------------------------------------------------------- #
# LLM (spec §4: temperature 0, one provider per run)
# --------------------------------------------------------------------------- #
LLM_PROVIDER = os.environ.get("AGENTFORGE_PROVIDER", "anthropic")  # anthropic | openai | openrouter
ANTHROPIC_MODEL = os.environ.get("AGENTFORGE_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.environ.get("AGENTFORGE_OPENAI_MODEL", "gpt-4o")
# OpenRouter is OpenAI-compatible; model ids are vendor-prefixed (e.g. openai/gpt-4o-mini,
# anthropic/claude-sonnet-4.6). A mid-tier default leaves headroom for workflow strategy to matter.
OPENROUTER_MODEL = os.environ.get("AGENTFORGE_OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
TEMPERATURE = 0.0
MAX_TOKENS = 2048
LLM_CACHE_ENABLED = True

# --------------------------------------------------------------------------- #
# Execution / reward (spec §9)
# --------------------------------------------------------------------------- #
EXEC_TIMEOUT_S = 30
FLOAT_ROUND = 4  # round floats before set comparison

# --------------------------------------------------------------------------- #
# BIRD data (spec §6) — see scripts/download_bird.py
# --------------------------------------------------------------------------- #
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
# For the first runs, restrict to a handful of databases to keep runtime small
# (spec §6). Empty list = use all databases found. Fill with db_ids after download.
BIRD_DB_SUBSET: list[str] = []

# --------------------------------------------------------------------------- #
# Domain (spec §14: ship a config stub only — second domain is deferred)
# --------------------------------------------------------------------------- #
DOMAIN = "bird_sql"
# DOMAINS = {"bird_sql": ..., "<second_domain>": ...}  # deferred, defuses overfitting critique


def active_model() -> str:
    """Model id for the currently selected provider."""
    return {
        "anthropic": ANTHROPIC_MODEL,
        "openai": OPENAI_MODEL,
        "openrouter": OPENROUTER_MODEL,
    }.get(LLM_PROVIDER, ANTHROPIC_MODEL)
