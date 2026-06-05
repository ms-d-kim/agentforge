"""Context-compaction QA tasks — long agent logs with one buried answer fact.

Each task is a long synthetic "agent log" (~40-120 lines) containing exactly one
line that states the answer fact, surrounded by distractor lines. A question asks
for that fact; ``answer`` is the gold short answer. A *compaction strategy* must
shrink the log to a budget; the downstream reader then answers using only the
compacted context. Reward is whether the strategy PRESERVED the answer line.

The tasks are deliberately spread across two axes so that different compaction
strategies win on different tasks (no single strategy dominates everywhere):

  * POSITION of the answer line: early / middle / late in the log.
  * KEYWORD OVERLAP between the answer line and the question, relative to the
    decoy distractors: high / low.

To pull the three strategies apart, some logs also seed a few HIGH-overlap
*decoy* lines (distractors that share the question's keywords but not the
answer). Decoys are clustered in one region of the log:

  * ``truncate`` (keep the last K lines) wins when the fact is late.
  * ``extractive`` (keep the highest question-overlap lines) wins when the fact
    out-scores the decoys, but is fooled when decoys consume its budget — so it
    misses low-overlap facts.
  * ``hierarchical`` (one representative line per chunk) keeps the fact whenever
    it stands out within its local chunk, independent of position and of how many
    decoys sit in *other* chunks — the robust all-rounder.

This domain is inspired by the prior project *ee392c-agent-mem* — "EE 392C:
Memory-Lifetime Characterization of LLM Agent-Workflow Replays" (Stanford,
Spring 2026; Minseok Kim & Kristen Guernsey), whose ``compaction_agent`` trace
ingests a log and summarizes it but only MEASURES the memory behavior. Here the
same compaction idea becomes selector ARMS with a real downstream-QA reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class CompactionTask:
    """One long-context QA instance.

    Attributes:
        task_id: stable identifier.
        context: the long agent log, as a tuple of lines. Exactly one line states
            the answer fact; the rest are distractors (some may be high-overlap
            decoys that share the question's keywords but not the answer).
        question: asks for the buried fact.
        answer: gold short answer (string/number-as-string) expected in the line.
        difficulty: ``"easy" | "medium" | "hard"`` (informational; harder tasks
            bury the fact further from the question's keywords / at the edges or
            behind more decoys).
    """

    task_id: str
    context: Tuple[str, ...]
    question: str
    answer: str
    difficulty: str

    @property
    def num_lines(self) -> int:
        return len(self.context)

    @property
    def context_text(self) -> str:
        """The log rendered as a single newline-joined string."""
        return "\n".join(self.context)

    def fact_line_index(self) -> int:
        """Index of the single line that contains the gold answer fact."""
        norm = self.answer.lower()
        for i, line in enumerate(self.context):
            if norm in line.lower():
                return i
        raise ValueError(f"task {self.task_id}: answer not found in any context line")


# --------------------------------------------------------------------------- #
# Synthetic log construction
# --------------------------------------------------------------------------- #
# Neutral "agent log" distractor lines. None contain any of the gold answers and
# none share the question keywords below, so they score zero overlap — the fact
# line and any seeded decoys are the only lines that can be singled out by
# keyword overlap. (Keeps ``extractive`` and ``hierarchical`` well-behaved.)
#
# These are deliberately written to share NO content words with any task
# question, so they all score zero keyword overlap. That keeps the only
# nonzero-overlap lines the fact line and the seeded decoys, which makes the
# strategy outcomes precise and reproducible.
_DISTRACTORS: Tuple[str, ...] = (
    "[trace] listdir /srv/app yielded 41 dirents",
    "[trace] opened /srv/app/main.py spanning 8123 bytes",
    "[trace] warm lookup absent for bundle-7af; rebuilding",
    "[trace] backoff hop 2 following a transient 503 upstream",
    "[trace] tokenizer prewarm done in 0.4s",
    "[trace] queued worker-12 at normal priority",
    "[trace] gc pause 11ms; heap holding near 612MB",
    "[trace] scanned TODO markers across 9 spots",
    "[trace] socket pool grew 8 -> 16",
    "[trace] outline: peek, then patch, then exercise",
    "[trace] snapshot saved at /var/ckpt/step_07",
    "[trace] green exercise batch, 128 cases",
    "[trace] assembled 1.9k tokens, none dropped",
    "[trace] vector flush pushed 32 entries",
    "[trace] pulse ok; lag 3ms; queue empty",
    "[trace] knobs refreshed from /etc/agent/agent.toml",
    "[trace] applied one hunk, 4 lines touched",
    "[trace] governor cleared a call, 880 left",
    "[trace] dns mapped api.internal in 2ms",
    "[trace] fetch returned 200 in 37ms",
    "[trace] mutex held for R-5521",
    "[trace] tier probe: hot working set under cap",
    "[trace] listdir /srv/data yielded 7 dirents",
    "[trace] amended: drop a redundant fetch here",
)


def _build_context(
    total: int,
    *,
    fact_at: int,
    fact_line: str,
    decoys: Tuple[str, ...] = (),
    decoy_at: int = 0,
) -> Tuple[str, ...]:
    """Compose a log of ``total`` lines.

    The ``fact_line`` is placed at index ``fact_at``; any ``decoys`` are placed
    contiguously starting at ``decoy_at`` (skipping the fact slot). All remaining
    slots cycle deterministically through the neutral distractor pool. Positions
    are clamped into range.
    """
    fact_at = max(0, min(total - 1, fact_at))
    placed = {fact_at: fact_line}

    # Place decoys in a contiguous block, hopping over the fact slot.
    j = max(0, min(total - 1, decoy_at))
    for dline in decoys:
        while j < total and (j in placed):
            j += 1
        if j >= total:
            break
        placed[j] = dline
        j += 1

    out: List[str] = []
    d = 0
    for i in range(total):
        if i in placed:
            out.append(placed[i])
        else:
            out.append(_DISTRACTORS[d % len(_DISTRACTORS)])
            d += 1
    return tuple(out)


# Task spec tuple:
#   (task_id, total_lines, fact_fraction, fact_line, question, answer,
#    difficulty, decoys, decoy_fraction)
#
# fact_fraction / decoy_fraction in [0,1] map to line positions (0=start,1=end).
# Decoys are HIGH-overlap distractor lines (share the question's content words,
# never the answer). They let us defeat ``extractive`` on demand: enough decoys
# in one region soak up its budget so a low-overlap fact elsewhere is dropped,
# while ``hierarchical`` still keeps the fact from its own chunk.
_TEMPLATES = (
    # 1. LATE + LOW overlap, decoys early -> TRUNCATE wins (recency). Extractive
    #    spends budget on the early decoys and misses the late, low-overlap fact.
    dict(
        task_id="deploy-port",
        total=70,
        fact_frac=0.97,
        fact_line="[note] the value finally chosen was 8443 once the rollout finished",
        question="What port did the deploy settle on?",
        answer="8443",
        difficulty="hard",
        decoys=(
            "[trace] the deploy port settle check is pending review",
            "[trace] deploy settle port audit scheduled for later",
            "[trace] port deploy settle dashboard refreshed",
        ),
        decoy_frac=0.10,
    ),
    # 2. LATE + LOW overlap, decoys early -> TRUNCATE wins. Extractive misses.
    dict(
        task_id="final-seed",
        total=64,
        fact_frac=0.95,
        fact_line="[note] in the end we locked it to 91247 for reproducibility",
        question="What random seed did the run lock to?",
        answer="91247",
        difficulty="hard",
        decoys=(
            "[trace] random seed run lock policy under discussion",
            "[trace] run random lock seed checklist opened",
            "[trace] seed lock run random review queued",
        ),
        decoy_frac=0.08,
    ),
    # 3. EARLY + HIGH overlap (no decoys) -> EXTRACTIVE wins; truncate drops the
    #    early line. Hierarchical also keeps it (stands out in its chunk).
    dict(
        task_id="db-host",
        total=72,
        fact_frac=0.06,
        fact_line="[config] the database host for the primary database is db-east-3",
        question="What is the database host for the primary database?",
        answer="db-east-3",
        difficulty="medium",
        decoys=(),
        decoy_frac=0.0,
    ),
    # 4. MIDDLE + HIGH overlap (no decoys) -> EXTRACTIVE wins; truncate misses
    #    the middle. Hierarchical keeps it too.
    dict(
        task_id="api-version",
        total=80,
        fact_frac=0.5,
        fact_line="[config] the api version pinned for the api gateway is v7",
        question="What api version is pinned for the api gateway?",
        answer="v7",
        difficulty="medium",
        decoys=(),
        decoy_frac=0.0,
    ),
    # 5. EARLY + HIGH overlap (no decoys) -> EXTRACTIVE wins; truncate misses.
    #    Hierarchical keeps it too.
    dict(
        task_id="owner-team",
        total=60,
        fact_frac=0.08,
        fact_line="[meta] the owner team responsible for the owner service is team-falcon",
        question="Which owner team is responsible for the owner service?",
        answer="team-falcon",
        difficulty="easy",
        decoys=(),
        decoy_frac=0.0,
    ),
    # 6. EARLY + LOW overlap, decoys late -> HIERARCHICAL's niche. Truncate (late
    #    window) misses the early fact; extractive's budget is eaten by the late
    #    high-overlap decoys, so it misses the low-overlap fact. Hierarchical
    #    keeps the fact from its early chunk.
    dict(
        task_id="region-code",
        total=66,
        fact_frac=0.05,
        fact_line="[env] the locale picked here was usw2 for this deployment box",
        question="What region environment code was chosen for the box?",
        answer="usw2",
        difficulty="hard",
        decoys=(
            "[trace] region environment code box survey opened",
            "[trace] box region environment code notes filed",
            "[trace] environment region box code review pending",
            "[trace] code box region environment digest sent",
            "[trace] region box environment code ticket raised",
            "[trace] environment code box region tab pinned",
            "[trace] box code region environment thread bumped",
            "[trace] code region environment box recap posted",
            "[trace] region environment box code memo archived",
        ),
        decoy_frac=0.62,
    ),
    # 7. MIDDLE + LOW overlap, decoys late -> HIERARCHICAL's niche again. Truncate
    #    misses the middle; extractive is soaked up by late decoys. Hierarchical
    #    keeps the fact from its middle chunk.
    dict(
        task_id="retry-limit",
        total=74,
        fact_frac=0.50,
        fact_line="[policy] the ceiling agreed for repeated calls was capped at 5x",
        question="What retry attempt limit was set for repeated requests?",
        answer="5x",
        difficulty="medium",
        decoys=(
            "[trace] retry attempt limit repeated requests memo drafted",
            "[trace] repeated requests retry attempt limit chart updated",
            "[trace] limit retry repeated attempt requests sync held",
            "[trace] requests limit repeated retry attempt log noted",
            "[trace] retry repeated requests attempt limit tab opened",
            "[trace] attempt limit retry requests repeated note filed",
            "[trace] repeated retry limit requests attempt recap sent",
            "[trace] requests retry attempt repeated limit digest cut",
            "[trace] limit repeated requests retry attempt thread bumped",
        ),
        decoy_frac=0.78,
    ),
    # 8. MIDDLE + LOW overlap, decoys early -> HIERARCHICAL's niche. Truncate
    #    misses the middle; extractive eaten by early decoys. Hierarchical keeps
    #    the fact from its middle chunk.
    dict(
        task_id="build-id",
        total=78,
        fact_frac=0.46,
        fact_line="[note] the stamp recorded for this cut was bx-2209 exactly",
        question="What build artifact identifier was recorded for the release?",
        answer="bx-2209",
        difficulty="medium",
        decoys=(
            "[trace] build artifact identifier release tracker opened",
            "[trace] release build artifact identifier board synced",
            "[trace] identifier release artifact build digest filed",
            "[trace] artifact build release identifier tab pinned",
            "[trace] release identifier build artifact note added",
            "[trace] build release artifact identifier recap posted",
            "[trace] identifier build release artifact memo archived",
            "[trace] artifact release build identifier thread bumped",
            "[trace] release build identifier artifact log noted",
        ),
        decoy_frac=0.04,
    ),
    # 9. LATE + HIGH overlap (no decoys) -> several strategies handle it (truncate
    #    by recency, extractive by overlap, hierarchical by chunk). Keeps overall
    #    accuracy realistic rather than a clean partition.
    dict(
        task_id="timeout-secs",
        total=68,
        fact_frac=0.94,
        fact_line="[config] the request timeout for the request handler is 30 seconds",
        question="What is the request timeout for the request handler?",
        answer="30 seconds",
        difficulty="easy",
        decoys=(),
        decoy_frac=0.0,
    ),
)


def make_tasks() -> List[CompactionTask]:
    """Build the fixed suite of compaction-QA tasks (deterministic)."""
    tasks: List[CompactionTask] = []
    for spec in _TEMPLATES:
        total = spec["total"]
        fact_at = int(round(spec["fact_frac"] * (total - 1)))
        decoy_at = int(round(spec["decoy_frac"] * (total - 1)))
        context = _build_context(
            total,
            fact_at=fact_at,
            fact_line=spec["fact_line"],
            decoys=tuple(spec["decoys"]),
            decoy_at=decoy_at,
        )
        task = CompactionTask(
            task_id=spec["task_id"],
            context=context,
            question=spec["question"],
            answer=spec["answer"],
            difficulty=spec["difficulty"],
        )
        # Fail fast if a template lost its (unique) answer-bearing line.
        _ = task.fact_line_index()
        tasks.append(task)
    return tasks


# Module-level fixed suite (8-10 tasks per spec).
TASKS: List[CompactionTask] = make_tasks()
