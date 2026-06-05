"""Agentic search/grep domain: AgentForge's selector over three retrieval strategies.

The task is to *locate* a target (a class/def) in a code corpus — here the
AgentForge ``src/`` tree itself. The three "arms" are different grep strategies
(simple top-hit, iterative LLM-disambiguated, broad-then-narrow-to-definition),
and the reward is whether the retrieved file matches the gold file. The selector
learns online which strategy wins on these tasks — no SQL, just callables that
issue real ``grep`` calls + a reward signal.

    python -m examples.search_domain --fake               # offline, no API key (deterministic)
    python -m examples.search_domain                       # live via OpenRouter (.env key)
    python -m examples.search_domain --policy ucb1 --episodes 40

Under ``--fake`` the arms differentiate by construction (see
src/domains/search/workflows.py): ``simple`` lands the easy "named symbol"
questions, ``iterative`` additionally resolves conceptual ones via its follow-up
"which file?" step, and ``broad_then_narrow`` is strongest on definition-style
questions. The selector should concentrate on ``iterative``.

This domain is the seed for a later RL phase — the arms' inner
"grep → observe → re-grep → answer" loop is a hand-coded rollout of a search MDP
(state = history + observations, action = next query/answer). The selector stays a
one-step bandit-over-arms here; see the design note in workflows.py.
"""
from __future__ import annotations

import argparse
import random

from src.domains.search.corpus import Corpus
from src.domains.search.tasks import get_tasks
from src.domains.search.workflows import (
    compute_search_reward,
    make_arms,
    make_fake_responder,
)
from src.llm import FakeLLMClient, build_client
from src.selector import WorkflowSelector


def reward(task, retrieved_file) -> float:
    """Scalar reward for the selector (drops the diagnostic message)."""
    return compute_search_reward(task, retrieved_file)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="AgentForge selector on an agentic-search task.")
    ap.add_argument("--fake", action="store_true", help="offline deterministic run (no API key)")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--policy", default="epsilon-greedy", choices=["epsilon-greedy", "ucb1"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm = FakeLLMClient(make_fake_responder()) if args.fake else build_client("openrouter")
    corpus = Corpus()
    tasks = get_tasks()
    print(
        "model=%s  policy=%s  episodes=%d  corpus=%s (%d files)"
        % (llm.model, args.policy, args.episodes, corpus.root.name, len(corpus.files()))
    )

    sel = WorkflowSelector(
        arms=make_arms(llm, corpus),
        reward=reward,
        policy=args.policy,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    hits = 0  # exact gold-file matches (reward == 1.0)
    cumulative = 0.0
    for i in range(args.episodes):
        task = rng.choice(tasks)
        result = sel.run(task)
        cumulative += result.reward or 0.0
        hits += int((result.reward or 0.0) == 1.0)
        if (i + 1) % 8 == 0 or i == 0:
            s = sel.stats()
            est = "  ".join("%s:%.2f(%d)" % (n, s[n]["mean"], s[n]["count"]) for n in s)
            acc = hits / (i + 1)
            print(
                "  ep %3d  picked=%-17s r=%.1f  found=%-26s | rolling_acc=%.0f%%  | %s"
                % (i + 1, result.arm, result.reward or 0.0, result.output or "(none)", acc * 100, est)
            )

    print(
        "\nrolling accuracy (exact file match): %d/%d = %.0f%%"
        % (hits, args.episodes, hits / args.episodes * 100)
    )
    print("mean reward (1.0=file, 0.5=same dir): %.2f" % (cumulative / args.episodes))
    print("selector learned to prefer: %s" % sel.best())
    print(
        "per-arm stats:",
        {n: "%.2f (n=%d)" % (v["mean"], v["count"]) for n, v in sel.stats().items()},
    )


if __name__ == "__main__":
    main()
