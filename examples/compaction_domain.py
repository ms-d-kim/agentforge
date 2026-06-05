"""Context-compaction / agent-memory domain: AgentForge selector over compaction.

The arms are three context-compaction strategies (``truncate``, ``extractive``,
``hierarchical``); each shrinks a long agent log to a small budget. A fixed
downstream reader then answers a question using ONLY the compacted context, and
the reward is 1.0 iff the gold answer survived compaction (the downstream-QA
signal). The selector learns online which strategy preserves answers best on
average — here, ``hierarchical`` (the robust all-rounder).

    python -m examples.compaction_domain --fake                 # offline, no API key
    python -m examples.compaction_domain --fake --episodes 60 --policy ucb1
    python -m examples.compaction_domain                        # live via OpenRouter

The reward function itself invokes the reader, so we pass the reader (real or
fake) into the reward through a closure (``make_reward_closure``). This domain is
inspired by *ee392c-agent-mem* (EE 392C, Stanford; Minseok Kim & Kristen
Guernsey), which profiled compaction but had no downstream task to reward.
"""
from __future__ import annotations

import argparse
import random

from src.domains.compaction.reward import make_fake_reader_responder, make_reward_closure
from src.domains.compaction.strategies import make_strategies
from src.domains.compaction.tasks import TASKS
from src.llm import FakeLLMClient, build_client
from src.selector import WorkflowSelector


def compression_ratio(task, compacted_context: str) -> float:
    """orig_len / compacted_len in characters (>= 1.0 means it shrank)."""
    orig = len(task.context_text)
    comp = max(1, len(compacted_context))
    return orig / comp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AgentForge selector over context-compaction strategies."
    )
    ap.add_argument("--fake", action="store_true", help="offline deterministic run (no API key)")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument(
        "--policy",
        default="epsilon-greedy",
        choices=["epsilon-greedy", "ucb1", "thompson"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--persist", default=None, help="state file to learn across runs")
    args = ap.parse_args()

    # The reader is the LLM that answers from the compacted context. Offline, a
    # fake reader returns the gold answer iff the fact survived into the compacted
    # context it is handed (so reward depends purely on what compaction kept).
    if args.fake:
        reader = FakeLLMClient(make_fake_reader_responder(TASKS))
    else:
        reader = build_client("openrouter")

    print(f"model={reader.model}  policy={args.policy}  episodes={args.episodes}  tasks={len(TASKS)}")

    arms = make_strategies()
    sel = WorkflowSelector(
        arms=arms,
        # The arm output IS the compacted context; this closure feeds it + the
        # reader into the downstream-QA reward and returns the float.
        reward=make_reward_closure(reader),
        policy=args.policy,
        seed=args.seed,
        persist=args.persist,
    )

    rng = random.Random(args.seed)
    correct = 0
    ratio_sum = 0.0
    for i in range(args.episodes):
        task = rng.choice(TASKS)
        result = sel.run(task)
        correct += int(result.reward or 0)
        ratio_sum += compression_ratio(task, result.output)
        if (i + 1) % 10 == 0 or i == 0:
            s = sel.stats()
            est = "  ".join(f"{n}:{s[n]['mean']:.2f}({int(s[n]['count'])})" for n in s)
            print(
                f"  ep {i + 1:>3}  picked={result.arm:<12} r={result.reward}  | {est}"
            )

    print(f"\nrolling accuracy: {correct}/{args.episodes} = {correct / args.episodes:.0%}")
    print(f"selector learned to prefer: {sel.best()}")
    print(
        "per-arm stats:",
        {n: f"{v['mean']:.2f} (n={int(v['count'])})" for n, v in sel.stats().items()},
    )
    print(f"average compression ratio (orig/compacted): {ratio_sum / args.episodes:.1f}x")
    if args.persist:
        print(f"state saved to {args.persist} — re-run to keep learning.")


if __name__ == "__main__":
    main()
