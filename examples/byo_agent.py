"""Bring-your-own-agent example: AgentForge on a NON-SQL task.

Demonstrates that the WorkflowSelector is domain-agnostic. Here the "arms" are two
*prompting strategies* for multi-step word problems, and the reward is exact-match
on the final number. The selector learns online which strategy wins — no SQL, no
BIRD, just your own callables + a reward signal.

    python -m examples.byo_agent --fake          # offline, no API key (deterministic)
    python -m examples.byo_agent                 # live via OpenRouter (.env key)
    python -m examples.byo_agent --policy ucb1 --persist examples/byo_state.json

This is the same kernel that runs the BIRD experiment — the point is that *your*
agent plugs in the same way.
"""
from __future__ import annotations

import argparse
import random
import re

from src.llm import FakeLLMClient, build_client
from src.selector import WorkflowSelector

# Multi-step word problems (integer answers) — terse "direct" answering tends to
# slip on these, while "think step by step" tends to land them. Real differentiation.
TASKS = [
    {"q": "A bakery sells muffins in boxes of 6. Monday they sold 14 boxes, Tuesday 9 boxes. Each muffin costs $3. How much money did they make over the two days?", "answer": 414},
    {"q": "A train travels 60 mph for 2.5 hours, then 40 mph for 1.5 hours. How many miles total?", "answer": 210},
    {"q": "Sarah has 3 times as many apples as Tom. Tom has 8 apples. They combine and share equally among 4 people. How many apples each?", "answer": 8},
    {"q": "A book has 240 pages. Jim reads 30 pages/day for 4 days, then 20 pages/day. How many days total to finish?", "answer": 10},
    {"q": "A store had 500 items. They sold 35% on day one and 120 items on day two. How many items are left?", "answer": 205},
    {"q": "Each gardener plants 12 trees/day. 5 gardeners work 3 days. If 18 trees died, how many living trees remain?", "answer": 162},
    {"q": "A pool holds 1200 liters. A pipe fills at 50 L/min but it leaks 10 L/min. How many minutes to fill?", "answer": 30},
    {"q": "Maria earns $15/hour, works 6 hours/day for 5 days, then gets a $50 bonus. How much total?", "answer": 500},
]

DIRECT_PROMPT = "Answer with ONLY the final number, no words, no reasoning.\n\n{q}"
COT_PROMPT = "Solve step by step. On the last line write 'ANSWER: <number>'.\n\n{q}"


def extract_int(text: str):
    nums = re.findall(r"-?\d+", (text or "").replace(",", ""))
    return int(nums[-1]) if nums else None


def make_arms(llm):
    return {
        "direct": lambda task: llm.complete(DIRECT_PROMPT.format(q=task["q"])),
        "cot": lambda task: llm.complete(COT_PROMPT.format(q=task["q"])),
    }


def reward(task, output) -> float:
    return 1.0 if extract_int(output) == task["answer"] else 0.0


def make_fake_responder():
    """Offline stand-in: chain-of-thought solves; terse 'direct' misses these."""
    qa = {t["q"]: t["answer"] for t in TASKS}

    def responder(prompt, system):
        ans = next((a for q, a in qa.items() if q in prompt), None)
        if ans is None:
            return "0"
        return f"... ANSWER: {ans}" if "step by step" in prompt else "0"

    return responder


def main() -> None:
    ap = argparse.ArgumentParser(description="AgentForge selector on a non-SQL task.")
    ap.add_argument("--fake", action="store_true", help="offline deterministic run (no API key)")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--policy", default="epsilon-greedy", choices=["epsilon-greedy", "ucb1"])
    ap.add_argument("--persist", default=None, help="state file to learn across runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm = FakeLLMClient(make_fake_responder()) if args.fake else build_client("openrouter")
    print(f"model={llm.model}  policy={args.policy}  episodes={args.episodes}")

    sel = WorkflowSelector(
        arms=make_arms(llm),
        reward=reward,
        policy=args.policy,
        persist=args.persist,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    correct = 0
    for i in range(args.episodes):
        task = rng.choice(TASKS)
        result = sel.run(task)
        correct += int(result.reward or 0)
        if (i + 1) % 10 == 0 or i == 0:
            s = sel.stats()
            est = "  ".join(f"{n}:{s[n]['mean']:.2f}({s[n]['count']})" for n in s)
            print(f"  ep {i+1:>3}  picked={result.arm:<7} r={result.reward}  | {est}")

    print(f"\nrolling accuracy: {correct}/{args.episodes} = {correct/args.episodes:.0%}")
    print(f"selector learned to prefer: {sel.best()}")
    print("stats:", {n: f"{v['mean']:.2f} (n={v['count']})" for n, v in sel.stats().items()})
    if args.persist:
        print(f"state saved to {args.persist} — re-run to keep learning.")


if __name__ == "__main__":
    main()
