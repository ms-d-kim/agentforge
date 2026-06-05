"""Code-generation example: AgentForge selector over Python-codegen strategies.

A second domain for the domain-agnostic ``WorkflowSelector`` (the first is the
BIRD text-to-SQL experiment in ``src/``). Here the "arms" are three code-writing
strategies and the reward is "do the unit tests pass?" — a cheap automatic 0/1
signal. The selector learns online which strategy wins on these problems and
concentrates pulls there.

    python -m examples.code_domain --fake                 # offline, no API key
    python -m examples.code_domain --fake --episodes 60 --policy ucb1
    python -m examples.code_domain                        # live via OpenRouter (.env key)

Under ``--fake`` the arms are engineered to differentiate (see
``src/domains/code``): ``direct`` solves only the easy tasks, ``decompose`` solves
easy + medium, and ``iterative`` repairs its way to all of them — so the selector
should learn to prefer ``iterative``.
"""
from __future__ import annotations

import argparse
import random

from src.domains.code import (
    TASKS,
    compute_code_reward,
    make_arms,
    make_fake_responder,
)
from src.llm import FakeLLMClient, build_client
from src.selector import WorkflowSelector


def reward(task, code) -> float:
    """0/1 reward: 1.0 iff the candidate code passes all of the task's tests."""
    score, _err = compute_code_reward(task, code)
    return score


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AgentForge selector on a Python code-generation task."
    )
    ap.add_argument(
        "--fake", action="store_true", help="offline deterministic run (no API key)"
    )
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument(
        "--policy",
        default="epsilon-greedy",
        choices=["epsilon-greedy", "ucb1", "thompson"],
    )
    ap.add_argument("--persist", default=None, help="state file to learn across runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm = FakeLLMClient(make_fake_responder()) if args.fake else build_client("openrouter")
    print(
        "model={}  policy={}  episodes={}".format(
            llm.model, args.policy, args.episodes
        )
    )

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
        if (i + 1) % 8 == 0 or i == 0:
            s = sel.stats()
            est = "  ".join(
                "{}:{:.2f}({})".format(n, s[n]["mean"], int(s[n]["count"])) for n in s
            )
            rolling = correct / (i + 1)
            print(
                "  ep {:>3}  picked={:<10} r={}  acc={:.0%}  | {}".format(
                    i + 1, result.arm, result.reward, rolling, est
                )
            )

    print(
        "\nrolling accuracy: {}/{} = {:.0%}".format(
            correct, args.episodes, correct / args.episodes
        )
    )
    print("selector learned to prefer: {}".format(sel.best()))
    print(
        "stats:",
        {
            n: "{:.2f} (n={})".format(v["mean"], int(v["count"]))
            for n, v in sel.stats().items()
        },
    )
    if args.persist:
        print("state saved to {} — re-run to keep learning.".format(args.persist))


if __name__ == "__main__":
    main()
