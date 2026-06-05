"""Cross-domain × policy compute sweep — the 'does it generalize?' matrix.

Runs the one-step bandit policies (ε-greedy / UCB1 / Thompson) over every new domain
(code, search, compaction) across seeds, all offline with fake LLMs. Produces:
  results/sweep.json          — the {domain: {policy: final_acc ± std}} matrix
  results/sweep_matrix.png     — grouped bars (domains × policies)
  results/sweep_curves.png     — per-domain learning curves (a line per policy)

The contextual (LinUCB) and multi-step (REINFORCE) rungs are reported separately
(see tests/test_policies.py and examples/rl_multihop.py) — this sweep is the breadth
across domains for the average-case bandits.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from src.llm import FakeLLMClient
from src.selector import WorkflowSelector

POLICIES = ["epsilon-greedy", "ucb1", "thompson"]


def build_code():
    from src.domains.code import TASKS, compute_code_reward, make_arms, make_fake_responder

    llm = FakeLLMClient(make_fake_responder())
    return "code", make_arms(llm), (lambda t, o: compute_code_reward(t, o)[0]), list(TASKS)


def build_search():
    from src.domains.search import compute_search_reward, make_arms, make_fake_responder
    from src.domains.search.corpus import Corpus
    from src.domains.search.tasks import get_tasks

    llm = FakeLLMClient(make_fake_responder())
    return "search", make_arms(llm, Corpus()), (lambda t, o: compute_search_reward(t, o)[0]), get_tasks()


def build_compaction():
    from src.domains.compaction.reward import make_fake_reader_responder, make_reward_closure
    from src.domains.compaction.strategies import make_strategies
    from src.domains.compaction.tasks import TASKS

    reader = FakeLLMClient(make_fake_reader_responder(TASKS))
    return "compaction", make_strategies(), make_reward_closure(reader), list(TASKS)


DOMAIN_BUILDERS = [build_code, build_search, build_compaction]


def _rolling(seq, w=10):
    return [sum(seq[max(0, i - w + 1): i + 1]) / len(seq[max(0, i - w + 1): i + 1]) for i in range(len(seq))]


def _run(arms, reward, tasks, policy, seed, episodes):
    sel = WorkflowSelector(arms, reward=reward, policy=policy, seed=seed)
    rng = random.Random(1000 + seed)
    return [sel.run(rng.choice(tasks)).reward or 0.0 for _ in range(episodes)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-domain policy sweep (offline).")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    matrix: dict = {}
    curves: dict = {}
    for build in DOMAIN_BUILDERS:
        name, arms, reward, tasks = build()
        matrix[name], curves[name] = {}, {}
        for pol in POLICIES:
            finals, curve_runs = [], []
            for s in seeds:
                rewards = _run(arms, reward, tasks, pol, s, args.episodes)
                finals.append(sum(rewards[-10:]) / 10)
                curve_runs.append(_rolling(rewards))
            n = min(len(c) for c in curve_runs)
            curves[name][pol] = [statistics.mean(c[i] for c in curve_runs) for i in range(n)]
            matrix[name][pol] = {
                "final_mean": statistics.mean(finals),
                "final_std": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
            }
        print(f"{name:<12} " + "  ".join(
            f"{p}={matrix[name][p]['final_mean']:.2f}±{matrix[name][p]['final_std']:.2f}" for p in POLICIES))

    results = Path(__file__).resolve().parent.parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "sweep.json").write_text(json.dumps(
        {"episodes": args.episodes, "seeds": seeds, "policies": POLICIES, "matrix": matrix}, indent=2))

    _plot_matrix(matrix, results / "sweep_matrix.png")
    _plot_curves(curves, results / "sweep_curves.png", args.episodes)
    print(f"\nwrote {results}/sweep.json, sweep_matrix.png, sweep_curves.png")


def _plot_matrix(matrix, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = list(matrix)
    x = range(len(domains))
    width = 0.25
    plt.figure(figsize=(9, 5))
    for j, pol in enumerate(POLICIES):
        means = [matrix[d][pol]["final_mean"] for d in domains]
        stds = [matrix[d][pol]["final_std"] for d in domains]
        plt.bar([xi + (j - 1) * width for xi in x], means, width, yerr=stds, capsize=3, label=pol)
    plt.xticks(list(x), domains)
    plt.ylabel("final rolling accuracy (last 10 eps)")
    plt.ylim(0, 1)
    plt.title("AgentForge selector across domains × policies (offline, 3 seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def _plot_curves(curves, out, episodes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = list(curves)
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4), sharey=True)
    if len(domains) == 1:
        axes = [axes]
    for ax, d in zip(axes, domains):
        for pol in POLICIES:
            ax.plot(range(1, len(curves[d][pol]) + 1), curves[d][pol], label=pol)
        ax.set_title(d)
        ax.set_xlabel("episode")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    plt.suptitle("Per-domain learning curves (bandit learns the best arm everywhere)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
