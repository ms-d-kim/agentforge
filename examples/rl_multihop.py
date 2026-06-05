"""Multi-step RL — the top rung of the ladder (the 'iterate-or-stop' control problem).

Trains REINFORCE on the multi-hop environment and compares it to fixed-horizon and
random baselines. The learned, state-conditioned policy adapts the number of hops per
task and beats any fixed horizon — the thing a one-step bandit cannot do.

    python -m examples.rl_multihop                 # train + compare (offline, no API key)
    python -m examples.rl_multihop --plot          # also save results/rl_multihop.png
"""
from __future__ import annotations

import argparse

from src.rl.reinforce import evaluate, fixed_horizon_baseline, random_baseline, train


def _rolling(hist, w=200):
    out = []
    for i in range(len(hist)):
        lo = max(0, i - w + 1)
        chunk = hist[lo : i + 1]
        out.append(sum(s for s, _ in chunk) / len(chunk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-step RL on the iterate-or-stop problem.")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    agent, hist = train(episodes=args.episodes, seed=args.seed)
    sr, steps = evaluate(agent, episodes=500, greedy=True)
    fixed = {k: fixed_horizon_baseline(k, episodes=500) for k in range(4)}
    best_fixed = max(fixed.values())
    rnd = random_baseline(episodes=500)

    print(f"\nMulti-hop control — {args.episodes} training episodes, seed {args.seed}")
    print(f"  REINFORCE (learned, state-conditioned): {sr:.2f}   (avg {steps:.1f} hops/task)")
    print(f"  best fixed-horizon baseline           : {best_fixed:.2f}   "
          f"(answer-after-k: {{{', '.join(f'{k}:{v:.2f}' for k, v in fixed.items())}}})")
    print(f"  random policy                         : {rnd:.2f}")
    if best_fixed > 0:
        print(f"  => RL beats best fixed-horizon by {sr - best_fixed:+.2f} ({sr / best_fixed:.1f}x)")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path

        from src.config import PLOTS_DIR

        results = Path(PLOTS_DIR).parent / "results"
        results.mkdir(parents=True, exist_ok=True)
        out = results / "rl_multihop.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        roll = _rolling(hist)
        ax1.plot(range(1, len(roll) + 1), roll, label="REINFORCE (learned)")
        ax1.axhline(best_fixed, ls="--", color="gray", label=f"best fixed horizon ({best_fixed:.2f})")
        ax1.axhline(rnd, ls=":", color="lightgray", label=f"random ({rnd:.2f})")
        ax1.set_xlabel("training episode")
        ax1.set_ylabel("rolling success rate")
        ax1.set_ylim(0, 1)
        ax1.set_title("Multi-step RL learns to stop at the right hop")
        ax1.legend(fontsize=8)

        labels = ["REINFORCE", "best fixed", "random"]
        vals = [sr, best_fixed, rnd]
        ax2.bar(labels, vals, color=["#1f77b4", "#888888", "#cccccc"])
        for i, v in enumerate(vals):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("success rate")
        ax2.set_title("Learned policy vs. fixed-horizon baselines")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"\nplot saved: {out}")


if __name__ == "__main__":
    main()
