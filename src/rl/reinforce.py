"""REINFORCE (Monte-Carlo policy gradient) over the multi-step search environment.

A linear-softmax policy ``π(a|s) = softmax(θ·f(s))`` is trained from terminal reward
via the policy-gradient theorem with reward-to-go and a running-mean baseline for
variance reduction. Small (4 actions × 7 features = 28 params), so it converges in
seconds with no API calls — the point is to demonstrate credit assignment over a
trajectory, the thing a one-step bandit cannot do.
"""
from __future__ import annotations

import random
from typing import List, Optional

from .environment import (
    ANSWER,
    FOLLOW,
    MultiHopEnv,
    N_ACTIONS,
    N_FEATURES,
    get_env_tasks,
)


class REINFORCE:
    def __init__(self, n_actions: int = N_ACTIONS, n_features: int = N_FEATURES,
                 lr: float = 0.2, gamma: float = 1.0, seed: int = 0):
        import numpy as np

        self.np = np
        self.theta = np.zeros((n_actions, n_features))
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.rng = random.Random(seed)
        self.baseline = 0.0
        self._b_n = 0

    def policy(self, features) -> "list":
        np = self.np
        z = self.theta @ np.asarray(features, dtype=float)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, features) -> int:
        p = self.policy(features)
        r = self.rng.random()
        c = 0.0
        for a in range(self.n_actions):
            c += float(p[a])
            if r <= c:
                return a
        return self.n_actions - 1

    def greedy(self, features) -> int:
        p = self.policy(features)
        return int(max(range(self.n_actions), key=lambda a: p[a]))

    def update(self, trajectory) -> None:
        """``trajectory`` = list of (features, action, reward) for one episode."""
        np = self.np
        # Reward-to-go returns.
        returns, g = [], 0.0
        for _, _, r in reversed(trajectory):
            g = r + self.gamma * g
            returns.append(g)
        returns.reverse()

        episode_return = returns[0] if returns else 0.0
        self._b_n += 1
        self.baseline += (episode_return - self.baseline) / self._b_n

        for (features, action, _), g_t in zip(trajectory, returns):
            f = np.asarray(features, dtype=float)
            p = self.policy(features)
            advantage = g_t - self.baseline
            for k in range(self.n_actions):
                indicator = 1.0 if k == action else 0.0
                self.theta[k] += self.lr * advantage * f * (indicator - p[k])


def run_episode(env: MultiHopEnv, agent: REINFORCE, task, greedy: bool = False):
    features = env.reset(task)
    trajectory, total, success, steps = [], 0.0, 0, 0
    done = False
    while not done:
        action = agent.greedy(features) if greedy else agent.act(features)
        nxt, reward, done, info = env.step(action)
        trajectory.append((features, action, reward))
        total += reward
        steps += 1
        if action == ANSWER and info.get("correct"):
            success = 1
        features = nxt
    return trajectory, total, success, steps


def train(episodes: int = 4000, seed: int = 0, lr: float = 0.2, step_cost: float = 0.05,
          max_hops: int = 5, signal_noise: float = 0.1):
    """Train REINFORCE on the multi-hop env. Returns (agent, history) where history
    is a list of (success, steps) per training episode."""
    env = MultiHopEnv(max_hops=max_hops, step_cost=step_cost, signal_noise=signal_noise, seed=seed)
    agent = REINFORCE(lr=lr, seed=seed)
    tasks = get_env_tasks()
    rng = random.Random(seed + 1)
    history = []
    for _ in range(episodes):
        task = rng.choice(tasks)
        trajectory, _total, success, steps = run_episode(env, agent, task)
        agent.update(trajectory)
        history.append((success, steps))
    return agent, history


def evaluate(agent: REINFORCE, episodes: int = 300, seed: int = 99, step_cost: float = 0.05,
             max_hops: int = 5, signal_noise: float = 0.1, greedy: bool = True):
    """Average success rate + steps of a (greedy) policy over fresh episodes."""
    env = MultiHopEnv(max_hops=max_hops, step_cost=step_cost, signal_noise=signal_noise, seed=seed)
    tasks = get_env_tasks()
    rng = random.Random(seed)
    succ, total_steps = 0, 0
    for _ in range(episodes):
        task = rng.choice(tasks)
        _t, _r, success, steps = run_episode(env, agent, task, greedy=greedy)
        succ += success
        total_steps += steps
    return succ / episodes, total_steps / episodes


def fixed_horizon_baseline(k: int, episodes: int = 300, seed: int = 99, step_cost: float = 0.05,
                           max_hops: int = 5, signal_noise: float = 0.1):
    """Non-learning baseline: FOLLOW exactly k times, then ANSWER. Caps at the
    fraction of tasks whose depth == k."""
    env = MultiHopEnv(max_hops=max_hops, step_cost=step_cost, signal_noise=signal_noise, seed=seed)
    tasks = get_env_tasks()
    rng = random.Random(seed)
    succ = 0
    for _ in range(episodes):
        task = rng.choice(tasks)
        env.reset(task)
        for _ in range(k):
            env.step(FOLLOW)
        _f, _r, _d, info = env.step(ANSWER)
        succ += int(info.get("correct", False))
    return succ / episodes


def random_baseline(episodes: int = 300, seed: int = 7, step_cost: float = 0.05,
                    max_hops: int = 5, signal_noise: float = 0.1):
    env = MultiHopEnv(max_hops=max_hops, step_cost=step_cost, signal_noise=signal_noise, seed=seed)
    agent = REINFORCE(seed=seed)  # untrained → uniform policy
    tasks = get_env_tasks()
    rng = random.Random(seed)
    succ = 0
    for _ in range(episodes):
        _t, _r, success, _s = run_episode(env, agent, rng.choice(tasks), greedy=False)
        succ += success
    return succ / episodes
