# AgentForge — Submission Index

> One place for every artifact. **AgentForge** is a self-improving agent SDK: an online bandit (→
> contextual → RL) that learns which multi-step workflow to run per task from your own reward signal.
> CS 153 (Frontier Systems), Spring 2026 · solo · Track: Automation / Agent Systems.

## The artifacts

| Artifact | Where |
|---|---|
| **Public code repo** | https://github.com/ms-d-kim/agentforge |
| **Demo video** (≈6 min, narrated, cloned voice) | [`video/agentforge_demo.mp4`](../video/agentforge_demo.mp4) — committed in-repo |
| **README** (overview · who-it's-for · architecture · how-it-compares · results) | [`README.md`](../README.md) |
| **Architecture diagram** | [`results/architecture.png`](../results/architecture.png) |
| **Results figures** (BIRD curve, cross-domain sweep, multi-step RL) | [`results/`](../results/) |
| **Per-arm summary + Wilson CIs** | [`results/exp_summary.txt`](../results/exp_summary.txt) |
| **Competitive positioning** (vs ADK/OpenAI/MS/LangGraph/CrewAI/DSPy/routers/ARC) | [`docs/POSITIONING.md`](POSITIONING.md) |
| **Pip-installable SDK** | `pip install -e .` → `from agentforge import WorkflowSelector` |

> The video is committed here so the repo is self-contained. For the course portal, also upload
> `video/agentforge_demo.mp4` to Gradescope / the project form (or an unlisted YouTube link).

## Reproduce in 60 seconds (offline — no API key, no download)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -p 'test_*.py'   # 67 tests, all offline
python -m examples.code_domain --fake                  # selector learns the best arm
python -m examples.rl_multihop --plot                  # multi-step RL beats fixed horizons
python -m scripts.sweep                                # cross-domain × policy matrix
```

Real BIRD-SQL run (needs an OpenRouter key + the BIRD dev download):
`python -m scripts.download_bird` → add `OPENROUTER_API_KEY` to `.env` →
`python -m scripts.run_real` → `python -m scripts.plot_results --prefix exp`.

## Headline results (all reproducible from this repo)

- **BIRD-SQL** (gpt-4o-mini, 3 seeds × 60 ep): bandit **22.3** vs random **20.3** vs best single arm
  **24.0** vs oracle **27.0**; learns online to down-weight the weak `decompose` arm.
- **Domain-agnostic**: same selector learns the best arm across **code** (tests pass), **search**
  (file-match), **compaction** (downstream-QA).
- **Policy ladder**: contextual LinUCB **1.00 vs ~0.50** for average-case on context-dependent tasks;
  multi-step REINFORCE **0.81 vs 0.35** best fixed-horizon.

## Rubric map (15 pts)

- **Problem & Insight** — README intro + "Who it's for" (painpoint/ICP/wedge) + POSITIONING white-space.
- **Execution & Technical Work** — the SDK + 4 domains + policy ladder + RL; 67 offline tests; real BIRD run.
- **Evaluation & Evidence** — 4 figures + Wilson CIs + reproducible results from the committed logs.
- **Communication & Presentation** — narrated video + README that renders the figures.
- **Process, Integrity & Disclosure** — AI-use disclosure (README), prior-work credit
  ([ee392c-agent-mem](https://github.com/ms-d-kim/ee392c-agent-mem)), public commit history, and the
  honest-caveats below.

## Honest caveats (stated plainly, not hidden)

- The BIRD bandit-vs-random win is a **3-seed mean and loses in seed 0**; the two top arms are
  statistically tied (overlapping Wilson CIs). The durable result is online avoidance of the weak arm.
- The offline `--fake` demos are **controlled demonstrations** (scripted fakes; rewards computed from
  real outputs) — they show mechanics, not that the SDK discovers strategy quality in arbitrary domains.
- The contextual + RL results are **synthetic stand-ins**; the code executor is a **toy harness, not a
  security sandbox**. The "white-space" positioning is scoped "to our knowledge (June 2026)."

