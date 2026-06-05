# Codex Adversarial Audit

One-line verdict: **the empirical demo mostly reproduces, but the repo overstates safety and the strength/generalizability of several claims. Honesty grade: B-.**

The strongest claims are backed by runnable code and committed logs. The weakest part is not the bandit math; it is the gap between "offline fake demo" and "real agent SDK", plus a concrete security failure in the Python code executor.

## Claim Verdict Table

| Claim | Verdict | Reproduced numbers / observation | Confidence | Severity |
|---|---|---:|---|---|
| Bandit learns the best arm in code/search/compaction and beats random | **QUALIFIED** | 10-seed fake-domain check, 24 episodes: code bandit 0.875 vs random 0.604, best=iterative 10/10; search 0.852 vs 0.750, best=iterative 9/10; compaction 0.596 vs 0.496, best=hierarchical 8/10 but loses to random on 2/10 seeds. | High | Medium |
| Contextual LinUCB reaches ~1.0 where average-case bandits stay ~0.5 | **HOLDS** | 10 seeds, last-100 accuracy: LinUCB with type features 1.000; constant features 0.497; noise features 0.489; Thompson 0.503. | High | Low |
| REINFORCE multi-hop gets ~0.81 vs ~0.35 fixed vs ~0.19 random | **HOLDS** | Example seed 0: 0.81 / 0.35 / 0.19. 10-seed check: RL mean 0.811, fixed mean 0.360, random mean 0.150. Zeroing the terminal signal drops learned policy to 0.340. | High | Low |
| Real BIRD logs show bandit beats random, downweights decompose, and direct/schema are tied | **QUALIFIED** | Raw logs: bandit [21,23,23] mean 22.33; random [23,18,20] mean 20.33; arm0 21; arm1 24; arm2 17; oracle 27. Per-arm rates direct 0.391, schema 0.406, decompose 0.208. 60 sampled SQL reward re-executions had 0 mismatches. Bandit loses to random in seed 0. | High | Medium |
| Competitive positioning: online + multi-step-workflow + persistent + SDK whitespace | **QUALIFIED** | Spot-check supports the narrow distinction from ARC/PILOT/orchestration SDKs, but the "none" framing is an unprovable market negative and omits newer online model-router work such as ParetoBandit. | Medium | Medium |

## Commands Run

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
# Ran 64 tests in 5.854s - OK

.venv/bin/python -W error -m unittest discover -s tests -p 'test_*.py'
# Ran 64 tests in 6.537s - OK

.venv/bin/python -m examples.code_domain --fake
# rolling accuracy: 21/24 = 88%; learned iterative

.venv/bin/python -m examples.search_domain --fake
# rolling exact-file accuracy: 20/24 = 83%; learned iterative

.venv/bin/python -m examples.compaction_domain --fake
# rolling accuracy: 18/24 = 75%; learned hierarchical

.venv/bin/python -m examples.byo_agent --fake
# rolling accuracy: 28/30 = 93%; learned cot

.venv/bin/python -m examples.rl_multihop
# REINFORCE 0.81; best fixed 0.35; random 0.19

.venv/bin/python -m scripts.sweep
# code epsilon-greedy=0.93+/-0.05 ucb1=0.83+/-0.05 thompson=1.00+/-0.00
# search epsilon-greedy=0.87+/-0.05 ucb1=0.87+/-0.06 thompson=0.85+/-0.07
# compaction epsilon-greedy=0.63+/-0.25 ucb1=0.53+/-0.21 thompson=0.67+/-0.21

.venv/bin/python -m pip install -e .
# Successfully installed agentforge-0.1.0

cd /tmp && /Users/minseok/dev/cs153/.venv/bin/python -c "from agentforge import WorkflowSelector; print(WorkflowSelector)"
# <class 'src.selector.WorkflowSelector'>
```

Note: `python -m scripts.sweep` rewrites `results/sweep.json`, `results/sweep_matrix.png`, and `results/sweep_curves.png`; after the run, `git status --short --untracked-files=all` was clean.

## Per-Claim Notes

### 1. Fake Code/Search/Compaction Domains

The offline examples run as advertised, but they are designed demonstrations, not independent evidence that real models naturally exhibit those arm rankings.

Reproduction script used same task streams by seed and compared epsilon-greedy against uniform random arm selection:

```text
code mean bandit=0.875 random=0.604 diff=+0.271; best counts {'iterative': 10}
search mean bandit=0.852 random=0.750 diff=+0.102; best counts {'broad_then_narrow': 1, 'iterative': 9}
compaction mean bandit=0.596 random=0.496 diff=+0.100; best counts {'extractive': 2, 'hierarchical': 8}
```

The important caveat is leakage through the **fake LLM**, not through reward. In code generation, the fake repair step returns the task reference implementation directly (`src/domains/code/__init__.py:197-203`). In search, the fake `choose_file` step returns a gold file path from a hand-built table (`src/domains/search/workflows.py:322-389`). Rewards are computed from actual arm outputs, which is fair for a deterministic toy. But docs should not call this "no gold leakage" without qualifying "the fake LLM has scripted access to reference/gold answers by design."

### 2. Contextual LinUCB

The contextual result holds and the ablation is clean:

```text
linucb_good  mean=1.000
linucb_const mean=0.497
linucb_noise mean=0.489
thompson     mean=0.503
```

`tests/test_policies.py:19-27` defines a synthetic type-conditioned task where the feature is exactly the task type. That is a legitimate pre-action observable and the ablation confirms the win collapses when features are uninformative. The claim should be framed as a toy contextual-bandit proof, not a real workload result.

### 3. REINFORCE Multi-Hop

The result holds across seeds:

```text
default per_seed sr/fixed/random/steps:
0.81/0.35/0.19/3.0, 0.82/0.36/0.15/3.0, 0.82/0.35/0.16/2.9,
0.81/0.35/0.14/3.0, 0.82/0.38/0.12/3.1, 0.82/0.41/0.13/3.0,
0.82/0.35/0.16/3.1, 0.78/0.34/0.16/3.0, 0.83/0.37/0.16/3.0,
0.79/0.34/0.13/3.0
default means sr=0.811 fixed=0.360 random=0.150
no_signal mean=0.340
```

The environment exposes a noisy terminal signal plus position features (`src/rl/environment.py:51-60`). Zeroing the terminal signal drops performance to fixed-horizon range, so the agent is using the intended signal rather than a hidden depth leak. The fixed-horizon cap is fair for the evaluated family because tasks are uniformly spread over depths 1, 2, and 3 (`src/rl/environment.py:81-83`).

### 4. Real BIRD Logs

Raw-log recomputation:

```text
bandit totals [21, 23, 23], mean 22.333333333333332
random totals [23, 18, 20], mean 20.333333333333332
arm0 totals [21, 21, 21], mean 21
arm1 totals [24, 24, 24], mean 24
arm2 totals [17, 17, 17], mean 17
oracle totals [27, 27, 27], mean 27
per-arm: direct 36/92 = 0.391; schema_explore 26/64 = 0.406; decompose 5/24 = 0.208
```

Reward integrity check:

```text
sampled 60 logged SQL pairs; mismatches 0
```

This supports the README numbers (`README.md:212-221`). The caveat is material: the bandit beats random only in mean and only in 2/3 seeds. Seed 0 loses to random, 21 vs 23. The honest claim is "mean improvement and avoidance of the weak decompose arm on this logged pool," not a statistically established win over random.

### 5. Positioning

The narrow distinction is plausible:

- PILOT frames LLM routing as online contextual bandits, but routes models rather than multi-step workflows: https://arxiv.org/abs/2508.21141
- ARC frames query-wise agent configuration/workflows, but appears as research/offline RL rather than a shipped online SDK: https://arxiv.org/abs/2602.11574
- LangSmith supports online evaluation/feedback, but I did not find evidence that it automatically mutates a persistent routing policy: https://docs.langchain.com/langsmith/evaluation-concepts
- OpenAI Agents SDK materials center eval/improvement loops and HITL controls, not an online bandit over workflows: https://openai.github.io/openai-agents-python/human_in_the_loop/

But `docs/POSITIONING.md:16-28` states an empty four-way intersection as fact. That is too strong without a maintained source matrix. Recent online model routers such as ParetoBandit weaken the broad "model routers are mostly offline" row even though they still do not refute the narrower workflow-selection claim: https://arxiv.org/abs/2604.00136

## Ranked Bugs / Issues

### P0 - Python code executor is not sandboxed

File: `src/domains/code/executor.py:31-44`, `src/domains/code/executor.py:60-102`

The executor claims to reject filesystem/network/process access via a static deny-list and then runs model code in a subprocess. The deny-list is trivially bypassed:

```bash
.venv/bin/python - <<'PY'
from src.domains.code.executor import execute_python
payload = """
def leak_ok():
    f = getattr(__builtins__, 'op' + 'en')('/etc/hosts')
    data = f.read(20)
    f.close()
    return isinstance(data, str) and len(data) > 0
"""
print(execute_python(payload, 'assert leak_ok() == True', timeout_s=5))
PY
```

Output:

```text
(True, 'ok')
```

Suggested fix: do not describe this as sandboxed. Run generated code in a real OS sandbox/container with a locked-down filesystem, no network, resource limits, and a reduced Python environment; or keep it as a toy executor and label it unsafe.

### P1 - Verification doc overstates "5/5 HOLD" and misses a security failure

File: `docs/VERIFICATION.md:11-20`, `docs/VERIFICATION.md:80-83`

The empirical numbers mostly reproduce, but the report is too self-serving. It does not mention the code-executor escape, and it says no gold leakage was found even though the fake code/search LLMs intentionally return reference implementations / gold file paths on specific prompt steps. That can be acceptable for offline tests, but the report should say so plainly.

Suggested fix: change the report to `HOLDS / QUALIFIED`, add the executor bypass, and distinguish "reward not hardcoded" from "fake LLM has scripted gold/reference access."

### P1 - Fake-domain demos are engineered, not independent evidence

Files: `src/domains/code/__init__.py:170-220`, `src/domains/search/workflows.py:322-401`, `src/domains/compaction/tasks.py:1-35`

The fake tasks are useful and deterministic, but the best arm is mostly constructed by fixture design: direct succeeds on easy, decompose on easy/medium, iterative repair returns reference implementations; search fake returns file choices from a table; compaction task templates are hand-shaped around strategy niches.

Suggested fix: keep these demos, but label them as controlled demonstrations. Do not use them as evidence that the SDK will discover strategy quality in arbitrary real domains without a real reward distribution.

### P1 - BIRD bandit-vs-random result is weak

Files: `README.md:212-221`, `results/exp_summary.txt`

The README is mostly honest here, but the headline "above random" is a three-seed mean. Seed-level totals are bandit `[21,23,23]` and random `[23,18,20]`, so the bandit loses seed 0. With a +2 mean margin and 3 seeds, this is not robust evidence of beating random.

Suggested fix: state "mean above random, not significant; seed 0 loses" near the headline. The stronger result is decompose avoidance and direct/schema tie.

### P2 - Competitive-positioning sources are not reproducible enough

File: `docs/POSITIONING.md:94-107`

The source list is mostly bare domains/arXiv IDs, not exact URLs, dates, or quoted evidence. The empty-market claim is high-drift and hard to prove.

Suggested fix: pin exact source URLs and access dates, add "to our knowledge", and add a row for newer online adaptive model routers while preserving the workflow-vs-model distinction.

### P2 - SQL reward intentionally ignores order

File: `src/reward.py:28-30`, `src/reward.py:53-55`

The current README says order-insensitive result-set comparison, so this is not a direct doc mismatch. But BIRD/SQL questions with meaningful ordering can be over-credited if generated and gold rows match as sets but in the wrong order.

Suggested fix: either keep the current set semantics and document the limitation, or detect `ORDER BY` in gold SQL and compare ordered normalized row lists in that case.

## Overclaimed vs Honestly Caveated

Overclaimed:

- "Safe Python execution" / "sandboxed subprocess" is false under adversarial code.
- `docs/VERIFICATION.md` says all claims hold with high confidence, but several are qualified.
- "No gold leakage" is too broad for the fake LLMs; the reward is real, but the fake model responses are scripted from reference/gold tables.
- The competitive "none does this" claim is too absolute for a moving product landscape.

Honestly caveated:

- README explicitly says the BIRD top two arms are statistically tied and headroom over best fixed arm is small.
- README keeps the core framing at selector-over-fixed-workflows, not self-rewriting agent.
- The fake examples disclose that arms are deliberately differentiated.
- The RL and contextual claims are clearly synthetic/offline demos.

## Most Embarrassing Reviewer Catch

The most embarrassing issue is the "sandboxed" Python code executor. A reviewer can bypass it in one line with `getattr(__builtins__, 'op' + 'en')('/etc/hosts')`, and the executor returns `(True, 'ok')`. Fix this before submission or remove all safety/sandbox language. For a course demo, a clear "toy subprocess executor, not a security sandbox" caveat is acceptable; for an SDK claim, it needs real isolation.
