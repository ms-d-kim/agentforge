# Full Adversarial Review — AgentForge

> A top-to-bottom adversarial pass: **five independent reviewers**, each owning a dimension and each
> *running* the code (falsification experiments, not just reading), then synthesized and verified.
> Complements the earlier author-run [VERIFICATION.md](VERIFICATION.md) and the independent
> [CODEX_AUDIT.md](CODEX_AUDIT.md). Findings marked **[FIXED]** were repaired in this same pass.

## Verdict

**The empirical work is sound and honestly disclosed; the math is correct; secrets are airtight; every
headline number reproduces.** The review surfaced **one real correctness bug** (persistence silently
dropped contextual/Thompson state — contradicting the "survives restarts" claim) and **one real
distribution defect** (wheel install was broken), both now fixed, plus a cluster of precision slips.

| Dimension | Reviewer grade | After fixes |
|---|---|---|
| Algorithms + RL correctness | B (math A; persistence bug) | **A−** (persistence fixed + tested) |
| Reward integrity + safety honesty | A− / A | **A** (int/float fixed) |
| Reproducibility | A− | **A−** (stale README number fixed) |
| Hygiene / security / packaging | A− (security A; packaging) | **A** (wheel + deps fixed) |
| Docs / artifact honesty | A− | **A−** (precision slips fixed) |

## Findings (consolidated by severity)

**[P0 · FIXED] LinUCB & Thompson persistence silently dropped learned state.** `save()`/`load()` only
round-tripped `means`/`counts`; the LinUCB `A`/`b` matrices and the Thompson `α`/`β` posterior were
discarded, so `policy="linucb"`/`"thompson"` did **not** resume across runs — falsifying the
repeatedly-advertised "persists across runs / survives restarts" claim for exactly the policies whose
value depends on it. The one persistence test only exercised epsilon-greedy, hiding it.
*Fix:* added `restore()` to every policy; `LinUCBBandit.snapshot()` now serializes `A`/`b`; `save`/`load`
persist/restore full state; added round-trip tests for `linucb` (asserts the contextual A→a0 / B→a1
mapping survives reload) and `thompson` (asserts the posterior survives). Verified.

**[P1 · FIXED] Wheel install was broken; `import agentforge` crashed off a non-editable install.**
`pyproject.toml` `packages` omitted `src.rl` and `src.domains.*`, which `agentforge/__init__` imports.
The advertised `pip install` → `from agentforge import WorkflowSelector` worked only via `-e`.
*Fix:* added the missing subpackages; verified a non-editable target install now imports `agentforge`
(+ `REINFORCE`) cleanly.

**[P1 · FIXED] `numpy` was declared a `dev` extra but is core** (LinUCB + REINFORCE import it) — masked
only by matplotlib's transitive pin. *Fix:* moved to core `dependencies`. Also reconciled the version
(`pyproject` `0.1.0` → `0.3.0`, matching `__init__`).

**[P1 · FIXED] Reward int-vs-float false mismatch.** `_norm_cell(1)`→`'1'` but `_norm_cell(1.0)`→`'1.0'`,
so a correct REAL-typed answer (e.g. `CAST(COUNT(*) AS REAL)`) was under-credited vs an integer gold —
contradicting the "no false mismatches" docstring. (Zero impact on the logged BIRD data; latent.)
*Fix:* normalize int-valued floats by value (`1 == 1.0`), still rounding non-integers.

**[P2 · FIXED] Doc precision slips a reviewer would catch:** the README `byo_agent` output block showed
an impossible `direct: 0.33 (n=3)` (the fake never lets `direct` succeed → it's always `0.00`); "Agent
Lightning" was cited with GEPA's arXiv ID (now `2508.03680`); the VERIFICATION table cells said "HOLD"
while its own banner said read them as QUALIFIED (cells reconciled to QUALIFIED); "regret grows
sub-linearly" overstated a noisy 60-episode curve (softened to "stays small, ≈5"); test counts were stale
(64→67, 17→18); the video now narrates the AI-use disclosure (was README-only); CLAUDE.md's MVP
scope-discipline list now notes which items v2 shipped.

**[P3 · accepted/noted, not fixed]** ORDER BY detection is textual (matches a subquery/CTE/string-literal
`ORDER BY`) — 0 impact on logged data; documented limitation. REINFORCE applies its gradient in-place
mid-trajectory (deviates from fixed-θ batch form by ~3e-3 at lr=0.5; harmless, unbiased). Direct bandit
constructors don't guard `n_arms==0` (the public `WorkflowSelector` does). Test-coverage gaps remain on
`experiment.py` (oracle/baselines), `llm.py` real-provider clients + cache, and the plotting/IO scripts —
acceptable for a course project but the biggest place to harden next.

## What's solid (verified by falsification, not assertion)

- **All estimators are mathematically correct.** LinUCB `A=I+Σxxᵀ, b=Σrx, score=θᵀx+α√(xᵀA⁻¹x)` matched a
  shadow computation exactly; Thompson/UCB1/ε-greedy matched hand calculations; **REINFORCE is a true
  score-function gradient** (analytic vs numerical match 8e-11; 200k-sample Monte-Carlo matched ∇J).
- **No gold leakage into rewards, no arm-name rigging.** Re-executing **all 1080** logged BIRD SQL pairs
  against the real DBs → **0 mismatches**. Fakes inject reference/gold only into the arm's I/O (disclosed);
  every reward is computed from the produced artifact (real subprocess / grep / fact-retention).
- **SQL executor is genuinely read-only** (DROP/UPDATE/INSERT all rejected; multi-statement blocked).
- **Code executor is now honestly labeled** ("NOT A SECURITY SANDBOX", names the `getattr(open)` bypass,
  notes macOS doesn't enforce the limits) — claims match reality.
- **Secrets are airtight** — `.env` untracked, full git history clean of every key shape, `.env.example`
  placeholders only.
- **Reproducibility is strong** — LinUCB 1.0 on 10/10 seeds; REINFORCE beats fixed in 10/10; sweep
  regenerates byte-identical; BIRD numbers exact; figures faithfully render their source data.

## Honest residual limitations (disclosed, not bugs)

The BIRD bandit-vs-random win is a 3-seed mean and **loses in seed 0**; the two top arms are statistically
tied; the `--fake` domain demos are **scripted controlled demonstrations** (mechanics, not proof the SDK
discovers strategy quality in arbitrary domains); the contextual/RL results are **synthetic stand-ins**;
the code executor is **not** a real sandbox. All are stated plainly across the README/VERIFICATION/CODEX
docs. The competitive "white-space" claim is scoped "to our knowledge (June 2026)" with counterexamples
invited.
