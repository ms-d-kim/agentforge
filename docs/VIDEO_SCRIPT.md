# AgentForge — Demo Video Script (~4 min)

Target 3–5 min (10 is the cap, not a goal). Screen-record a terminal + the plots in `plots/`.
Structure follows the CS 153 rubric's required Q1–Q4. `{{...}}` = fill from your real run.

---

## [0:00–0:35] Cold open — the problem (Q1: why did you build this?)

> "LLM agents are stochastic, and different tasks want different *strategies*. For text-to-SQL,
> sometimes one prompt is enough; sometimes you need to explore the schema first; sometimes you
> need to decompose the question. Today, practitioners hand-pick one workflow per use case and
> freeze it. **The bottleneck I went after: nobody is *learning* which workflow to use.**
> AgentForge treats workflow selection as a multi-armed bandit — it learns online which workflow
> wins on which kind of task."

*Visual:* title slide → the three arms (direct / schema-explore / decompose).

## [0:35–1:50] How it works (Q2 — Automation/Agent Systems: the agent + data)

> "Three fixed workflows are the bandit's arms. An ε-greedy policy picks one per task, the chosen
> workflow generates SQL, and I get a **binary reward** by executing the candidate against the gold
> query on the real SQLite database — execution accuracy, not string match. The reward updates that
> arm's running mean. That's the whole loop, and it runs over the BIRD-SQL benchmark."

*Visual:* the architecture diagram (spec §5), then **run one episode live** in the terminal so the
viewer sees: arm selected → SQL generated → executed → reward → bandit means updated → JSONL line.

> "Everything is plain Python — no agent framework. Provider is OpenRouter; model is
> `openai/gpt-4o-mini` — deliberately mid-tier, so workflow *strategy* still matters. Temperature 0,
> seeded RNG, and every LLM response is cached, so runs are reproducible and cheap (~$1)."

## [1:50–3:10] Does it work? (Q3 framing via evidence — the headline)

> "Here's the learning curve, averaged over 3 seeds, 60 episodes each, on a balanced 60-task pool
> spanning four BIRD databases and all three difficulty levels."

*Visual:* `results/exp_01_cumulative_reward.png`.

> "The bandit (blue) climbs toward the oracle, clearly beats the random baseline, and sits just
> under the best single workflow. Per-workflow accuracy is 35% direct / 40% schema-explore /
> 28% decompose — so the arms are genuinely differentiated and there's something real to learn."

*Visual:* `results/exp_02_arm_fractions.png` and `results/exp_04_arm_means.png`.

> "Selection mass shifts off the weak `decompose` arm — from a third of pulls down to about 13% —
> and onto the two strong workflows, while the per-arm value estimates separate. Cumulative regret
> against the oracle stays small — the bandit tracks close to the oracle (60 episodes is too short to
> claim a sub-linear *rate*)."

*Visual:* `results/exp_03_cumulative_regret.png` + the per-arm summary table (Wilson CIs).

> "Honest limitations: BIRD's gold SQL has documented label noise; 60 episodes is a short horizon,
> so the curves are noisy; and here the two strongest arms — direct and schema-explore — are
> statistically tied (0.39 vs 0.41, overlapping confidence intervals). So the bandit's decisive win
> is learning *online* to avoid the weak arm, rather than separating two near-identical ones — and
> it still lands close to the best fixed policy and the oracle without being told the ranking."

## [3:10–4:10] It's reusable — use it on your own agent (Q3: impact) + what's next (Q4)

> "Here's the important part — this isn't SQL-specific. The selector is a domain-agnostic layer: bring
> your own strategies and any reward signal. To prove it, here's the *same* selector — no SQL, no BIRD
> — learning which *prompting strategy* wins on multi-step word problems."

*Visual:* run `python -m examples.byo_agent` live — it converges to `cot` (1.00) over `direct` (0.33),
concentrating ~27 of 30 pulls on the winner.

> "That's the whole pitch: `pip install`, wrap your agent's strategies in `WorkflowSelector`, give it
> a signal — tests passing, an eval score, a thumbs-up — and it self-improves, concentrating on what
> works and dropping what doesn't, persisting across runs. The closest analogs are LLM routers like
> RouteLLM, but they swap a single *model*; this selects among multi-step *workflows* or tool-use
> patterns. Any one-person lab building an agent can drop it in instead of hand-tuning each turn."

> "What I'd add next (Q4): a cost-adjusted reward (accuracy per dollar); a *contextual* bandit that
> conditions on task features, so it picks per-task instead of on average; and the longer-term vision —
> making the workflows themselves learnable objects improved from expert demonstrations, with the
> bandit selecting over the evolving library."

## [3:50–4:00] Close

> "AgentForge: don't hand-pick your agent's workflow — let it learn. Repo and README are linked
> below. Thanks for watching."

---

### Recording checklist
- [ ] Repo is public; README renders the four plots.
- [ ] Live episode runs on camera (have `.env` key ready; it'll be a cache hit, so instant).
- [ ] All four plots + summary table visible at full resolution.
- [ ] State the model, seed count, episode count, and dataset explicitly.
