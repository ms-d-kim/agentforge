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
> `{{MODEL e.g. openai/gpt-4o-mini}}`. Temperature 0, seeded RNG, and every LLM response is cached
> so runs are reproducible and cheap."

## [1:50–3:10] Does it work? (Q3 framing via evidence — the headline)

> "Here's the learning curve, averaged over {{N_SEEDS}} seeds."

*Visual:* `plots/exp_01_cumulative_reward.png`.

> "The bandit (blue) climbs toward the oracle and beats the random baseline and the single-arm
> baselines. The smoke test confirmed the arms are genuinely differentiated — per-workflow accuracy
> was {{direct%}} / {{explore%}} / {{decompose%}} — so there's something real to learn."

*Visual:* `plots/exp_02_arm_fractions.png` and `plots/exp_04_arm_means.png`.

> "Selection mass shifts toward the best arm over time, and the per-arm value estimates separate.
> Cumulative regret against the oracle grows **sub-linearly** — the signature of a learning bandit."

*Visual:* `plots/exp_03_cumulative_regret.png` + the per-arm summary table (Wilson CIs).

> "Honest limitations: BIRD's gold SQL has documented label noise, 60 episodes is a short horizon
> so the curves are noisy, and {{the best single arm is strong — the bandit's job is to *find* it
> without knowing it in advance}}."

## [3:10–3:50] Use cases + impact (Q3) and what's next (Q4)

> "Why it matters: this is a **learned policy at the workflow-selection layer** — a layer that's
> normally hardcoded. The closest commercial analogs are LLM routers like RouteLLM, but they swap a
> single model; AgentForge selects among multi-step *workflows*. Any agent system with a library of
> strategies could drop this in to stop hand-tuning."

> "What I'd add next (Q4): a cost-adjusted reward (accuracy per dollar), a contextual bandit that
> conditions on task features, and the longer-term vision — making the workflows themselves
> learnable objects improved from expert demonstrations, with the bandit selecting over the evolving
> library."

## [3:50–4:00] Close

> "AgentForge: don't hand-pick your agent's workflow — let it learn. Repo and README are linked
> below. Thanks for watching."

---

### Recording checklist
- [ ] Repo is public; README renders the four plots and the AI-use disclosure.
- [ ] Live episode runs on camera (have `.env` key ready; it'll be a cache hit, so instant).
- [ ] All four plots + summary table visible at full resolution.
- [ ] State the model, seed count, episode count, and dataset explicitly.
- [ ] Mention AI tool usage verbally (Claude Code) — it's a rubric item.
