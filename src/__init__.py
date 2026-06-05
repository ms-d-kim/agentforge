"""AgentForge — a self-improving workflow selector for text-to-SQL.

An epsilon-greedy multi-armed bandit learns, across repeated episodes, which of
three fixed SQL-generation workflows to apply to a given BIRD-SQL task. See
docs/agentforge_spec.md and CLAUDE.md for the load-bearing framing and scope.
"""
