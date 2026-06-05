"""Domain instances of the AgentForge selector.

Each subpackage plugs a set of strategy "arms" + a reward into the domain-agnostic
WorkflowSelector (src/selector.py), demonstrating that the learned selection layer
generalizes across tasks: text-to-SQL (src/workflows), code generation, agentic
search, and context compaction.
"""
