"""The three fixed workflow arms behind a uniform interface (spec §7).

Import the registry helper to get arm instances in stable order.
"""
from __future__ import annotations

from .base import Workflow, WorkflowResult
from .direct import DirectWorkflow
from .explore import SchemaExploreWorkflow
from .decompose import DecomposeWorkflow


def build_workflows() -> list[Workflow]:
    """Return the three arms in the stable order defined by config.ARM_NAMES."""
    return [DirectWorkflow(), SchemaExploreWorkflow(), DecomposeWorkflow()]


__all__ = [
    "Workflow",
    "WorkflowResult",
    "DirectWorkflow",
    "SchemaExploreWorkflow",
    "DecomposeWorkflow",
    "build_workflows",
]
