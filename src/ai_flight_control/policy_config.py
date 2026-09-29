from dataclasses import dataclass
from typing import Literal

PolicyMode = Literal["reference", "neural"]


@dataclass
class PolicyConfig:
    """Which layer uses reference rules vs trained NumPy MLP policies."""

    stability: PolicyMode = "reference"
    maneuver: PolicyMode = "reference"
    mission: PolicyMode = "reference"

    @staticmethod
    def all_neural() -> "PolicyConfig":
        return PolicyConfig(stability="neural", maneuver="neural", mission="neural")

    @staticmethod
    def rollout_order() -> "PolicyConfig":
        """Stability only (first in user rollout order)."""
        return PolicyConfig(stability="neural", maneuver="reference", mission="reference")
