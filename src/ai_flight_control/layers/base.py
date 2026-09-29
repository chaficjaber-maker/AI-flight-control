from typing import Protocol, runtime_checkable

from ai_flight_control.state import AdaptationContext


@runtime_checkable
class AdaptivePolicy(Protocol):
    """
    AI-adaptive control building block.

    Reference implementations use explicit online rules; replace `step` / `adapt`
    internals with neural policies while keeping the same I/O types.
    """

    def reset(self) -> None: ...

    def adapt(self, context: AdaptationContext) -> None:
        """Update internal parameters (bounded online adaptation)."""
        ...
