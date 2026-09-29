from typing import Protocol

from ai_flight_control.dynamics import LongitudinalState


class FlightAgent(Protocol):
    """Control policy interface; swap PID for learned policies later."""

    def reset(self) -> None: ...

    def act(
        self,
        state: LongitudinalState,
        target_altitude_m: float,
        dt: float,
    ) -> float:
        """Return normalized thrust command in [-1, 1]."""
        ...
